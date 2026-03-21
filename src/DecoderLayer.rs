use std::collections::HashMap;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, Linear};
use crate::configuration_qwen::Qwen2MoeConfig;
use crate::load::load_linear_from_files;
use std::path::PathBuf;
use crate::linear::new_uninitialized_linear;
use std::collections::HashSet;
use crate::RmsNorm::Qwen2MoeRMSNorm;
use crate::Attention::Qwen2MoeAttention;
use crate::SparseMoeBlock::{Qwen2MoeSparseMoeBlock, topk_routing, Idx};
use crate::Cache::Cache;


#[derive(Debug)]
pub struct Qwen2MoeDecoderLayer {
    pub config: Qwen2MoeConfig,
    pub hidden_size: usize,
    pub layer_idx: usize,
    pub self_attn: Qwen2MoeAttention,
    pub mlp: Qwen2MoeSparseMoeBlock,
    pub input_layernorm: Qwen2MoeRMSNorm,
    pub post_attention_layernorm: Qwen2MoeRMSNorm,
    pub next_gate_cpu: Linear,
}

impl Qwen2MoeDecoderLayer {
    pub fn new(
        cfg: &Qwen2MoeConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Qwen2MoeAttention::new(cfg, Some(layer_idx))?;

        let mlp = Qwen2MoeSparseMoeBlock::new(cfg, layer_idx)?;

        let input_layernorm = Qwen2MoeRMSNorm::new(cfg.hidden_size, cfg.device.clone(), cfg.rms_norm_eps);
        let post_attention_layernorm = Qwen2MoeRMSNorm::new(cfg.hidden_size, cfg.device.clone(), cfg.rms_norm_eps);

        // next_gate_cpu: Linear(hidden_size, num_experts), no bias, on CPU
        let next_gate_cpu = new_uninitialized_linear(cfg.hidden_size, cfg.num_experts, false, &Device::Cpu)?;

        Ok(Self {
            config: cfg.clone(),
            hidden_size: cfg.hidden_size,
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            next_gate_cpu,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        self.self_attn.init_weights(path)?;
        self.mlp.init_weights(path)?;

        let base = PathBuf::from(path);
        let original_dir = base.join("original");

        // input_layernorm 权重路径
        let input_ln_path = original_dir.join(format!(
            "model.layers.{}.input_layernorm.weight",
            self.layer_idx
        ));
        self.input_layernorm.init_weights(input_ln_path.to_str().unwrap())?;

        // post_attention_layernorm 权重路径
        let post_ln_path = original_dir.join(format!(
            "model.layers.{}.post_attention_layernorm.weight",
            self.layer_idx
        ));
        self.post_attention_layernorm
            .init_weights(post_ln_path.to_str().unwrap())?;

        // 如果不是最后一层，加载下一层 gate 的线性权重
        if self.layer_idx < self.config.num_hidden_layers - 1 {
            let next_gate_path = original_dir.join(format!(
                "model.layers.{}.mlp.gate.weight",
                self.layer_idx + 1
            ));
            self.next_gate_cpu = load_linear_from_files(next_gate_path.to_str().unwrap(), None, &self.config.device)?;
        }

        Ok(())
    }

    pub fn predict(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let hidden_states = hidden_states.reshape(&[batch_size * seq_len, hidden_dim])?;
        let router_logits = self.next_gate_cpu.forward(&hidden_states)?;

        let (_, selected_experts) = topk_routing(&router_logits, 5, false)?;
        Ok(selected_experts)
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<&mut Cache>,
        cache_position: Option<&Tensor>,
        prefetch_expert_list: Option<Vec<usize>>,
        next_layer: Option<&mut Qwen2MoeDecoderLayer>,
    ) -> Result<(Tensor, Option<Cache>, Option<Vec<usize>>)> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(&hidden_states)?;

        let (hidden_states, present_key_value) = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            cache_position,
        )?;
        let hidden_states = (hidden_states + residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let mut next_prefetch_expert_list: Option<Vec<usize>> = None;
        if self.layer_idx + 1 < self.config.num_hidden_layers
            && self.config.offload_map[&(self.layer_idx + 1)] != 0
        {
            // Step 1: clone 并转到 CPU（可异步）或当前设备
            let hidden_cpu = hidden_states.clone(); // 可以加 `.to(Device::Cpu)?` 如果需要明确转设备

            // Step 2: 预测 top-5 expert（每个 token）
            let selected_experts = self.predict(&hidden_cpu)?; // shape: [batch*seq, 5]
            let selected_flat = selected_experts.flatten(0, selected_experts.rank() - 1)?.to_dtype(DType::I64)?; // flatten 到 1D

            // Step 3: 统计频次
            let ids: Vec<usize> = selected_flat.to_vec1::<i64>()?.into_iter().map(|v| v as usize).collect();
            let mut counter = HashMap::<usize, usize>::new();
            for id in &ids {
                *counter.entry(*id).or_insert(0) += 1;
            }

            // Step 4: 选出按频次排序的专家列表
            let mut most_common: Vec<(usize, usize)> = counter.into_iter().collect();
            most_common.sort_by(|a, b| b.1.cmp(&a.1)); // 按频次降序

            let top_experts: Vec<usize> = most_common.iter().map(|(k, _)| *k).collect();

            // Step 5: 选择 int2 experts（累计覆盖 30% 的 token）
            let value_sum: usize = most_common.iter().map(|(_, v)| *v).sum();
            let target_sum = (value_sum as f32 * 0.3) as usize;
            let mut current_sum = 0;
            let mut int2_experts = Vec::new();
            for (key, value) in most_common.iter().rev() {
                if current_sum + value > target_sum {
                    break;
                }
                current_sum += value;
                int2_experts.push(*key);
            }

            // Step 6: 触发 next_layer 的专家权重加载（假设 next_layer 一定存在）
            if let Some(next) = next_layer {
                let int2_experts_set: HashSet<usize> = int2_experts.iter().copied().collect();
                next.mlp.load_weights(Idx::Multiple(top_experts.clone()), false, Some(&int2_experts_set))?;
            }

            // Step 7: 返回值构造
            next_prefetch_expert_list = Some(top_experts);
            drop(hidden_cpu); // 显式释放
        }

        // 当前层 MoE 执行
        let hidden_states = self.mlp.forward(&hidden_states, prefetch_expert_list.as_deref())?; // 支持 prefetch_expert_list

        let hidden_states = (hidden_states + residual)?;

        Ok((hidden_states, present_key_value.cloned(), next_prefetch_expert_list))
    }
}