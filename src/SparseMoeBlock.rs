use crate::MLP::Qwen2MoeMLP;
use crate::configuration_qwen::Qwen2MoeConfig;
use crate::expert_ARC_cahce::ARC_Cache;
use crate::linear::new_uninitialized_linear;
use crate::load::load_linear_from_files;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::encoding::one_hot;
use candle_nn::{Linear, Module};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Qwen2MoeSparseMoeBlock {
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub layer_idx: usize,
    pub device: Device,
    pub num_in_mem: usize,
    pub arc_cache: ARC_Cache, // 自定义 ARC_Cache 类型
    pub gate: Linear,         // Candle 的线性层，没有 bias
    pub experts: Vec<Qwen2MoeMLP>,
    pub shared_expert: Qwen2MoeMLP,
    pub shared_expert_gate: Linear,
}

impl Qwen2MoeSparseMoeBlock {
    pub fn new(cfg: &Qwen2MoeConfig, layer_idx: usize) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let top_k = cfg.num_experts_per_tok;
        let norm_topk_prob = cfg.norm_topk_prob;
        let device = cfg.device.clone();
        let num_in_mem = num_experts - cfg.offload_map[&layer_idx];

        let arc_cache = ARC_Cache::new(num_in_mem);

        // gate 线性层，没有 bias
        let gate = new_uninitialized_linear(cfg.hidden_size, cfg.num_experts, false, &device)?;

        // experts 逐个 new Qwen2MoeMLP
        let mut experts = Vec::with_capacity(num_experts);
        for idx in 0..num_experts {
            let expert = Qwen2MoeMLP::new(cfg, layer_idx, false);
            experts.push(expert);
        }

        let shared_expert = Qwen2MoeMLP::new(cfg, layer_idx, true);

        let shared_expert_gate = new_uninitialized_linear(cfg.hidden_size, 1, false, &device)?;

        Ok(Self {
            num_experts,
            top_k,
            norm_topk_prob,
            layer_idx,
            device,
            num_in_mem,
            arc_cache,
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        let base = PathBuf::from(path);
        let original_dir = base.join("original");

        // gate 权重路径
        let gate_path =
            original_dir.join(format!("model.layers.{}.mlp.gate.weight", self.layer_idx));
        // let gate_loader = ExpertTensorLoader::new(gate_path);
        // let gate_weight = gate_loader.load_tensor("tensor", &self.device)?;
        self.gate = load_linear_from_files(gate_path.to_str().unwrap(), None, &self.device)?;

        // shared_expert_gate 权重路径
        let shared_expert_gate_path = original_dir.join(format!(
            "model.layers.{}.mlp.shared_expert_gate.weight",
            self.layer_idx
        ));
        // let shared_expert_gate_loader = ExpertTensorLoader::new(shared_expert_gate_path);
        // let shared_expert_gate_weight = shared_expert_gate_loader.load_tensor("tensor", &self.device)?;
        self.shared_expert_gate = load_linear_from_files(
            shared_expert_gate_path.to_str().unwrap(),
            None,
            &self.device,
        )?;

        // experts 权重初始化，传入 idx 和 num_in_mem
        for idx in 0..self.num_experts {
            self.experts[idx].init_weights(path, Some(idx), Some(self.num_in_mem))?;
        }

        // shared_expert 权重初始化，不传 idx 和 num_in_mem，调用 shared_expert 的 init_weights
        self.shared_expert.init_weights(path, None, None)?;

        Ok(())
    }

    pub fn load_weights(
        &mut self,
        idx: Idx,
        is_now: bool,
        int2_experts: Option<&HashSet<usize>>,
    ) -> Result<()> {
        match idx {
            Idx::Single(i) => {
                if self.arc_cache.is_evicted(i) {
                    // 单个专家固定 nbit=2
                    self.experts[i].load_weights(is_now, Some(2))?;
                }
            }
            Idx::Multiple(idxs) => {
                for i in idxs {
                    if self.arc_cache.is_evicted(i) {
                        let nbit = if int2_experts.map_or(false, |set| set.contains(&i)) {
                            2
                        } else {
                            4
                        };
                        self.experts[i].load_weights(is_now, Some(nbit))?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn post_comp(&mut self, expert_idx: usize) -> Result<()> {
        if self.num_in_mem == 0 {
            self.experts[expert_idx].clear();
        } else if self.layer_idx != 0 {
            if self.arc_cache.is_evicted(expert_idx) {
                self.experts[expert_idx].clear();
            } else {
                self.experts[expert_idx].quan_experts()?;
            }
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        prefetch_expert_idx: Option<&[usize]>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let hidden_states = hidden_states.reshape(&[batch_size * seq_len, hidden_dim])?;

        // gate score & top-k
        let router_logits = self.gate.forward(&hidden_states)?; // shape: (B*S, n_experts)
        let (routing_weights, selected_experts) =
            topk_routing(&router_logits, self.top_k, self.norm_topk_prob)?;
        let routing_weights = routing_weights.to_dtype(hidden_states.dtype())?;

        let mut final_hidden_states = Tensor::zeros(
            &[batch_size * seq_len, hidden_dim],
            hidden_states.dtype(),
            hidden_states.device(),
        )?;

        let selected_experts = selected_experts.to_dtype(DType::I64)?;
        let one_hot_encoded = one_hot(
            selected_experts.clone(),
            self.num_experts,
            1f32, // on_value
            0f32, // off_value
        )?; // shape: [bsz*seqlen*top_k, num_experts]
        let selected_shape = selected_experts.dims(); // [bsz*seqlen, top_k]
        let mut expert_mask = one_hot_encoded.reshape(&[
            selected_shape[0], // bsz*seqlen
            selected_shape[1], // top_k
            self.num_experts,
        ])?; // [bsz*seqlen, top_k, num_experts]
        expert_mask = expert_mask.permute((2, 1, 0))?;

        let expert_index: Vec<usize> = selected_experts
            .flatten_all()?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let mut load_experts = vec![];

        let prefetch_expert_idx = match prefetch_expert_idx {
            Some(v) => v.to_vec(),
            None => {
                let unique: HashSet<_> = expert_index.iter().copied().collect();
                unique.into_iter().collect()
            }
        };

        if self.layer_idx == 0 || prefetch_expert_idx.is_empty() {
            // do nothing extra
        } else {
            let mut counter = HashMap::new();
            for &idx in &expert_index {
                *counter.entry(idx).or_insert(0usize) += 1;
            }
            let mut freq_counter: Vec<_> = counter.into_iter().collect();
            freq_counter.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

            for (idx, _) in freq_counter {
                if !prefetch_expert_idx.contains(&idx) {
                    self.load_weights(Idx::Single(idx), true, None)?; // is_now=true
                    load_experts.push(idx);
                }
            }

            if self.num_in_mem != 0 {
                let evicted = self.arc_cache.update_list(&expert_index);
                for idx in evicted {
                    self.experts[idx].clear();
                }
            }
        }

        // Run computation for prefetched experts
        for &expert_idx in &prefetch_expert_idx {
            if !expert_index.contains(&expert_idx) {
                if self.arc_cache.is_evicted(expert_idx) {
                    self.experts[expert_idx].clear();
                }
                continue;
            }

            self.experts[expert_idx].dequan_experts()?;
            let expert_layer = &mut self.experts[expert_idx];

            let (idx, top_x) = index_positions(&expert_mask, expert_idx)?;
            let n_tokens = top_x.len();
            let current_state = select(&hidden_states, &top_x)?;

            let top_x_u32: Vec<u32> = top_x.iter().map(|&x| x as u32).collect();
            let row_idx = Tensor::from_vec(top_x_u32, n_tokens, routing_weights.device())?;
            let idx_u32: Vec<u32> = idx.iter().map(|&x| x as u32).collect();
            let col_idx = Tensor::from_vec(idx_u32, (n_tokens, 1), routing_weights.device())?;
            let weights = routing_weights
                .index_select(&row_idx, 0)?
                .gather(&col_idx, 1)?;

            let expert_output = expert_layer.forward(&current_state)?;
            let weights = weights
                .to_dtype(expert_output.dtype())?
                .broadcast_as(expert_output.shape())?;
            let current_hidden_states = (expert_output * weights)?;

            let top_x_i64: Vec<i64> = top_x.iter().map(|&x| x as i64).collect();
            let top_x_tensor = Tensor::from_vec(top_x_i64, n_tokens, &self.device)?;
            final_hidden_states = final_hidden_states.index_add(
                &top_x_tensor,
                &current_hidden_states.to_dtype(hidden_states.dtype())?,
                0,
            )?;
            self.post_comp(expert_idx)?;
        }

        if !load_experts.is_empty() {
            for &expert_idx in &load_experts {
                self.experts[expert_idx].dequan_experts()?;
                let expert_layer = &mut self.experts[expert_idx];

                let (idx, top_x) = index_positions(&expert_mask, expert_idx)?;
                let n_tokens = top_x.len();
                let current_state = select(&hidden_states, &top_x)?;

                let top_x_u32: Vec<u32> = top_x.iter().map(|&x| x as u32).collect();
                let row_idx = Tensor::from_vec(top_x_u32, n_tokens, routing_weights.device())?;
                let idx_u32: Vec<u32> = idx.iter().map(|&x| x as u32).collect();
                let col_idx = Tensor::from_vec(idx_u32, (n_tokens, 1), routing_weights.device())?;
                let weights = routing_weights
                    .index_select(&row_idx, 0)?
                    .gather(&col_idx, 1)?;

                let expert_out = expert_layer.forward(&current_state)?;
                let weights = weights
                    .to_dtype(expert_out.dtype())?
                    .broadcast_as(expert_out.shape())?;
                let current_hidden_states = (expert_out * weights)?;

                let top_x_i64: Vec<i64> = top_x.iter().map(|&x| x as i64).collect();
                let top_x_tensor = Tensor::from_vec(top_x_i64, n_tokens, &self.device)?;
                final_hidden_states = final_hidden_states.index_add(
                    &top_x_tensor,
                    &current_hidden_states.to_dtype(hidden_states.dtype())?,
                    0,
                )?;
                self.post_comp(expert_idx)?;
            }
        }

        // shared expert
        let shared_expert_out = self.shared_expert.forward(&hidden_states)?;
        let gate_out = candle_nn::ops::sigmoid(&self.shared_expert_gate.forward(&hidden_states)?)?;
        let gate_out = gate_out.broadcast_as(shared_expert_out.shape())?;
        let shared = (gate_out * shared_expert_out)?;

        let output = (final_hidden_states + shared)?;
        output.reshape(&[batch_size, seq_len, hidden_dim])
    }
}

pub enum Idx {
    Single(usize),
    Multiple(Vec<usize>),
}

pub fn topk_routing(routing_logits: &Tensor, k: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    // step 1: softmax to get routing_weights
    //let routing_weights = candle_nn::ops::softmax_last_dim(routing_logits)?;
    let routing_weights = candle_nn::ops::softmax(&routing_logits, 1)?.to_dtype(DType::F32)?;

    // step 2: get top-k expert indices for each token
    let topk_indices = routing_weights
        .arg_sort_last_dim(false)? // descending sort
        .narrow(D::Minus1, 0, k)? // take top-k
        .contiguous()?; // ensure memory layout

    // step 3: gather routing weights for top-k experts
    let topk_weights = routing_weights.gather(&topk_indices, D::Minus1)?;

    // optional: normalize top-k weights
    let topk_weights = if norm {
        let sum = topk_weights.sum_keepdim(D::Minus1)?;
        topk_weights.broadcast_div(&sum)?
    } else {
        topk_weights
    };

    Ok((topk_weights, topk_indices))
}

/* pub fn one_hot(indices: &Tensor, num_classes: usize) -> Result<Tensor> {
    let (batch, top_k) = indices.dims2()?; // shape: [B*S, top_k]
    let device = indices.device();
    let indices = indices.to_dtype(DType::I64)?; // index_add 需要 I64 类型

    // 展开 indices 到一维：shape [B*S * top_k]
    let flat_indices = indices.flatten(0, 1)?;

    // 创建 shape 为 [B*S * top_k, num_classes] 的 one-hot 模板
    let mut one_hot = Tensor::zeros(&[batch * top_k, num_classes], DType::F32, device)?;

    // 构造行号 [0, 1, 2, ..., batch*top_k-1]
    let row_ids = Tensor::arange(0u32, (batch * top_k) as u32, device)?.to_dtype(DType::I64)?;

    // 拼成二维索引 [row, col] -> shape: [B*S*top_k, 2]
    let indices = Tensor::stack(&[&row_ids, &flat_indices], 1)?;

    // 每个位置加 1.0
    let updates = Tensor::ones(&[batch * top_k], DType::F32, device)?;

    // 使用 index_add 添加
    one_hot = one_hot.index_add(&indices, &updates, 0)?;

    // reshape 成 [B*S, top_k, num_experts]
    one_hot.reshape(&[batch, top_k, num_classes])
} */

pub fn index_positions(mask: &Tensor, expert_idx: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    // mask shape: [num_experts, top_k, B*S]
    let mask_slice = mask.get(expert_idx)?; // shape: [top_k, B*S]
    let (top_k, batch_seq) = mask_slice.dims2()?;
    let mask_data = mask_slice.to_vec2::<f32>()?;

    let mut idx = Vec::new();
    let mut top_x = Vec::new();

    for i in 0..top_k {
        for j in 0..batch_seq {
            if mask_data[i][j] > 0.0 {
                idx.push(i);
                top_x.push(j);
            }
        }
    }

    Ok((idx, top_x))
}

pub fn select(t: &Tensor, indices: &[usize]) -> Result<Tensor> {
    let idx: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_tensor = Tensor::from_vec(idx, indices.len(), t.device())?;
    t.index_select(&idx_tensor, 0)
}
