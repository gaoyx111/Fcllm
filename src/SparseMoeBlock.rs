use crate::MLP::Qwen2MoeMLP;
use crate::configuration_qwen::Qwen2MoeConfig;
use crate::expert_ARC_cahce::ARC_Cache;
use crate::linear::new_uninitialized_linear;
use crate::load::load_linear_from_files;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

const MIN_DEQUANT_CACHE_FOR_FULLY_OFFLOADED_LAYER: usize = 2;

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
        let compressed_num_in_mem = num_experts - cfg.offload_map[&layer_idx];
        let expert_quan_bit = cfg.quan_map[&layer_idx];
        let num_in_mem = dequant_cache_capacity(compressed_num_in_mem, expert_quan_bit);

        let arc_cache = ARC_Cache::new(num_in_mem);

        // gate 线性层，没有 bias
        let gate = new_uninitialized_linear(cfg.hidden_size, cfg.num_experts, false, &device)?;

        // experts 逐个 new Qwen2MoeMLP
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
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

    pub fn prewarm_cpu_dequant_cache(&mut self) -> Result<usize> {
        let mut warmed = 0;
        for expert in &mut self.experts {
            if expert.warm_dequant_cpu_cache()? {
                warmed += 1;
            }
        }
        Ok(warmed)
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
        } else {
            if self.arc_cache.is_evicted(expert_idx) {
                self.experts[expert_idx].clear();
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

        if batch_size * seq_len == 1 && prefetch_expert_idx.is_none() {
            let (routing_weights, expert_ids) =
                topk_routing_single_token(&router_logits, self.top_k, self.norm_topk_prob)?;
            let final_hidden_states =
                self.forward_single_token(&hidden_states, &routing_weights, &expert_ids)?;
            let shared_expert_out = self.shared_expert.forward(&hidden_states)?;
            let gate_out =
                candle_nn::ops::sigmoid(&self.shared_expert_gate.forward(&hidden_states)?)?;
            let shared = shared_expert_out.broadcast_mul(&gate_out)?;

            let output = (final_hidden_states + shared)?;
            return output.reshape(&[batch_size, seq_len, hidden_dim]);
        }

        let (routing_weights, selected_experts) =
            topk_routing(&router_logits, self.top_k, self.norm_topk_prob)?;
        let routing_weights = routing_weights.to_dtype(hidden_states.dtype())?;
        let selected_experts = selected_experts.to_dtype(DType::I64)?;

        let mut final_hidden_states = Tensor::zeros(
            &[batch_size * seq_len, hidden_dim],
            hidden_states.dtype(),
            hidden_states.device(),
        )?;

        let (assignments_by_expert, expert_index, expert_order) =
            build_expert_assignments(&selected_experts)?;
        let mut load_experts = vec![];
        let using_external_prefetch = prefetch_expert_idx.is_some();

        let prefetch_expert_idx = match prefetch_expert_idx {
            Some(v) => v.to_vec(),
            None => expert_order,
        };

        if using_external_prefetch {
            let prefetched: HashSet<usize> = prefetch_expert_idx.iter().copied().collect();
            for &idx in assignments_by_expert.keys() {
                if !prefetched.contains(&idx) {
                    self.load_weights(Idx::Single(idx), true, None)?;
                    load_experts.push(idx);
                }
            }
        }

        if self.num_in_mem != 0 {
            let evicted = self.arc_cache.update_list(&expert_index);
            for idx in evicted {
                self.experts[idx].clear();
            }
        }

        // Run computation for prefetched experts
        for &expert_idx in &prefetch_expert_idx {
            let Some(assignments) = assignments_by_expert.get(&expert_idx) else {
                if self.arc_cache.is_evicted(expert_idx) {
                    self.experts[expert_idx].clear();
                }
                continue;
            };

            self.experts[expert_idx].dequan_experts()?;
            let expert_layer = &mut self.experts[expert_idx];

            let (idx, top_x) = split_expert_assignments(assignments);
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

                let Some(assignments) = assignments_by_expert.get(&expert_idx) else {
                    continue;
                };
                let (idx, top_x) = split_expert_assignments(assignments);
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
        let shared = shared_expert_out.broadcast_mul(&gate_out)?;

        let output = (final_hidden_states + shared)?;
        output.reshape(&[batch_size, seq_len, hidden_dim])
    }

    fn forward_single_token(
        &mut self,
        hidden_states: &Tensor,
        routing_weights: &[f32],
        expert_ids: &[usize],
    ) -> Result<Tensor> {
        if self.num_in_mem != 0 {
            let evicted = self.arc_cache.update_list(expert_ids);
            for idx in evicted {
                self.experts[idx].clear();
            }
        }

        let profile_moe = moe_profile_enabled();
        let mut final_hidden_states: Option<Tensor> = None;
        for (topk_idx, &expert_idx) in expert_ids.iter().enumerate() {
            let expert_output = if profile_moe {
                let cached_before = self.experts[expert_idx].has_dequantized_weights_on_device();
                let dequant_start = Instant::now();
                self.experts[expert_idx].dequan_experts()?;
                let dequant_elapsed = dequant_start.elapsed();
                let forward_start = Instant::now();
                let expert_output = self.experts[expert_idx].forward(hidden_states)?;
                let forward_elapsed = forward_start.elapsed();
                eprintln!(
                    "[moe-profile] layer={} expert={} cached_before={} dequant_ms={:.2} forward_ms={:.2}",
                    self.layer_idx,
                    expert_idx,
                    cached_before,
                    dequant_elapsed.as_secs_f64() * 1000.0,
                    forward_elapsed.as_secs_f64() * 1000.0
                );
                expert_output
            } else {
                self.experts[expert_idx].dequan_experts()?;
                self.experts[expert_idx].forward(hidden_states)?
            };
            let weighted = expert_output.affine(routing_weights[topk_idx] as f64, 0.0)?;
            final_hidden_states = Some(match final_hidden_states {
                Some(acc) => (acc + weighted)?,
                None => weighted,
            });
            self.post_comp(expert_idx)?;
        }

        final_hidden_states
            .ok_or_else(|| candle_core::Error::Msg("single-token MoE selected no experts".into()))
    }
}

pub enum Idx {
    Single(usize),
    Multiple(Vec<usize>),
}

fn moe_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FCLLM_PROFILE_MOE").is_some())
}

fn dequant_cache_capacity(compressed_cache_capacity: usize, quant_bit: usize) -> usize {
    if quant_bit == 0 {
        return compressed_cache_capacity;
    }
    if compressed_cache_capacity == 0 {
        return MIN_DEQUANT_CACHE_FOR_FULLY_OFFLOADED_LAYER;
    }

    // A dequantized BF16 expert is roughly 16 / quant_bit times larger than
    // the packed expert. Keep the resident cache inside the old packed-weight
    // memory envelope, then let ARC choose the hottest experts.
    ((compressed_cache_capacity * quant_bit) / 16)
        .max(1)
        .min(compressed_cache_capacity)
}

type ExpertAssignments = HashMap<usize, Vec<(usize, usize)>>;

fn build_expert_assignments(
    selected_experts: &Tensor,
) -> Result<(ExpertAssignments, Vec<usize>, Vec<usize>)> {
    let selected = selected_experts.to_vec2::<i64>()?;
    let mut assignments: ExpertAssignments = HashMap::new();
    let mut flat_experts = Vec::new();
    let mut expert_order = Vec::new();

    for (token_idx, row) in selected.iter().enumerate() {
        for (topk_idx, &expert_idx) in row.iter().enumerate() {
            let expert_idx = expert_idx as usize;
            flat_experts.push(expert_idx);

            match assignments.entry(expert_idx) {
                Entry::Vacant(entry) => {
                    expert_order.push(expert_idx);
                    entry.insert(vec![(topk_idx, token_idx)]);
                }
                Entry::Occupied(mut entry) => {
                    entry.get_mut().push((topk_idx, token_idx));
                }
            }
        }
    }

    Ok((assignments, flat_experts, expert_order))
}

fn split_expert_assignments(assignments: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    assignments.iter().copied().unzip()
}

fn topk_routing_rows(
    logits_rows: &[Vec<f32>],
    k: usize,
    norm: bool,
    experts: usize,
) -> Result<(Vec<f32>, Vec<i64>)> {
    if k == 0 || k > experts {
        return Err(candle_core::Error::Msg(format!(
            "invalid top-k {k} for {experts} experts"
        )));
    }

    let mut all_weights = Vec::with_capacity(logits_rows.len() * k);
    let mut all_indices = Vec::with_capacity(logits_rows.len() * k);

    for logits in logits_rows {
        let (weights, top) = topk_routing_row(logits, k, norm, experts)?;
        all_weights.extend(weights);
        all_indices.extend(top.into_iter().map(|(idx, _)| idx as i64));
    }

    Ok((all_weights, all_indices))
}

fn topk_routing_row(
    logits: &[f32],
    k: usize,
    norm: bool,
    experts: usize,
) -> Result<(Vec<f32>, Vec<(usize, f32)>)> {
    if logits.len() != experts {
        return Err(candle_core::Error::Msg(format!(
            "routing logits row has {} experts, expected {experts}",
            logits.len()
        )));
    }

    let top = select_topk_desc(logits, k);
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let denom: f32 = logits.iter().map(|value| (*value - max_logit).exp()).sum();
    let mut weights: Vec<f32> = top
        .iter()
        .map(|&(_, value)| (value - max_logit).exp() / denom)
        .collect();
    if norm {
        let top_sum: f32 = weights.iter().sum();
        if top_sum != 0.0 {
            for weight in &mut weights {
                *weight /= top_sum;
            }
        }
    }

    Ok((weights, top))
}

fn topk_routing_single_token(
    routing_logits: &Tensor,
    k: usize,
    norm: bool,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let (tokens, experts) = routing_logits.dims2()?;
    if tokens != 1 {
        return Err(candle_core::Error::Msg(format!(
            "single-token routing expected 1 token, got {tokens}"
        )));
    }

    let logits = routing_logits
        .to_dtype(DType::F32)?
        .reshape((experts,))?
        .to_vec1::<f32>()?;
    let (weights, top) = topk_routing_row(&logits, k, norm, experts)?;
    Ok((weights, top.into_iter().map(|(idx, _)| idx).collect()))
}

fn select_topk_desc(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (idx, &value) in values.iter().enumerate() {
        let insert_at = top
            .iter()
            .position(|&(_, existing)| value > existing)
            .unwrap_or(top.len());
        if insert_at < k {
            top.insert(insert_at, (idx, value));
            if top.len() > k {
                top.pop();
            }
        }
    }
    top
}

pub fn topk_routing(routing_logits: &Tensor, k: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    let (tokens, experts) = routing_logits.dims2()?;
    let logits = routing_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let (weights, indices) = topk_routing_rows(&logits, k, norm, experts)?;
    let device = routing_logits.device();

    Ok((
        Tensor::from_vec(weights, (tokens, k), device)?,
        Tensor::from_vec(indices, (tokens, k), device)?,
    ))
}

pub fn select(t: &Tensor, indices: &[usize]) -> Result<Tensor> {
    let idx: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let idx_tensor = Tensor::from_vec(idx, indices.len(), t.device())?;
    t.index_select(&idx_tensor, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_cache_capacity_stays_within_packed_budget() {
        assert_eq!(dequant_cache_capacity(0, 4), 2);
        assert_eq!(dequant_cache_capacity(45, 4), 11);
        assert_eq!(dequant_cache_capacity(60, 4), 15);
        assert_eq!(dequant_cache_capacity(60, 2), 7);
        assert_eq!(dequant_cache_capacity(60, 0), 60);
    }

    #[test]
    fn expert_assignments_preserve_topk_and_first_seen_order() {
        let selected =
            Tensor::from_vec(vec![3i64, 1, 3, 2, 1, 4], (3, 2), &Device::Cpu).expect("tensor");
        let (assignments, flat, order) = build_expert_assignments(&selected).expect("assignments");

        assert_eq!(flat, vec![3, 1, 3, 2, 1, 4]);
        assert_eq!(order, vec![3, 1, 2, 4]);
        assert_eq!(assignments.get(&3).unwrap(), &vec![(0, 0), (0, 1)]);
        assert_eq!(assignments.get(&1).unwrap(), &vec![(1, 0), (0, 2)]);
        assert_eq!(assignments.get(&2).unwrap(), &vec![(1, 1)]);
        assert_eq!(assignments.get(&4).unwrap(), &vec![(1, 2)]);
    }

    #[test]
    fn single_token_cpu_topk_keeps_descending_order() {
        let ids = select_topk_desc(&[0.1, 3.0, -1.0, 2.5, 0.7], 3)
            .into_iter()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();

        assert_eq!(ids, vec![1, 3, 4]);
    }

    #[test]
    fn single_token_routing_matches_softmax_topk_normalization() -> Result<()> {
        let logits = Tensor::from_vec(vec![0.1f32, 3.0, -1.0, 2.5, 0.7], (1, 5), &Device::Cpu)?;
        let (weights, ids) = topk_routing_single_token(&logits, 3, true)?;

        assert_eq!(ids, vec![1, 3, 4]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        Ok(())
    }

    #[test]
    fn topk_routing_handles_multiple_tokens_without_full_sort() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![0.1f32, 3.0, -1.0, 2.5, 0.7, 2.0, -1.0, 0.0, 0.5, 4.0],
            (2, 5),
            &Device::Cpu,
        )?;
        let (weights, ids) = topk_routing(&logits, 2, true)?;

        assert_eq!(ids.to_vec2::<i64>()?, vec![vec![1, 3], vec![4, 0]]);
        for row in weights.to_vec2::<f32>()? {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(row[0] > row[1]);
        }
        Ok(())
    }
}
