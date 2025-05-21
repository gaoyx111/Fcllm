use candle_core::{Result, Tensor, DType, Device, Module};
use candle_nn::ops;
use std::collections::{HashSet, HashMap};
use tokenizers::Tokenizer;


pub fn causal_mask_with_cache_position(
    attention_mask: Option<&Tensor>,
    seq_len: usize,
    target_len: usize,
    dtype: DType,
    device: &candle_core::Device,
    min_value: f64,
    cache_position: &Tensor,
    batch_size: usize,
) -> Result<Tensor> {
    let mut mask = Tensor::full((seq_len, target_len), min_value, dtype, device)?;
    if seq_len > 1 {
        let triu = mask.triu(1)?;
        mask = triu;
    }

    let position = Tensor::arange(0u32, target_len as u32, device)?.to_dtype(dtype)?;
    let cache_pos = cache_position.to_dtype(dtype)?.reshape((batch_size, 1))?;
    let comp = position.broadcast_gt(&cache_pos)?; // shape [B, T]
    mask = mask.broadcast_mul(&comp.unsqueeze(1)?)?; // shape [B, 1, S, T]

    let mut mask = mask.unsqueeze(0)?; // [1, 1, S, T]
    mask = mask.expand((batch_size, 1, seq_len, target_len))?;

    if let Some(attn_mask) = attention_mask {
        let pad_mask = mask.narrow(3, 0, attn_mask.dims()[1])?.broadcast_add(&attn_mask.unsqueeze(1)?.unsqueeze(2)?)?;
        let is_zero = pad_mask.equal(0.0)?;
        mask = mask.masked_fill(&is_zero, min_value)?;
    }

    Ok(mask)
}


pub struct QwenRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QwenRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(-1)?;
        let x = x / (variance + self.eps)?.sqrt()?;
        let x = x.to_dtype(x_dtype)?;
        x.broadcast_mul(&self.weight)
    }
}


pub struct QwenRotaryEmbedding {
    inv_freq: Tensor,
    cos_cached: Tensor,
    sin_cached: Tensor,
    max_seq_len: usize,
    dim: usize,
}

impl QwenRotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        let i = Tensor::arange_step(0f32, dim as f32, 2f32, device)?.to_dtype(DType::F32)?;
        let inv_freq = (base as f32).powf(-i / dim as f32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.matmul(&inv_freq.unsqueeze(0)?)?; // shape [max_seq_len, dim/2]
        let emb = Tensor::cat(&[&freqs, &freqs], -1)?; // [max_seq_len, dim]
        let cos_cached = emb.cos()?;
        let sin_cached = emb.sin()?;
        Ok(Self { inv_freq, cos_cached, sin_cached, max_seq_len, dim })
    }

    pub fn get_cos_sin(&mut self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        if seq_len > self.max_seq_len {
            unimplemented!("RoPE cache extension not implemented yet");
        }
        Ok((
            self.cos_cached.i(0..seq_len)?,
            self.sin_cached.i(0..seq_len)?,
        ))
    }
}


pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    let shape = hidden_states.dims();
    let (b, k, s, h) = (shape[0], shape[1], shape[2], shape[3]);
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }
    let x = hidden_states.reshape((b, k, 1, s, h))?;
    let x = x.broadcast_as((b, k, n_rep, s, h))?;
    x.reshape((b, k * n_rep, s, h))
}


pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hidden_dim = x.dims()[3];
    let (x1, x2) = (
        x.i((.., .., .., 0..hidden_dim/2))?,
        x.i((.., .., .., hidden_dim/2..))?,
    );
    Tensor::cat(&[&(-x2)?, &x1], -1)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_ids: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.index_select(0, position_ids)?.unsqueeze(1)?;
    let sin = sin.index_select(0, position_ids)?.unsqueeze(1)?;
    let q_rot = q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?;
    let k_rot = k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?;
    Ok((q_rot, k_rot))
}


#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    pub key: Tensor,   // [B, H_kv, T, D]
    pub value: Tensor, // [B, H_kv, T, D]
}

#[derive(Debug, Clone)]
pub struct KvCache {
    pub entries: Vec<Option<KvCacheEntry>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
        }
    }

    pub fn get(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.entries[layer_idx].as_ref()
    }

    pub fn update(&mut self, layer_idx: usize, entry: KvCacheEntry) {
        self.entries[layer_idx] = Some(entry);
    }
}


pub struct QwenAttention {
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_emb: QwenRotaryEmbedding,
}

impl QwenAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,             // [B, T, D]
        attention_mask: Option<&Tensor>,    // [B, 1, T, S]
        position_ids: &Tensor,              // [B, T]
        cache_entry: Option<&mut KvCacheEntry>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // 1. QKV projection
        let q = self.q_proj.forward(hidden_states)?; // [B, T, D]
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // 2. 转为 [B, H, T, D_head]
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, H, T, D]
        let mut k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?; // [B, H_kv, T, D]
        let mut v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. rotary embedding
        let q = self.rotary_emb.apply(&q, position_ids)?;
        let k = self.rotary_emb.apply(&k, position_ids)?;

        // 4. 如果使用缓存，拼接旧KV
        if let Some(cache) = cache_entry {
            if !cache.key.is_empty()? {
                k = Tensor::cat(&[&cache.key, &k], 2)?; // dim=2 = time
                v = Tensor::cat(&[&cache.value, &v], 2)?;
            }
            // 更新缓存
            cache.key = k.clone();
            cache.value = v.clone();
        }

        // 5. Q x K^T
        let q = q.contiguous()?.reshape((b_sz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.contiguous()?.reshape((b_sz * self.num_kv_heads, -1, self.head_dim))?;
        let v = v.contiguous()?.reshape((b_sz * self.num_kv_heads, -1, self.head_dim))?;

        let attn_weights = q.matmul(&k.transpose(1, 2)?)? / (self.head_dim as f64).sqrt();

        // 6. Apply causal/attention mask
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_probs = attn_weights.softmax(-1)?;

        // 7. attention output
        let attn_output = attn_probs.matmul(&v)?;
        let attn_output = attn_output
            .reshape((b_sz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}


#[derive(Clone)]
pub enum ExpertWeight {
    Full {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
    Quant {
        gate: QuantTensor,
        up: QuantTensor,
        down: QuantTensor,
    },
}

#[derive(Clone)]
pub struct QuantTensor {
    pub nbits: Tensor,
    pub shape: Tensor,
    pub w_q: Tensor,
    pub scale: Tensor,
    pub zero: Tensor,
}

pub struct QwenMoeExpertMLP {
    pub layer_idx: usize,
    pub is_shared: bool,
    pub device: Device,
    pub quan_bit: usize, // 0, 2, 4
    pub expert_idx: Option<usize>,
    pub weight_fp16_path: Option<String>,
    pub weight_quant_path: Option<(String, String)>, // (int4, int2)
    pub expert_fp16: Option<ExpertWeight>,
    pub expert_quant: Option<(ExpertWeight, ExpertWeight)>,
    pub current_loaded: Option<ExpertWeight>,
}

impl QwenMoeExpertMLP {
    pub fn new(layer_idx: usize, device: Device, is_shared: bool, quan_bit: usize) -> Self {
        Self {
            layer_idx,
            device,
            is_shared,
            quan_bit,
            expert_idx: None,
            weight_fp16_path: None,
            weight_quant_path: None,
            expert_fp16: None,
            expert_quant: None,
            current_loaded: None,
        }
    }

    pub fn init_shared_weights(&mut self, path: &str) -> Result<()> {
        let p = format!("{}/original/model.layers.{}.mlp.shared_expert.", path, self.layer_idx);
        let gate = load_file_tensor(&format!("{}gate_proj.weight", p), &self.device)?;
        let up = load_file_tensor(&format!("{}up_proj.weight", p), &self.device)?;
        let down = load_file_tensor(&format!("{}down_proj.weight", p), &self.device)?;
        self.current_loaded = Some(ExpertWeight::Full { gate, up, down });
        Ok(())
    }

    pub fn init_expert_paths(&mut self, path: &str, idx: usize) {
        self.expert_idx = Some(idx);
        let fp16 = format!("{}/original/model.layers.{}.mlp.experts.{}.weight", path, self.layer_idx, idx);
        let int4 = format!("{}/quantized/int4/model.layers.{}.mlp.experts.{}.weight", path, self.layer_idx, idx);
        let int2 = format!("{}/quantized/int2/model.layers.{}.mlp.experts.{}.weight", path, self.layer_idx, idx);
        self.weight_fp16_path = Some(fp16);
        self.weight_quant_path = Some((int4, int2));
    }

    pub fn load_fp16_weights(&mut self) -> Result<()> {
        let path = self.weight_fp16_path.as_ref().unwrap();
        let map = load_file(path, &self.device)?;
        let gate = map.get("gate").unwrap().clone();
        let up = map.get("up").unwrap().clone();
        let down = map.get("down").unwrap().clone();
        self.expert_fp16 = Some(ExpertWeight::Full { gate, up, down });
        Ok(())
    }

    pub fn load_quant_weights(&mut self) -> Result<()> {
        let (int4_path, int2_path) = self.weight_quant_path.as_ref().unwrap();
        let int4 = load_quant_tensor_file(int4_path, &self.device)?;
        let int2 = load_quant_tensor_file(int2_path, &self.device)?;
        self.expert_quant = Some((int4, int2));
        Ok(())
    }

    pub fn load_to_gpu(&mut self, nbit: usize) -> Result<()> {
        let weight = match nbit {
            0 => self.expert_fp16.as_ref().unwrap().clone(),
            4 => self.expert_quant.as_ref().unwrap().0.clone(),
            2 => self.expert_quant.as_ref().unwrap().1.clone(),
            _ => panic!("Unsupported quant bit."),
        };
        self.current_loaded = Some(weight);
        Ok(())
    }

    pub fn dequan_experts(&mut self) -> Result<()> {
        if let Some((int4, int2)) = &self.expert_quant {
            let target = match self.quan_bit {
                4 => int4,
                2 => int2,
                _ => panic!("Invalid quan_bit"),
            };
            let gate = dequantize(&target.gate)?;
            let up = dequantize(&target.up)?;
            let down = dequantize(&target.down)?;
            self.current_loaded = Some(ExpertWeight::Full { gate, up, down });
        }
        Ok(())
    }

    pub fn quan_experts(&mut self) -> Result<()> {
        self.load_to_gpu(self.quan_bit)
    }

    pub fn clear(&mut self) {
        self.current_loaded = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.current_loaded {
            Some(ExpertWeight::Full { gate, up, down }) => {
                let h1 = candle_nn::ops::linear(x, gate, None)?;
                let h2 = candle_nn::ops::linear(x, up, None)?;
                let act = h1.silu()? * h2;
                candle_nn::ops::linear(&act, down, None)
            }
            Some(ExpertWeight::Quant { .. }) => {
                todo!("dequant and compute")
            }
            None => Err(candle_core::Error::Msg("Expert weight not loaded".into())),
        }
    }
}

fn load_file_tensor(path: &str, device: &Device) -> Result<Tensor> {
    Ok(load_file(path, device)?["tensor"].clone())
}

fn load_quant_tensor_file(path: &str, device: &Device) -> Result<ExpertWeight> {
    let map = load_file(path, device)?;
    let build = |prefix: &str| QuantTensor {
        nbits: map[&format!("{}_nbits", prefix)].clone(),
        shape: map[&format!("{}_shape", prefix)].clone(),
        w_q: map[prefix].clone(),
        scale: map[&format!("{}_scale", prefix)].clone(),
        zero: map[&format!("{}_zero", prefix)].clone(),
    };
    Ok(ExpertWeight::Quant {
        gate: build("gate"),
        up: build("up"),
        down: build("down"),
    })
}

fn dequantize(q: &QuantTensor) -> Result<Tensor> {
    let w_q = &q.w_q;
    let zero = &q.zero;
    let scale = &q.scale;
    let w = (w_q - zero)? * scale;
    Ok(w)
}



pub struct QwenMoeBlock {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub gate: candle_nn::Linear,
    pub shared_expert: QwenMoeExpertMLP,
    pub shared_gate: candle_nn::Linear,
    pub experts: Vec<QwenMoeExpertMLP>,
    pub arc_cache: ArcCache,
    pub layer_idx: usize,
    pub device: Device,
    pub num_in_mem: usize,
}

impl QwenMoeBlock {
    pub fn new(
        layer_idx: usize,
        config: &QwenConfig,
        gate: candle_nn::Linear,
        shared_gate: candle_nn::Linear,
        experts: Vec<QwenMoeExpertMLP>,
        shared_expert: QwenMoeExpertMLP,
        device: Device,
    ) -> Self {
        let num_experts = config.num_experts;
        let top_k = config.num_experts_per_tok;
        let hidden_size = config.hidden_size;
        let num_in_mem = num_experts - config.offload_map[layer_idx];

        Self {
            num_experts,
            top_k,
            hidden_size,
            gate,
            shared_gate,
            experts,
            shared_expert,
            arc_cache: ArcCache::new(num_in_mem),
            layer_idx,
            device,
            num_in_mem,
        }
    }

    /// 使用 softmax 路由分数 top-k，预测当前输入会路由到哪些 expert
    pub fn predict_experts(&self, hidden_states: &Tensor) -> Result<Vec<usize>> {
        let (batch, seq_len, hidden_dim) = hidden_states.dims3()?;
        let x = hidden_states.reshape((batch * seq_len, hidden_dim))?;
        let router_logits = self.gate.forward(&x)?;
        let routing_weights = router_logits.softmax(-1)?;
        let (_weight, expert_ids) = routing_weights.topk(self.top_k, -1, true, false)?;

        let mut expert_set = std::collections::HashSet::new();
        let num_tokens = batch * seq_len;
        for i in 0..num_tokens {
            for k in 0..self.top_k {
                let id = expert_ids.i((i, k))?.to_scalar::<i64>()? as usize;
                expert_set.insert(id);
            }
        }

        let mut expert_list: Vec<usize> = expert_set.into_iter().collect();
        expert_list.sort();
        Ok(expert_list)
    }

    pub fn forward(&mut self, hidden_states: &Tensor, _prefetch_expert_idx: Option<&[usize]>) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = {
            let d = hidden_states.dims();
            (d[0], d[1], d[2])
        };
        let x = hidden_states.reshape((batch * seq_len, hidden_dim))?;

        // 1. 路由分数
        let router_logits = self.gate.forward(&x)?;
        let routing_weights = router_logits.softmax(-1)?.to_dtype(x.dtype())?;

        // 2. top-k experts
        let (weights, selected_experts) = routing_weights.topk(self.top_k, -1, true, false)?;

        // 3. 构造专家分配表
        let mut expert_outputs: HashMap<usize, Vec<(usize, Tensor)>> = HashMap::new();
        for token_idx in 0..(batch * seq_len) {
            for k in 0..self.top_k {
                let expert_id = selected_experts.i((token_idx, k))?.to_scalar::<i64>()? as usize;
                let weight = weights.i((token_idx, k))?.to_scalar::<f32>()?;
                let token = x.i(token_idx)?;
                let weighted = token * weight;
                expert_outputs.entry(expert_id).or_default().push((token_idx, weighted));
            }
        }

        // 4. 使用 ARC 缓存策略更新专家状态
        let needed_expert_ids: Vec<usize> = expert_outputs.keys().copied().collect();
        let evicted_experts = self.arc_cache.update_list(&needed_expert_ids);
        for evicted in evicted_experts {
            self.experts[evicted].clear(); // 卸载显存专家
        }

        // 5. 加载并计算专家输出
        let mut result = Tensor::zeros((batch * seq_len, hidden_dim), x.dtype(), &self.device)?;
        for (&expert_id, entries) in expert_outputs.iter() {
            if self.arc_cache.is_evicted(expert_id) {
                self.experts[expert_id].load_to_gpu(2)?; // 默认 int2
            }

            let input = Tensor::stack(&entries.iter().map(|(_, t)| t).collect::<Vec<_>>(), 0)?;
            let output = self.experts[expert_id].forward(&input)?;

            for (i, (token_idx, _)) in entries.iter().enumerate() {
                let token_out = output.i(i)?;
                result = result.index_add(0, &Tensor::from_slice(&[*token_idx], &self.device)?, &token_out)?;
            }

            // 可选清理
            if self.arc_cache.is_evicted(expert_id) {
                self.experts[expert_id].clear();
            }
        }

        // 6. 共享专家 fallback（sigmoid gate）
        let shared_out = self.shared_expert.forward(&x)?;
        let shared_weight = self.shared_gate.forward(&x)?.sigmoid()?;
        let shared_scaled = shared_out * shared_weight;

        let final_out = result + shared_scaled;
        final_out.reshape((batch, seq_len, hidden_dim))
    }
}



pub struct QwenDecoderLayer {
    pub attention_norm: QwenRmsNorm,
    pub attention: QwenAttention,
    pub ffn_norm: QwenRmsNorm,
    pub moe_block: QwenMoeBlock,
}

impl QwenDecoderLayer {
    pub fn predict(&self, hidden_states: &Tensor) -> Result<Vec<usize>> {
        self.moe_block.predict_experts(hidden_states)
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        cache_entry: Option<&mut KvCacheEntry>,
    ) -> Result<Tensor> {
        // 1. Attention 路径
        let residual = hidden_states.clone();
        let x = self.attention_norm.forward(hidden_states)?;
        let x = self.attention.forward(&x, attention_mask, position_ids, cache_entry)?; // 传入缓存
        let hidden_states = (x + residual)?;

        // 2. MLP + MoE 路径
        let residual = hidden_states.clone();
        let x = self.ffn_norm.forward(&hidden_states)?;
        let x = self.moe_block.forward(&x, None)?;
        let hidden_states = (x + residual)?;

        Ok(hidden_states)
    }
}


pub fn build_decoder_layer(
    vb: &candle_nn::VarBuilder,
    config: &QwenConfig,
    layer_idx: usize,
    device: &Device,
) -> Result<QwenDecoderLayer> {
    let attention_norm_weight = vb.get(&format!("layers.{layer_idx}.input_layernorm.weight"))?;
    let ffn_norm_weight = vb.get(&format!("layers.{layer_idx}.post_attention_layernorm.weight"))?;
    let attention_norm = QwenRmsNorm::new(attention_norm_weight, config.rms_norm_eps);
    let ffn_norm = QwenRmsNorm::new(ffn_norm_weight, config.rms_norm_eps);

    let q_proj = candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp(&format!("layers.{layer_idx}.self_attn.q_proj")),
    )?;
    let k_proj = candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp(&format!("layers.{layer_idx}.self_attn.k_proj")),
    )?;
    let v_proj = candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp(&format!("layers.{layer_idx}.self_attn.v_proj")),
    )?;
    let o_proj = candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp(&format!("layers.{layer_idx}.self_attn.o_proj")),
    )?;

    let rotary_emb = QwenRotaryEmbedding::new(
        config.hidden_size / config.num_attention_heads,
        config.rope_max_length,
        10000.0,
        device,
    )?;

    let attention = QwenAttention {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        rotary_emb,
    };

    let mut experts = vec![];
    for i in 0..config.num_experts {
        let mut expert = QwenMoeExpertMLP::new(layer_idx, device.clone(), false, config.quant_bit);
        expert.init_expert_paths(&config.model_path, i);
        experts.push(expert);
    }

    let mut shared_expert = QwenMoeExpertMLP::new(layer_idx, device.clone(), true, config.quant_bit);
    shared_expert.init_shared_weights(&config.model_path)?;

    let gate = candle_nn::linear(
        config.hidden_size,
        config.num_experts,
        vb.pp(&format!("layers.{layer_idx}.mlp.gate")),
    )?;
    let shared_gate = candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp(&format!("layers.{layer_idx}.mlp.shared_gate")),
    )?;

    let moe_block = QwenMoeBlock::new(
        layer_idx,
        config.num_experts,
        config.top_k,
        config.hidden_size,
        device.clone(),
        config.max_expert_in_gpu,
        gate,
        shared_gate,
        experts,
        shared_expert,
    );

    Ok(QwenDecoderLayer {
        attention_norm,
        attention,
        ffn_norm,
        moe_block,
    })
}

#[derive(Debug, Clone)]
pub struct DynamicCache {
    pub entries: Vec<Option<KvCacheEntry>>, // 每层的 kv
    pub max_len: usize,                     // 累积的缓存长度
}

impl DynamicCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            max_len: 0,
        }
    }

    /// 获取指定层的缓存
    pub fn get_entry(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.entries[layer_idx].as_ref()
    }

    /// 获取可变引用
    pub fn get_entry_mut(&mut self, layer_idx: usize) -> Option<&mut KvCacheEntry> {
        self.entries[layer_idx].as_mut()
    }

    /// 更新某一层缓存（自动拼接）
    pub fn append_entry(&mut self, layer_idx: usize, new_entry: KvCacheEntry) -> Result<()> {
        if let Some(ref mut entry) = self.entries[layer_idx] {
            entry.key = Tensor::cat(&[&entry.key, &new_entry.key], 2)?;     // [B, H, T+ΔT, D]
            entry.value = Tensor::cat(&[&entry.value, &new_entry.value], 2)?;
        } else {
            self.entries[layer_idx] = Some(new_entry);
        }
        Ok(())
    }

    /// 每轮调用后更新 max_len
    pub fn update_len(&mut self, delta: usize) {
        self.max_len += delta;
    }

    /// 获取当前 max_len，用于生成 position_ids 或 mask
    pub fn len(&self) -> usize {
        self.max_len
    }

    pub fn is_empty(&self) -> bool {
        self.max_len == 0
    }
}



pub struct QwenModel {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<QwenDecoderLayer>,
    pub norm: QwenRmsNorm,
    pub device: Device,
}

impl QwenModel {
    pub fn new(vb: &VarBuilder, config: &QwenConfig, device: &Device) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let norm_weight = vb.get("model.norm.weight")?;
        let norm = QwenRmsNorm::new(norm_weight, config.rms_norm_eps);

        let mut layers = vec![];
        for i in 0..config.num_hidden_layers {
            let layer = build_decoder_layer(vb, config, i, device)?;
            layers.push(layer);
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: device.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_position: &Tensor,
        cache: Option<&mut DynamicCache>,
        config: &QwenConfig,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let (batch, seq_len) = input_ids.dims2()?;

        let mask = causal_mask_with_cache_position(
            attention_mask,
            seq_len,
            seq_len,
            hidden_states.dtype(),
            &self.device,
            f32::NEG_INFINITY as f64,
            cache_position,
            batch,
        )?;

        for i in 0..self.layers.len() {
            let cache_entry = cache.and_then(|c| c.get_entry_mut(i));
            hidden_states = self.layers[i].forward(
                &hidden_states,
                Some(&mask),
                cache_position,
                cache_entry,
            )?;

            if config.decoder_sparse_step == 1
                && i + 1 < self.layers.len()
                && config.offload_map[i + 1] != 0
            {
                let experts = self.layers[i + 1].predict(&hidden_states)?;
                for &eid in &experts {
                    self.layers[i + 1].moe_block.experts[eid].load_to_gpu(2)?;
                }
            }
        }

        if let Some(cache) = cache {
            cache.update_len(seq_len);
        }

        self.norm.forward(&hidden_states)
    }

}


pub struct QwenForCausalLM {
    pub model: QwenModel,
    pub lm_head: candle_nn::Linear,
    pub device: Device,
}

impl QwenForCausalLM {
    pub fn new(vb: &VarBuilder, config: &QwenConfig, device: &Device) -> Result<Self> {
        let model = QwenModel::new(vb, config, device)?;
        let lm_head = candle_nn::linear(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
        )?;
        Ok(Self {
            model,
            lm_head,
            device: device.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.model.forward(input_ids, attention_mask, position_ids)?;
        self.lm_head.forward(&hidden_states)
    }
}


impl QwenForCausalLM {
    pub fn generate(
        &mut self,
        input_ids: &Tensor,         // [1, T]
        max_length: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Tensor> {
        let mut generated = input_ids.clone(); // shape: [1, T]
        let mut dynamic_cache = DynamicCache::new(self.model.layers.len());

        for _ in 0..max_length {
            let cur_len = generated.dims()[1];

            // 1. 获取最后一个 token（注意！支持增量）
            let last_token = generated.i((.., -1))?.clone(); // [1]

            // 2. 生成 cache_position（动态缓存位置）
            let cache_position = {
                let start = dynamic_cache.len() as u32;
                Tensor::arange(start, start + 1, &self.device)?.unsqueeze(0)? // [1,1]
            };

            // 3. 构造 causal mask（使用 cache_position）
            let causal_mask = causal_mask_with_cache_position(
                None,
                1,
                1,
                last_token.dtype(),
                &self.device,
                f32::NEG_INFINITY as f64,
                &cache_position,
                1, // batch_size
            )?;

            // 4. 推理（使用 DynamicCache）
            let logits = self.model.forward(
                &last_token,
                Some(&causal_mask),
                &cache_position,
                Some(&mut dynamic_cache),
                &self.config, // 现在你 forward 中已包含 config
            )?;

            let logits = logits.i((.., -1, ..))?; // [1, vocab]
            let next_token = logits.argmax(-1)?;  // [1]
            let next_token_id = next_token.to_vec1::<u32>()?[0];

            // 5. 拼接到 generated
            let next_token = next_token.unsqueeze(0)?.unsqueeze(0)?; // [1,1]
            generated = Tensor::cat(&[&generated, &next_token], 1)?;

            // 6. 更新 DynamicCache 长度
            dynamic_cache.update_len(1);

            // 7. 停止条件
            if let Some(eos_id) = eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }
        }

        Ok(generated)
    }


    pub fn generate_stream_text<'a>(
        &'a mut self,
        tokenizer: &'a Tokenizer,
        input_ids: &Tensor,
        eos_token_id: Option<u32>,
    ) -> impl Iterator<Item = Result<String>> + 'a {
        let mut generated = input_ids.clone();
        let mut cache = DynamicCache::new(self.model.layers.len());

        std::iter::from_fn(move || {
            let cur_len = generated.dims()[1];
            let last_token = generated.i((.., -1)).ok()?.clone();

            let pos = cache.len() as u32;
            let position_ids = Tensor::arange(pos, pos + 1, &self.device).ok()?.unsqueeze(0).ok()?;

            let mask = causal_mask_with_cache_position(
                None,
                1,
                1,
                last_token.dtype(),
                &self.device,
                f32::NEG_INFINITY as f64,
                &position_ids,
                1,
            ).ok()?;

            let logits = self.model.forward(
                &last_token,
                Some(&mask),
                &position_ids,
                Some(&mut cache),
            ).ok()?;

            let logits = logits.i((.., -1, ..)).ok()?;
            let next_token = logits.argmax(-1).ok()?;
            let next_token_id = next_token.to_vec1::<u32>().ok()?[0];

            // 拼接
            let next_token = next_token.unsqueeze(0).ok()?.unsqueeze(0).ok()?;
            generated = Tensor::cat(&[&generated, &next_token], 1).ok()?;

            // 解码
            let decoded = tokenizer.decode(&[next_token_id], true).ok()?;

            // 停止条件
            if let Some(eos_id) = eos_token_id {
                if next_token_id == eos_id {
                    return None;
                }
            }

            Some(Ok(decoded))
        })
    }
}
