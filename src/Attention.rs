use crate::Cache::Cache;
use crate::RotaryEmbedding::{Qwen2MoeRotaryEmbedding, apply_rotary_pos_emb, repeat_kv};
use crate::configuration_qwen::Qwen2MoeConfig;
use crate::linear::new_uninitialized_linear;
use crate::load::load_linear_from_files;
use candle_core::{DType, Device, Error, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Arc;
// use safetensors::tensor as st;
// use safetensors::tensor::SafeTensors;

#[derive(Debug, Clone)]
pub struct Qwen2MoeAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rotary_emb: Arc<Qwen2MoeRotaryEmbedding>,
    pub kv_cache: Option<(Tensor, Tensor)>,

    pub layer_idx: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub is_causal: bool,
    pub attention_dropout: f64,
    pub device: Device,
}

impl Qwen2MoeAttention {
    pub fn new(cfg: &Qwen2MoeConfig, layer_idx: Option<usize>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        // 检查 hidden_size 是否能整除 num_heads
        if hidden_sz % num_heads != 0 {
            return Err(Error::msg(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                hidden_sz, num_heads
            )));
        }

        let q_proj = new_uninitialized_linear(hidden_sz, num_heads * head_dim, true, &cfg.device)?;
        let k_proj =
            new_uninitialized_linear(hidden_sz, num_kv_heads * head_dim, true, &cfg.device)?;
        let v_proj =
            new_uninitialized_linear(hidden_sz, num_kv_heads * head_dim, true, &cfg.device)?;
        let o_proj = new_uninitialized_linear(num_heads * head_dim, hidden_sz, false, &cfg.device)?;

        let rotary_emb = Arc::new(Qwen2MoeRotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            DType::F32,
            &cfg.device,
        )?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,

            layer_idx,
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            is_causal: true,
            attention_dropout: cfg.attention_dropout,
            device: cfg.device.clone(),
        })
    }

    pub fn init_weights(&mut self, base_path: &str) -> Result<()> {
        let layer_idx = self.layer_idx.unwrap();
        let prefix = format!("{}/original/model.layers.{layer_idx}.self_attn.", base_path);

        self.q_proj = load_linear_from_files(
            &format!("{}q_proj.weight", prefix),
            Some(&format!("{}q_proj.bias", prefix)),
            &self.device,
        )?;
        self.k_proj = load_linear_from_files(
            &format!("{}k_proj.weight", prefix),
            Some(&format!("{}k_proj.bias", prefix)),
            &self.device,
        )?;
        self.v_proj = load_linear_from_files(
            &format!("{}v_proj.weight", prefix),
            Some(&format!("{}v_proj.bias", prefix)),
            &self.device,
        )?;
        self.o_proj =
            load_linear_from_files(&format!("{}o_proj.weight", prefix), None, &self.device)?;

        Ok(())
    }

    pub fn forward<'a>(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mut past_key_value: Option<&'a mut Cache>,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<&'a mut Cache>)> {
        let (bsz, q_len, _) = hidden_states.dims3()?;

        // 1. q/k/v projection
        let query_states = self
            .q_proj
            .forward(hidden_states)?
            .reshape((bsz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (B, H, Q, D)
        let key_states = self
            .k_proj
            .forward(hidden_states)?
            .reshape((bsz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = self
            .v_proj
            .forward(hidden_states)?
            .reshape((bsz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 2. rotary embedding
        // Check layer_idx when caching is used
        let kv_seq_len = if let Some(cache) = &past_key_value {
            let layer_idx = self.layer_idx.ok_or_else(|| {
                Error::Msg(format!(
                    "Missing layer_idx: Cache structure changed since v4.36. \
                    If you're using {} with past_key_value, please set layer_idx.",
                    std::any::type_name::<Self>()
                ))
            })?;
            let cached_len = cache.kvs[layer_idx]
                .as_ref()
                .map_or(0, |(k, _)| k.dim(2).unwrap_or(0));
            q_len + cached_len
        } else {
            q_len
        };
        let (cos, sin) =
            self.rotary_emb
                .forward(kv_seq_len, value_states.dtype(), value_states.device())?;

        // 3. apply RoPE
        let (query_states, key_states) = apply_rotary_pos_emb(
            &query_states,
            &key_states,
            &cos,
            &sin,
            position_ids.unwrap(),
            1,
        )?;

        // 4. 更新 Cache
        let (key_states, value_states) = if let Some(ref mut cache) = past_key_value {
            cache.update_kv(
                &key_states,
                &value_states,
                self.layer_idx.unwrap(),
                &sin,
                &cos,
                cache_position,
            )?
        } else {
            (key_states, value_states)
        };

        // 5. repeat kv
        let key_states = repeat_kv(&key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(&value_states, self.num_kv_groups)?;

        // 6. Attention score (upcast to f32 for numerical stability)
        // .contiguous() is required on CUDA — transpose/expand only changes strides without moving data
        let query_f32 = query_states.to_dtype(DType::F32)?.contiguous()?;
        let key_f32 = key_states.to_dtype(DType::F32)?.contiguous()?;
        let value_f32 = value_states.to_dtype(DType::F32)?.contiguous()?;

        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let attn_weights = query_f32
            .matmul(&key_f32.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;

        let kv_seq_len = key_states.dim(2)?;

        // 7. Apply causal mask
        let attn_weights = if let Some(mask) = attention_mask {
            let causal_mask = mask.i((.., .., .., ..kv_seq_len))?;
            (attn_weights + causal_mask)?
        } else if q_len > 1 {
            // During prefill (q_len > 1), create causal mask so each token
            // can only attend to itself and previous tokens (matching Python SDPA is_causal=True)
            let mask = create_causal_mask(q_len, kv_seq_len, attn_weights.device())?;
            let mask = mask.broadcast_as(attn_weights.shape())?;
            (attn_weights + mask)?
        } else {
            attn_weights
        };

        // 8. Softmax + matmul all in F32, then cast back to original dtype at the end
        let attn_weights =
            candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?.contiguous()?;
        let attn_output = attn_weights.matmul(&value_f32)?;

        let attn_output = attn_output
            .to_dtype(query_states.dtype())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, q_len, self.hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok((attn_output, past_key_value))
    }
}

/// Build a [1, 1, q_len, kv_seq_len] causal mask filled with -inf above the diagonal.
fn create_causal_mask(q_len: usize, kv_seq_len: usize, device: &Device) -> Result<Tensor> {
    let min_val = f32::NEG_INFINITY;
    let offset = kv_seq_len - q_len; // past tokens before the current query window
    let mut data = vec![0f32; q_len * kv_seq_len];
    for i in 0..q_len {
        for j in 0..kv_seq_len {
            if j > i + offset {
                data[i * kv_seq_len + j] = min_val;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_seq_len), device)
}
