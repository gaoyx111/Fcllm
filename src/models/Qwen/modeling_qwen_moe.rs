use candle_core::{DType, Tensor, Device, D, IndexOp};
use candle_nn::{self, Module, VarBuilder, Linear, linear, linear_no_bias, Embedding};
use std::collections::{HashSet, HashMap, BTreeMap};
use std::sync::Arc;
use dashmap::DashMap;
use std::path::PathBuf;
use rayon::prelude::*;
use anyhow::{Result, bail, anyhow};
use counter::Counter;
use tqdm::tqdm;
use std::time::Instant;
use half::f16;
use maplit::hashmap;
use safetensors::SafeTensors;
use std::fs;
use std::error::Error;

// Qwen 同文件夹模块
use super::configuration_qwen::*;
// 顶层 crate 的模块
use crate::expert_ARC_cache::ARCCache;
use crate::quantizer::dequantize;
use crate::utils::*;
use crate::Args;


fn _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: Option<&Tensor>,
    sequence_length: usize,
    target_length: usize,
    dtype: DType,
    device: &Device,
    min_dtype: f32,
    cache_position: &Tensor,
    batch_size: usize,
) -> Result<Option<Tensor>> {
    let causal_mask = if let Some(attn_mask) = attention_mask {
        if attn_mask.dims().len() == 4 {
            // 直接使用4D掩码
            Some(attn_mask.clone())
        } else {
            // 生成基础掩码
            let mut mask = Tensor::full(
                &[sequence_length, target_length],
                min_dtype as f64,
                device,
            )?.to_dtype(dtype)?;
            
            if sequence_length != 1 {
                mask = mask.triu(1)?; // 上三角，对角线为1
            }

            // 应用cache_position掩码
            let cache_pos = cache_position.reshape(&[cache_position.dims()[0], 1])?;
            let target_range = Tensor::arange(0f32, target_length as f32, device)?;
            let condition = target_range.broadcast_sub(&cache_pos.to_dtype(DType::F32)?)?.gt(0.0)?;
            mask = mask.broadcast_mul(&condition.to_dtype(dtype)?)?;

            // 扩展为4D [B, 1, S, T]
            let mut causal_mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            causal_mask = causal_mask.expand(&[batch_size, 1, sequence_length, target_length])?;

            // 处理padding_mask
            if let Some(attn_mask) = attention_mask {
                let mask_length = attn_mask.dims()[1]; // 假设attention_mask是[B, L]
                let padding_mask = causal_mask
                    .i((.., .., .., 0..mask_length))?
                    .broadcast_add(&attn_mask.unsqueeze(1)?.unsqueeze(1)?.to_dtype(dtype)?)?;
                
                let zero_mask = padding_mask.eq(0.0)?;
                let filled = causal_mask
                    .i((.., .., .., 0..mask_length))?
                    .masked_fill(&zero_mask, min_dtype as f64)?; // 注意类型转换
                
                causal_mask.i((.., .., .., 0..mask_length))?.copy_(&filled)?;
            }

            Some(causal_mask)
        }
    } else {
        // 无显式mask，生成纯因果掩码
        let mut mask = Tensor::full(
            &[sequence_length, target_length],
            min_dtype as f64,
            device,
        )?.to_dtype(dtype)?;
        if sequence_length != 1 {
            mask = mask.triu(1)?;
        }
        let target_range = Tensor::arange(0f32, target_length as f32, device)?;
        let cache_pos = cache_position.reshape(&[cache_position.dims()[0], 1])?;
        let condition = target_range.broadcast_sub(&cache_pos.to_dtype(DType::F32)?)?.gt(0.0)?;
        mask = mask.broadcast_mul(&condition.to_dtype(dtype)?)?;
        let causal_mask = mask.unsqueeze(0)?.unsqueeze(0)?.expand(&[batch_size, 1, sequence_length, target_length])?;
        Some(causal_mask)
    };

    Ok(causal_mask)
}


pub struct Qwen2MoeRMSNorm {
    weight: Tensor,
    eps: f64,
    device: Device,
}

impl Qwen2MoeRMSNorm {
    pub fn new(hidden_size: usize, device: &Device, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[hidden_size], DType::F32, device)?;
        Ok(Self {
            weight,
            eps,
            device: device.clone(),
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        let loaded = candle_core::safetensors::load(path, &self.device)?;
        self.weight = loaded.get("tensor").unwrap().clone();
        Ok(())
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states.pow(2.0)?.mean_last_dim_keepdim()?;
        let normed = hidden_states.broadcast_mul(&(variance.add(self.eps)?.rsqrt()?))?;
        Ok(self.weight.broadcast_mul(&normed)?.to_dtype(input_dtype)?)
    }

    pub fn extra_repr(&self) -> String {
        format!("{:?}, eps={}", self.weight.dims(), self.eps)
    }
}


// 旋转位置编码实现
pub struct Qwen2MoeRotaryEmbedding {
    dim: usize,
    max_position_embeddings: usize,
    base: f64,
    inv_freq: Tensor,
    device: Device,
}

impl Qwen2MoeRotaryEmbedding {
    pub fn new(dim: usize, max_position_embeddings: usize, base: f64, device: &Device) -> Result<Self> {
        let freqs = Tensor::arange_step(0f32, dim as f32, 2f32, device)?.to_dtype(DType::F32)?;
        let inv_freq = (1f64 / base).powf(freqs.to_vec1::<f32>()?.iter().map(|&x| x / dim as f32).collect::<Vec<_>>())?
            .to_tensor(&[1, dim/2], DType::F32, device)?;

        Ok(Self {
            dim,
            max_position_embeddings,
            base,
            inv_freq,
            device: device.clone(),
        })
    }

    pub fn forward(&self, _x: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(0u32, seq_len as u32, &self.device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&self.inv_freq)?;
        let emb = Tensor::cat(&[freqs.clone(), freqs], 2)?;
        
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        
        Ok((cos, sin))
    }
}

// 辅助函数
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let size = x.dims();
    let last = *size.last().unwrap();
    let x1 = x.narrow(-1, 0, last / 2)?;
    let x2 = x.narrow(-1, last / 2, last / 2)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], -1)?)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_ids: &Tensor,
    unsqueeze_dim: Option<usize>, // 改为可选参数
) -> Result<(Tensor, Tensor)> {
    let unsqueeze_dim = unsqueeze_dim.unwrap_or(1); // 默认值为1
    
    let cos = cos.index_select(0, position_ids)?.unsqueeze(unsqueeze_dim)?;
    let sin = sin.index_select(0, position_ids)?.unsqueeze(unsqueeze_dim)?;
    
    let q_embed = q.broadcast_mul(&cos)?.add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k.broadcast_mul(&cos)?.add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    
    Ok((q_embed, k_embed))
}

pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    let shape = hidden_states.dims();
    let (b, num_kv_heads, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }
    let expanded = hidden_states.unsqueeze(2)?.expand(&[b, num_kv_heads, n_rep, s, d])?;
    expanded.reshape(&[b, num_kv_heads * n_rep, s, d])
}


// 缓存特质定义
pub trait Cache {
    fn get_usable_length(&self, current_len: usize, layer_idx: usize) -> usize;
    fn update(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        rope_args: Option<(&Tensor, &Tensor, Option<&Tensor>)>,
    ) -> Result<(Tensor, Tensor)>;

    fn is_static(&self) -> bool;
    fn get_max_length(&self) -> usize;

    fn to_legacy_cache(&self) -> Option<Box<dyn Cache>> {
        None
    }
    fn get_seq_length(&self) -> usize;
    fn boxed_clone(&self) -> Box<dyn Cache>;
}

// DynamicCache 实现
pub struct DynamicCache {
    pub past_keys: HashMap<usize, Tensor>,
    pub past_values: HashMap<usize, Tensor>,
}

impl DynamicCache {
    pub fn new() -> Self {
        Self {
            past_keys: HashMap::new(),
            past_values: HashMap::new(),
        }
    }

    // 从旧缓存格式（Python兼容）转换
    pub fn from_legacy_cache(legacy_cache: Option<Vec<Tensor>>) -> Self {
        let mut cache = Self::new();
        if let Some(legacy) = legacy_cache {
            // 假设旧缓存格式为 [key_layer0, value_layer0, key_layer1, value_layer1, ...]
            for i in 0..legacy.len() {
                if i % 2 == 0 {
                    // 偶数索引为key，奇数索引为value
                    let layer_idx = i / 2;
                    let key = legacy[i].clone();
                    let value = legacy[i + 1].clone();
                    cache.past_keys.insert(layer_idx, key);
                    cache.past_values.insert(layer_idx, value);
                }
            }
        }
        cache
    }
}

impl Cache for DynamicCache {
    fn get_usable_length(&self, _current_len: usize, layer_idx: usize) -> usize {
        self.past_keys
            .get(&layer_idx)
            .map_or(0, |k| k.dims()[2])
    }

    fn update(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        _rope_args: Option<(&Tensor, &Tensor, Option<&Tensor>)>,
    ) -> Result<(Tensor, Tensor)> {
        let new_key = if let Some(prev_key) = self.past_keys.get(&layer_idx) {
            Tensor::cat(&[prev_key, key], 2)?
        } else {
            key.clone()
        };

        let new_value = if let Some(prev_value) = self.past_values.get(&layer_idx) {
            Tensor::cat(&[prev_value, value], 2)?
        } else {
            value.clone()
        };

        self.past_keys.insert(layer_idx, new_key.clone());
        self.past_values.insert(layer_idx, new_value.clone());

        Ok((new_key, new_value))
    }

    fn is_static(&self) -> bool {
        false
    }

    fn get_max_length(&self) -> usize {
        self.past_keys.values().map(|k| k.dims()[2]).max().unwrap_or(0)
    }

    fn to_legacy_cache(&self) -> Option<Box<dyn Cache>> {
        let mut tensors: Vec<Tensor> = Vec::new();
        let max_layer = self
            .past_keys
            .keys()
            .chain(self.past_values.keys())
            .cloned()
            .max()
            .unwrap_or(0);

        for layer_idx in 0..=max_layer {
            if let (Some(k), Some(v)) = (
                self.past_keys.get(&layer_idx),
                self.past_values.get(&layer_idx),
            ) {
                tensors.push(k.clone());
                tensors.push(v.clone());
            }
        }

        Some(Box::new(LegacyCache::new(tensors)))
    }

    fn get_seq_length(&self) -> usize {
        self.get_max_length()
    }

    fn boxed_clone(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct LegacyCache {
    pub tensors: Vec<Tensor>, // [key_0, value_0, key_1, value_1, ...]
}

impl LegacyCache {
    pub fn new(tensors: Vec<Tensor>) -> Self {
        Self { tensors }
    }
}

impl Cache for LegacyCache {
    fn get_usable_length(&self, _current_len: usize, layer_idx: usize) -> usize {
        let key_idx = layer_idx * 2;
        if key_idx < self.tensors.len() {
            self.tensors[key_idx].dims()[2]
        } else {
            0
        }
    }

    fn update(
        &mut self,
        _key: &Tensor,
        _value: &Tensor,
        _layer_idx: usize,
        _rope_args: Option<(&Tensor, &Tensor, Option<&Tensor>)>,
    ) -> Result<(Tensor, Tensor)> {
        Err(anyhow::anyhow!(
            "LegacyCache does not support update (read-only)"
        ))
    }

    fn is_static(&self) -> bool {
        true
    }

    fn get_max_length(&self) -> usize {
        self.tensors
            .iter()
            .step_by(2) // only keys
            .map(|t| t.dims()[2])
            .max()
            .unwrap_or(0)
    }

    fn get_seq_length(&self) -> usize {
        0
    }

    fn boxed_clone(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
}

// ---- 权重加载工具函数 ----

fn load_linear(prefix: &str, name: &str, device: &Device) -> Result<Linear> {
    let weight = Tensor::load(&format!("{}/{}.weight", prefix, name), device)?;
    let bias = Tensor::load(&format!("{}/{}.bias", prefix, name), device)?;
    Ok(Linear::new(weight, Some(bias)))
}

fn load_linear_no_bias(prefix: &str, name: &str, device: &Device) -> Result<Linear> {
    let weight = Tensor::load(&format!("{}/{}.weight", prefix, name), device)?;
    Ok(Linear::new(weight, None))
}

// ---- Qwen2MoeAttention 实现 ----

pub struct Qwen2MoeAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rotary_emb: Qwen2MoeRotaryEmbedding,

    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_key_value_heads: usize,
    pub num_key_value_groups: usize,

    pub attention_dropout: f64,
    pub is_causal: bool,
}

impl Qwen2MoeAttention {
    pub fn new(
        config: &Qwen2MoeConfig,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_emb = Qwen2MoeRotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;
        if config.hidden_size % config.num_attention_heads != 0 {
            bail!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                config.hidden_size,
                config.num_attention_heads
            );
        }

        let q_proj = linear_with_bias_from_device(config.hidden_size,config.num_attention_heads * head_dim, device)?;
        let k_proj = linear_with_bias_from_device(config.hidden_size,config.num_key_value_heads * head_dim, device)?;
        let v_proj = linear_with_bias_from_device(config.hidden_size,config.num_key_value_heads * head_dim, device)?;
        
        let o_proj=linear_no_bias_from_device(config.num_attention_heads * head_dim,config.hidden_size,device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim,
            num_key_value_heads: config.num_key_value_heads,
            num_key_value_groups: config.num_attention_heads / config.num_key_value_heads,
            attention_dropout: config.attention_dropout,
            is_causal: true,
        })
    }

    pub fn init_weights(&mut self, base_path: &str, layer_idx: usize, device: &Device) -> Result<()> {
        let prefix = format!("{}/original/model.layers.{}.self_attn", base_path, layer_idx);
        self.q_proj = load_linear(&prefix, "q_proj", device)?;
        self.k_proj = load_linear(&prefix, "k_proj", device)?;
        self.v_proj = load_linear(&prefix, "v_proj", device)?;
        self.o_proj = load_linear_no_bias(&prefix, "o_proj", device)?;
        Ok(())
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        past_key_value: Option<&mut dyn Cache>,
        cache_position: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<(Tensor, Option<&mut dyn Cache>)> {
        let (b, q_len, _) = hidden_states.dims3()?;

        let query = self.q_proj.forward(hidden_states)?;
        let key = self.k_proj.forward(hidden_states)?;
        let value = self.v_proj.forward(hidden_states)?;

        let query = query.reshape((b, q_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let mut key = key.reshape((b, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;
        let mut value = value.reshape((b, q_len, self.num_key_value_heads, self.head_dim))?.transpose(1, 2)?;

        let mut kv_seq_len = key.dims()[2];
        if let Some(cache) = past_key_value.as_ref() {
            kv_seq_len += cache.get_usable_length(kv_seq_len, layer_idx);
        }

        let (cos, sin) = self.rotary_emb.forward(&value, kv_seq_len)?;
        let (mut query, mut key_out) = apply_rotary_pos_emb(&query, &key, &cos, &sin, position_ids, 1)?;

        if let Some(cache) = past_key_value.as_mut() {
            let (k, v) = cache.update(&key_out, &value, layer_idx, Some((&cos, &sin, cache_position)))?;
            key = k;
            value = v;
        } else {
            key = key_out;
        }

        let key = repeat_kv(&key, self.num_key_value_groups)?;
        let value = repeat_kv(&value, self.num_key_value_groups)?;

        let mut attn_weights = query.matmul(&key.transpose(2, 3)?)?;
        attn_weights = attn_weights.broadcast_div((self.head_dim as f64).sqrt())?;

        if let Some(mask) = attention_mask {
            let mask = mask.i((.., .., .., ..key.dims()[2]))?;
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = attn_weights.softmax(candle_core::D::Minus1)?;
        let attn_output = attn_weights.matmul(&value)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b, q_len, self.hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok((attn_output, past_key_value))
    }
}



pub struct Qwen2MoeSdpaAttention {
    pub base: Qwen2MoeAttention,
    pub layer_idx: usize,
}

impl Qwen2MoeSdpaAttention {
    pub fn new(
        config: &Qwen2MoeConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let device = config.device.clone();
        let base = Qwen2MoeAttention::new(config, layer_idx, &device)?;
        Ok(Self { base, layer_idx })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        past_key_value: Option<&mut dyn Cache>,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<&mut dyn Cache>)> {
        let (b, q_len, _) = hidden_states.dims3()?;

        let query = self.base.q_proj.forward(hidden_states)?;
        let key = self.base.k_proj.forward(hidden_states)?;
        let value = self.base.v_proj.forward(hidden_states)?;

        let query = query.reshape((b, q_len, self.base.num_heads, self.base.head_dim))?.transpose(1, 2)?;
        let mut key = key.reshape((b, q_len, self.base.num_key_value_heads, self.base.head_dim))?.transpose(1, 2)?;
        let mut value = value.reshape((b, q_len, self.base.num_key_value_heads, self.base.head_dim))?.transpose(1, 2)?;

        let mut kv_seq_len = key.dims()[2];
        if let Some(cache) = past_key_value.as_ref() {
            kv_seq_len += cache.get_usable_length(kv_seq_len, self.layer_idx);
        }

        let (cos, sin) = self.base.rotary_emb.forward(&value, kv_seq_len)?;
        let (mut query, mut key_out) = apply_rotary_pos_emb(&query, &key, &cos, &sin, position_ids, 1)?;

        if let Some(cache) = past_key_value.as_mut() {
            let (k, v) = cache.update(&key_out, &value, self.layer_idx, Some((&cos, &sin, cache_position)))?;
            key = k;
            value = v;
        } else {
            key = key_out;
        }

        key = repeat_kv(&key, self.base.num_key_value_groups)?;
        value = repeat_kv(&value, self.base.num_key_value_groups)?;

        let is_causal = attention_mask.is_none() && q_len > 1;

        let attn_output = scaled_dot_product_attention(
            &query,
            &key,
            &value,
            attention_mask,
            self.base.attention_dropout,
            is_causal,
        )?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, q_len, self.base.hidden_size))?;

        let attn_output = self.base.o_proj.forward(&attn_output)?;

        Ok((attn_output, past_key_value))
    }
}

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f64,
    is_causal: bool,
) -> Result<Tensor> {
    // 计算缩放因子
    let scale = 1.0 / (*query.dims().last().unwrap() as f64).sqrt();

    // 计算注意力得分
    let mut attn_weights = query.matmul(&key.transpose(-2, -1)?)?;
    attn_weights = attn_weights.broadcast_mul(scale)?;

    // 应用注意力掩码
    if let Some(mask) = attn_mask {
        attn_weights = attn_weights.broadcast_add(mask)?;
    }

    // 应用因果掩码
    if is_causal {
        let (batch_size, num_heads, seq_len, _) = attn_weights.dims4()?;
        let causal_mask = Tensor::tril(seq_len, seq_len, query.device())?
            .reshape(&[1, 1, seq_len, seq_len])?
            .broadcast_to(&[batch_size, num_heads, seq_len, seq_len])?;

        // 构造所有元素为 -inf 的 Tensor（用于掩码）
        let dims = attn_weights.dims();
        let neg_inf = Tensor::full(dims.as_slice(), f64::NEG_INFINITY, query.device())?;

        // 使用 where_cond：如果 causal_mask 为 true，则保留 attn_weights；否则填充 -inf
        attn_weights = causal_mask.where_cond(&attn_weights, &neg_inf)?;
    }

    // 应用 softmax
    attn_weights = attn_weights.softmax(D::Minus1)?;

    // 应用 dropout（如果需要）
    if dropout_p > 0.0 {
        attn_weights = attn_weights.dropout(dropout_p)?;
    }

    // 计算最终的注意力输出
    let output = attn_weights.matmul(value)?;
    Ok(output)
}


pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// 激活函数查找表，等价于 PyTorch 的 ACT2FN
pub fn get_act_fn(name: &str) -> Result<fn(&Tensor) -> Result<Tensor>> {
    match name.to_lowercase().as_str() {
        "silu" => Ok(silu),
        "gelu" => Ok(gelu),
        "relu" => Ok(relu),
        _ => Err(candle_core::Error::Msg(format!("Unsupported activation: {}", name))),
    }
}

#[derive(Debug, Clone)]
pub struct TensorSet {
    pub tensors: HashMap<String, Tensor>, // key: "weight", or packed keys like int4 group
}

#[derive(Debug, Clone)]
pub enum TensorOrMap {
    Single(Tensor),
    Map(HashMap<String, Tensor>),
}

pub fn load_tensor_map(
    path: &str,
    device: &Device,
) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut map = HashMap::new();

    for name in tensors.names() {
        let st = tensors.tensor(name)?;
        let shape = st.shape().to_vec();
        let dtype = st.dtype();
        let tensor = Tensor::from_bytes(st.data(), dtype, &shape, device)?;
        map.insert(name.to_string(), tensor);
    }

    Ok(map)
}

fn clone_tensor_set(set: &TensorSet, device: &Device) -> Result<TensorOrMap> {
    if set.tensors.len() == 1 {
        // Assume the key is "weight"
        let tensor = set
            .tensors
            .get("weight")
            .ok_or_else(|| candle_core::Error::Msg("Missing 'weight' key in TensorSet".into()))?;
        Ok(TensorOrMap::Single(tensor.to_device(device)?))
    } else {
        // Copy all key-value tensors to device
        let mut map = HashMap::new();
        for (k, v) in &set.tensors {
            map.insert(k.clone(), v.to_device(device)?);
        }
        Ok(TensorOrMap::Map(map))
    }
}


pub struct Qwen2MoeMLP {
    pub config: Qwen2MoeConfig,
    pub layer_idx: usize,
    pub is_shared: bool,
    pub quan_bit: u8,

    pub gate: Option<TensorOrMap>,
    pub up: Option<TensorOrMap>,
    pub down: Option<TensorOrMap>,

    pub gate_cpu: HashMap<u8, TensorSet>,
    pub up_cpu: HashMap<u8, TensorSet>,
    pub down_cpu: HashMap<u8, TensorSet>,

    pub act_fn: fn(&Tensor) -> Result<Tensor, dyn Error>,

    pub idx: Option<usize>,
    pub weight_path: HashMap<u8, String>,
}

impl Qwen2MoeMLP {
    pub fn new(config: &Qwen2MoeConfig, layer_idx: usize, is_shared: bool) -> Result<Self> {
        let quan_bit = if layer_idx == 0 {
            0
        } else {
            config.quan_map[layer_idx]
        };

        let act_fn = get_act_fn(&config.hidden_act)?; 

        Ok(Self {
            config: config.clone(),
            layer_idx,
            is_shared,
            quan_bit,
            gate: None,
            up: None,
            down: None,
            gate_cpu: HashMap::new(),
            up_cpu: HashMap::new(),
            down_cpu: HashMap::new(),
            act_fn,
            idx: None,
            weight_path: HashMap::new(),
        })
    }

    pub fn init_weights(
        &mut self,
        base_path: &str,
        idx: Option<usize>,
        num_in_mem: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let device = &self.config.device;
        let init_device = Device::Cpu;

        if idx.is_none() {
            // shared_expert
            let path_prefix = format!("{}/original/model.layers.{}.mlp.shared_expert", base_path, self.layer_idx);

            self.gate = Some(load_file(&format!("{}.gate_proj.weight", path_prefix), device)?);
            self.up = Some(load_file(&format!("{}.up_proj.weight", path_prefix), device)?);
            self.down = Some(load_file(&format!("{}.down_proj.weight", path_prefix), device)?);
        } else {
            let idx = idx.unwrap();
            self.idx = Some(idx);

            self.weight_path.insert(0, format!("{}/original/model.layers.{}.mlp.experts.{}.weight", base_path, self.layer_idx, idx));
            self.weight_path.insert(4, format!("{}/quantized/int4/model.layers.{}.mlp.experts.{}.weight", base_path, self.layer_idx, idx));
            self.weight_path.insert(2, format!("{}/quantized/int2/model.layers.{}.mlp.experts.{}.weight", base_path, self.layer_idx, idx));

            // fp16
            if self.quan_bit == 0 {
                let mut weights = load_tensor_map(self.weight_path.get(&0).unwrap(), &init_device)?;
                let gate = weights.remove("gate").ok_or("missing gate")?.pin_memory();
                let up = weights.remove("up").ok_or("missing up")?.pin_memory();
                let down = weights.remove("down").ok_or("missing down")?.pin_memory();

                self.gate_cpu.insert(0, TensorSet { tensors: hashmap! { "weight".to_string() => gate.clone() } });
                self.up_cpu.insert(0, TensorSet { tensors: hashmap! { "weight".to_string() => up.clone() } });
                self.down_cpu.insert(0, TensorSet { tensors: hashmap! { "weight".to_string() => down.clone() } });

                self.gate = Some(gate.to(device)?);
                self.up = Some(up.to(device)?);
                self.down = Some(down.to(device)?);

                // free pinned CPU
                self.gate_cpu.remove(&0);
                self.up_cpu.remove(&0);
                self.down_cpu.remove(&0);
            }

            // int4
            let mut int4_weight = load_tensor_map(self.weight_path.get(&4).unwrap(), &init_device)?;
            self.gate_cpu.insert(4, Self::extract_keys("gate", &mut int4_weight)?);
            self.up_cpu.insert(4, Self::extract_keys("up", &mut int4_weight)?);
            self.down_cpu.insert(4, Self::extract_keys("down", &mut int4_weight)?);

            // int2
            let mut int2_weight = load_tensor_map(self.weight_path.get(&2).unwrap(), &init_device)?;
            self.gate_cpu.insert(2, Self::extract_keys("gate", &mut int2_weight)?);
            self.up_cpu.insert(2, Self::extract_keys("up", &mut int2_weight)?);
            self.down_cpu.insert(2, Self::extract_keys("down", &mut int2_weight)?);

            if let Some(n) = num_in_mem {
                if idx < n && self.quan_bit != 0 {
                    self.gate = Some(clone_tensor_set(&self.gate_cpu[&self.quan_bit], device)?);
                    self.up = Some(clone_tensor_set(&self.up_cpu[&self.quan_bit], device)?);
                    self.down = Some(clone_tensor_set(&self.down_cpu[&self.quan_bit], device)?);
                }
            }
        }

        Ok(())
    }

    pub fn extract_keys(prefix: &str, weight: &mut HashMap<String, Tensor>) -> Result<TensorSet> {
        let get = |suffix: &str| {
            weight
                .remove(&format!("{}_{}", prefix, suffix))
                .ok_or_else(|| format!("Missing key: {}_{}", prefix, suffix).into())
        };

        let tensors = if let Some(w_q) = weight.remove(prefix) {
            hashmap! {
                "nbits".to_string() => get("nbits")?.pin_memory(),
                "shape".to_string() => get("shape")?.pin_memory(),
                "W_q".to_string() => w_q.pin_memory(),
                "scale".to_string() => get("scale")?.pin_memory(),
                "zero".to_string() => get("zero")?.pin_memory(),
            }
        } else {
            hashmap! {
                "weight".to_string() => get("weight")?.pin_memory()
            }
        };

        Ok(TensorSet { tensors })
    }

    pub fn load_from_cpu(
        &self,
        tensor_set: &TensorSet,
    ) -> Result<TensorOrMap> {
        let device = &self.config.device;

        if tensor_set.tensors.len() == 1 {
            let tensor = tensor_set
                .tensors
                .get("weight")
                .ok_or("Missing key: weight")?;
            Ok(TensorOrMap::Single(tensor.to_device(device)?))
        } else {
            let mut map = HashMap::new();
            for (k, v) in &tensor_set.tensors {
                let to_dev = if k == "nbits" || k == "shape" {
                    v.clone() // remain on CPU
                } else {
                    v.to_device(device)?
                };
                map.insert(k.clone(), to_dev);
            }
            Ok(TensorOrMap::Map(map))
        }
    }


    pub fn load_weights(&mut self, is_now: bool, nbit: Option<usize>) -> Result<()> {
        let quan_bit = nbit.unwrap_or(self.quan_bit);

        if is_now {
            self.gate = Some(self.load_from_cpu(self.gate_cpu.get(&quan_bit).ok_or("Missing gate")?)?);
            self.up = Some(self.load_from_cpu(self.up_cpu.get(&quan_bit).ok_or("Missing up")?)?);
            self.down = Some(self.load_from_cpu(self.down_cpu.get(&quan_bit).ok_or("Missing down")?)?);
        } else {
            let gate = self.gate_cpu.get(&quan_bit).cloned().ok_or("Missing gate")?;
            let up = self.up_cpu.get(&quan_bit).cloned().ok_or("Missing up")?;
            let down = self.down_cpu.get(&quan_bit).cloned().ok_or("Missing down")?;

            let (gate_res, (up_res, down_res)) = rayon::join(
                || self.load_from_cpu(&gate),
                || rayon::join(
                    || self.load_from_cpu(&up),
                    || self.load_from_cpu(&down),
                ),
            );

            self.gate = Some(gate_res?);
            self.up = Some(up_res?);
            self.down = Some(down_res?);
        }

        Ok(())
    }


    pub fn dequan_experts(&mut self) -> Result<()> {
        if !self.is_shared && self.quan_bit != 0 {
            if let Some(ref gate) = self.gate {
                self.gate = Some(TensorOrMap::Single(dequantize(gate)?));
            }
            if let Some(ref up) = self.up {
                self.up = Some(TensorOrMap::Single(dequantize(up)?));
            }
            if let Some(ref down) = self.down {
                self.down = Some(TensorOrMap::Single(dequantize(down)?));
            }
        }
        Ok(())
    }

    pub fn quan_experts(&mut self) -> Result<()> {
        let quan_bit = self.quan_bit;

        if quan_bit != 0 {
            if let Some(weight) = self.gate_cpu.get(&quan_bit) {
                self.gate = Some(self.load_from_cpu(weight)?);
            }
            if let Some(weight) = self.up_cpu.get(&quan_bit) {
                self.up = Some(self.load_from_cpu(weight)?);
            }
            if let Some(weight) = self.down_cpu.get(&quan_bit) {
                self.down = Some(self.load_from_cpu(weight)?);
            }
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.gate = None;
        self.up = None;
        self.down = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use TensorOrMap::*;

        let gate = match &self.gate {
            Some(Single(t)) => t,
            _ => return Err("gate weight not loaded or invalid format".into()),
        };
        let up = match &self.up {
            Some(Single(t)) => t,
            _ => return Err("up weight not loaded or invalid format".into()),
        };
        let down = match &self.down {
            Some(Single(t)) => t,
            _ => return Err("down weight not loaded or invalid format".into()),
        };

        let x_gate = x.linear(gate, None)?.silu();
        let x_up = x.linear(up, None)?;
        let hidden = x_gate.mul(&x_up)?;
        let out = hidden.linear(down, None)?;

        Ok(out)
    }

}


// 用于支持多种索引方式
pub enum ExpertIndex {
    Single(usize),
    Batch(Vec<usize>),
}

/// 加载 .safetensors 文件中的 "tensor" 并返回 Tensor
pub fn load_file(path: &str, device: &Device) -> Result<Tensor> {
    let data = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    let st = tensors.tensor("tensor")?;

    let shape = st.shape().to_vec();
    let dtype = st.dtype();
    let tensor = Tensor::from_bytes(st.data(), dtype, &shape, device)?;
    Ok(tensor)
}

pub fn linear_no_bias_from_device(
    in_dim: usize,
    out_dim: usize,
    device: &Device,
) -> candle_core::Result<Linear> {
    // Kaiming Normal 初始化标准差
    let std = (2.0 / in_dim as f64).sqrt() as f32;
    // 使用正态分布，均值0，std=std
    let weight = Tensor::randn(0.0, std, (out_dim, in_dim), device)?;

    Ok(Linear::new(weight, None))
}

pub fn linear_with_bias_from_device(
    in_dim: usize,
    out_dim: usize,
    device: &Device,
) -> candle_core::Result<Linear> {
    // Kaiming Normal 初始化标准差
    let std = (2.0 / in_dim as f64).sqrt() as f32;
    // 初始化权重
    let weight = Tensor::randn(0.0, std, (out_dim, in_dim), device)?;
    // 初始化偏置为 0
    let bias = Tensor::zeros(out_dim, DType::U8, device)?;

    Ok(Linear::new(weight, Some(bias)))
}

pub struct Qwen2MoeSparseMoeBlock {
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub layer_idx: usize,
    pub device: Device,
    pub num_in_mem: usize,

    pub arc_cache: ARCCache,

    pub gate: Linear,
    pub shared_expert_gate: Linear,
    pub experts: Vec<Qwen2MoeMLP>,
    pub shared_expert: Qwen2MoeMLP,
}

impl Qwen2MoeSparseMoeBlock {
    pub fn new(config: &Qwen2MoeConfig, layer_idx: usize) -> Result<Self> {
        let num_experts = config.num_experts;
        let top_k = config.num_experts_per_tok;
        let norm_topk_prob = config.norm_topk_prob;
        let device = config.device.clone();
        let num_in_mem = num_experts - config.offload_map[layer_idx];

        let arc_cache = ARCCache::new(num_in_mem);

        let gate = linear_no_bias_from_device(config.hidden_size, num_experts, &device)?;
        let shared_expert_gate = linear_no_bias_from_device(config.hidden_size, 1, &device)?;

        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(Qwen2MoeMLP::new(config, layer_idx, false)?);
        }

        let shared_expert = Qwen2MoeMLP::new(config, layer_idx, true)?;

        Ok(Self {
            num_experts,
            top_k,
            norm_topk_prob,
            layer_idx,
            device,
            num_in_mem,
            arc_cache,
            gate,
            shared_expert_gate,
            experts,
            shared_expert,
        })
    }


    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        // 加载 gate 权重
        let gate_path = format!("{}/original/model.layers.{}.mlp.gate.weight", path, self.layer_idx);
        let gate_weight = load_file(&gate_path, &self.device)?;
        self.gate = candle_nn::Linear::new(gate_weight, None)?;

        // 加载 shared_expert_gate 权重
        let shared_expert_gate_path =
            format!("{}/original/model.layers.{}.mlp.shared_expert_gate.weight", path, self.layer_idx);
        let shared_expert_gate_weight = load_file(&shared_expert_gate_path, &self.device)?;
        self.shared_expert_gate = candle_nn::Linear::new(shared_expert_gate_weight, None)?;

        // 加载每个专家权重
        for idx in 0..self.num_experts {
            self.experts[idx].init_weights(path, Some(idx), Some(self.num_in_mem))?;
        }

        // 加载共享专家权重
        self.shared_expert.init_weights(path, None, None)?;

        Ok(())
    }

    pub fn load_weights(
        &mut self,
        idx: ExpertIndex,
        is_now: bool,
        int2_experts: Option<&HashSet<usize>>,
    ) -> Result<()> {
        match idx {
            ExpertIndex::Single(i) => {
                if self.arc_cache.is_evicted(i) {
                    let nbit = if let Some(set) = int2_experts {
                        if set.contains(&i) { 2 } else { 4 }
                    } else {
                        4
                    };
                    self.experts[i].load_weights(is_now, Some(nbit))?;
                }
            }
            ExpertIndex::Batch(indices) => {
                for &i in &indices {
                    if self.arc_cache.is_evicted(i) {
                        let nbit = if let Some(set) = int2_experts {
                            if set.contains(&i) { 2 } else { 4 }
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
                // let self_clone = self.clone();
                // rayon::spawn_fifo(move || {
                //     let _ = self_clone.experts[expert_idx].quan_experts();
                // });
                self.experts[expert_idx].quan_experts()?; //同步
            }
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        prefetch_expert_idx: Option<&Vec<usize>>,
    ) -> Result<Tensor> {
        let (batch_size, sequence_length, hidden_dim) = hidden_states.dims3()?;
        let hidden_states = hidden_states.reshape((batch_size * sequence_length, hidden_dim))?;

        // gate computation
        let router_logits = self.gate.forward(&hidden_states)?;
        let mut routing_weights = candle_nn::ops::softmax(&router_logits, 1)?;
        let (routing_weights, selected_experts) = routing_weights.topk(self.top_k, 1)?;

        let routing_weights = if self.norm_topk_prob {
            let sum = routing_weights.sum_keepdim(1)?;
            routing_weights.broadcast_div(&sum)?
        } else {
            routing_weights
        };

        let routing_weights = routing_weights.to_dtype(hidden_states.dtype())?;

        // 初始化 final_hidden_states
        let mut final_hidden_states = Tensor::zeros(
            (batch_size * sequence_length, hidden_dim),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;

        // 构建 expert_mask，shape: (num_experts, top_k, batch_size)
        let expert_mask = selected_experts
            .one_hot(self.num_experts)?
            .transpose(0, 2)?
            .transpose(1, 2)?;

        // flatten selected_experts，得到选中专家索引列表
        let expert_index = selected_experts.flatten_all()?.to_vec1::<usize>()?;

        let mut load_experts = Vec::new();

        // 预取专家与淘汰专家列表
        let (prefetch_expert_idx, evicted_list) = if self.layer_idx == 0 || prefetch_expert_idx.is_none() {
            (
                expert_index.iter().copied().collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>(),
                vec![],
            )
        } else {
            let prefetch_expert_idx = prefetch_expert_idx.unwrap();

            // 统计专家出现频率并排序
            let mut freq_counter = std::collections::BTreeMap::new();
            for &idx in &expert_index {
                *freq_counter.entry(idx).or_insert(0) += 1;
            }
            let mut freq_counter: Vec<_> = freq_counter.into_iter().collect();
            freq_counter.sort_by(|a, b| b.1.cmp(&a.1));

            // 加载未预取专家
            for (idx, _) in freq_counter {
                if !prefetch_expert_idx.contains(&idx) {
                    self.load_weights(ExpertIndex::Single(idx), false, None)?;
                    load_experts.push(idx);
                }
            }

            // 更新淘汰列表
            let evicted_list = if self.num_in_mem != 0 {
                self.arc_cache.update_list(&expert_index)
            } else {
                vec![]
            };

            // 清理淘汰专家缓存
            for idx in &evicted_list {
                self.experts[*idx].clear();
            }

            (prefetch_expert_idx.clone(), evicted_list)
        };

        // 计算预取专家
        for &expert_idx in &prefetch_expert_idx {
            if !expert_index.contains(&expert_idx) {
                if self.arc_cache.is_evicted(expert_idx) {
                    self.experts[expert_idx].clear();
                }
                continue;
            }

            self.experts[expert_idx].dequan_experts()?;
            let expert_layer = &self.experts[expert_idx];

            let (idx, top_x) = expert_mask.get(expert_idx)?.nonzero2()?;
            let current_state = hidden_states.index_select(&top_x,0 )?.reshape((-1, hidden_dim as i64))?;
            let current_hidden_states = expert_layer.forward(&current_state)?
                .broadcast_mul(&routing_weights.index((top_x.clone(), idx.clone()))?.unsqueeze(1)?)?;

            final_hidden_states = final_hidden_states.index_add(&top_x, &current_hidden_states.to_dtype(final_hidden_states.dtype())?, 0)?;
            self.post_comp(expert_idx)?;
        }

        // 计算按时加载的专家
        for &expert_idx in &load_experts {
            self.experts[expert_idx].dequan_experts()?;
            let expert_layer = &self.experts[expert_idx];

            let (idx, top_x) = expert_mask.get(expert_idx)?.nonzero2()?;
            let current_state = hidden_states.index_select(&top_x,0 )?.reshape((-1, hidden_dim as i64))?;
            let current_hidden_states = expert_layer.forward(&current_state)?
                .broadcast_mul(&routing_weights.index((top_x.clone(), idx.clone()))?.unsqueeze(1)?)?;

            final_hidden_states = final_hidden_states.index_add(&top_x, &current_hidden_states.to_dtype(final_hidden_states.dtype())?, 0)?;
            self.post_comp(expert_idx)?;
        }

        // 共享专家计算
        let shared_expert_output = self.shared_expert.forward(&hidden_states)?;
        let shared_gate = (self.shared_expert_gate.forward(&hidden_states)?.neg()?.exp()? + 1.0)?.recip()?;
        let shared_expert_output = shared_expert_output.broadcast_mul(&shared_gate)?;

        let final_hidden_states = final_hidden_states.add(&shared_expert_output)?;
        Ok(final_hidden_states.reshape((batch_size, sequence_length, hidden_dim))?)
    }

}


pub struct Qwen2MoeDecoderLayer {
    pub config: Qwen2MoeConfig,
    pub hidden_size: usize,
    pub layer_idx: usize,

    pub self_attn: Qwen2MoeSdpaAttention,
    pub mlp: Qwen2MoeSparseMoeBlock,

    pub input_layernorm: Qwen2MoeRMSNorm,
    pub post_attention_layernorm: Qwen2MoeRMSNorm,

    pub next_gate_cpu: Linear,
}

impl Qwen2MoeDecoderLayer {
    pub fn new(config: &Qwen2MoeConfig, layer_idx: usize) -> Result<Self> {
        let device = config.device.as_ref().unwrap_or(&Device::Cpu);
        Ok(Self {
            config: config.clone(),
            hidden_size: config.hidden_size,
            layer_idx,
            self_attn: Qwen2MoeSdpaAttention::new(config, layer_idx)?,
            mlp: Qwen2MoeSparseMoeBlock::new(config, layer_idx)?,
            input_layernorm: Qwen2MoeRMSNorm::new(
                config.hidden_size,
                device,
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: Qwen2MoeRMSNorm::new(
                config.hidden_size,
                device,
                config.rms_norm_eps,
            )?,
            next_gate_cpu: linear_no_bias_from_device(
                config.hidden_size,
                config.num_experts,
                &Device::Cpu,
            )?,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        self.self_attn.base.init_weights(path, self.layer_idx, self.config.device.as_ref().expect("Device is required"))?;
        self.mlp.init_weights(path)?;

        let input_ln_path = format!(
            "{}/original/model.layers.{}.input_layernorm.weight",
            path, self.layer_idx
        );
        let post_ln_path = format!(
            "{}/original/model.layers.{}.post_attention_layernorm.weight",
            path, self.layer_idx
        );
        self.input_layernorm.init_weights(&input_ln_path)?;
        self.post_attention_layernorm.init_weights(&post_ln_path)?;

        if self.layer_idx < self.config.num_hidden_layers - 1 {
            let gate_path = format!(
                "{}/original/model.layers.{}.mlp.gate.weight",
                path, self.layer_idx + 1
            );
            let new_weight = load_file(&gate_path, self.config.device.as_ref().expect("Device is required"))?;
            self.next_gate_cpu = Linear::new(new_weight, None);
            // self.next_gate_cpu.weight = load_file(&gate_path, &self.config.device)?;
        }

        Ok(())
    }

    pub fn predict(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (_, _, hidden_dim) = hidden_states.dims3()?;
        let hidden_states = hidden_states.reshape(&[-1, hidden_dim as i64])?;
        let router_logits = self.next_gate_cpu.forward(&hidden_states)?;

        let routing_weights = candle_nn::ops::softmax(&router_logits, 1)?;
        let dim = routing_weights.dims()?.len() - 1;
        let (_, selected_experts) = routing_weights.topk(5, dim)?;
        Ok(selected_experts)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<&mut dyn Cache>,
        cache_position: Option<&Tensor>,
        prefetch_expert_list: Option<&Vec<usize>>,
        next_layer: Option<&Qwen2MoeDecoderLayer>,
    ) -> Result<(Tensor, Option<&mut dyn Cache>, Option<Vec<usize>>)> {
        let residual = hidden_states.clone(); // 因为后面要加上注意力输出

        // LayerNorm 输入
        let hidden_states = self.input_layernorm.forward(&hidden_states)?;

        // 自注意力
        let (attn_output, present_cache) = self.self_attn.forward(
            &hidden_states,
            position_ids,
            attention_mask,
            past_key_value,
            cache_position,
        )?;

        // 残差连接
        let hidden_states = residual.try_add(&attn_output)?;

        // FC 层前再次残差保存
        let residual = hidden_states.clone();

        // 第二个 LayerNorm
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        
        let mut next_prefetch_expert_list = None;

        if let Some(next_layer) = next_layer {
            if self.layer_idx < self.config.num_hidden_layers - 1 &&
                self.config.offload_map[self.layer_idx + 1] != 0 {

                // 将 hidden_states 转移到 CPU
                let hidden_cpu = hidden_states.to_device(&Device::Cpu)?;

                // predict 返回 Tensor，转为 Vec<i64>
                let selected_experts = self.predict(&hidden_cpu)?;
                let selected_experts = selected_experts.to_vec1::<i64>()?;

                // 构建 Counter
                let expert_counts: Counter<i64> = selected_experts.iter().copied().collect();

                // 排序 (高频 -> 低频)
                let most_common_items: Vec<(i64, usize)> = expert_counts.most_common_ordered();

                // 提取专家 ID 列表
                let prefetch_list: Vec<usize> = most_common_items.iter().map(|&(id, _)| id as usize).collect();

                // 计算累计和与目标
                let value_sum: usize = most_common_items.iter().map(|&(_, v)| v).sum();
                let target_sum = (value_sum as f32 * 0.3).round() as usize;

                let mut current_sum = 0;
                let mut int2_experts = Vec::new();

                for &(expert_id, count) in most_common_items.iter().rev() {
                    if current_sum + count > target_sum {
                        break;
                    }
                    current_sum += count;
                    int2_experts.push(expert_id);
                }

                let int2_expert_set: HashSet<usize> =
                    int2_experts.iter().map(|&x| x as usize).collect();

                next_layer
                    .mlp
                    .load_weights(&prefetch_list, false, Some(&int2_expert_set))?;

                next_prefetch_expert_list = Some(prefetch_list);
            }
        }

        let hidden_states = self.mlp.forward(&hidden_states, prefetch_expert_list)?;
        let hidden_states = residual + hidden_states;

        Ok((hidden_states, present_cache, next_prefetch_expert_list))
    }
}


pub struct Qwen2MoeModel {
    pub config: Qwen2MoeConfig,
    pub padding_idx: Option<usize>,
    pub vocab_size: usize,
    
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen2MoeDecoderLayer>,
    pub norm: Qwen2MoeRMSNorm,
}


impl Qwen2MoeModel {
    pub fn new(config: &Qwen2MoeConfig) -> Result<Self> {
        let padding_idx = Some(config.pad_token_id);
        let vocab_size = config.vocab_size;
        let device = config.device.as_ref().ok_or_else(|| anyhow!("Device not set in config"))?;
        // let embed_tokens = Embedding::new(
        //     vocab_size, 
        //     config.hidden_size, 
        //     padding_idx, 
        //     &config.device
        // )?;
        let embeddings = Tensor::randn(
            0.0,
            1.0,
            &[vocab_size, config.hidden_size],
            device,
        )?;
        let embed_tokens = Embedding::new(embeddings, config.hidden_size);
        
        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| Qwen2MoeDecoderLayer::new(config, layer_idx))
            .collect::<Result<Vec<_>>>()?;
        
        let norm = Qwen2MoeRMSNorm::new(
            config.hidden_size, 
            device, 
            config.rms_norm_eps
        )?;
        
        Ok(Self {
            config: config.clone(),
            padding_idx,
            vocab_size,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        for i in tqdm(0..self.config.num_hidden_layers) {
            self.layers[i].init_weights(path)?;
        }

        let ln_path = format!("{}/original/model.norm.weight", path);
        let embed_path = format!("{}/original/model.embed_tokens.weight", path);

        self.norm.init_weights(&ln_path)?;

        let device = self.config.device.as_ref().unwrap();
        let embed_weight = load_file(&embed_path, device)?;
        self.embed_tokens = Embedding::new(embed_weight, self.config.hidden_size);

        Ok(())
    }

    pub fn get_input_embeddings(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn set_input_embeddings(&mut self, value: Embedding) {
        self.embed_tokens = value;
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_values: Option<Box<dyn Cache>>,
        inputs_embeds: Option<&Tensor>,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Box<dyn Cache>>)> {
        // 输入互斥检查
        let mut inputs_embeds = match (input_ids, inputs_embeds) {
            (Some(ids), None) => {
                let emb = self.embed_tokens.forward(ids)?;

                if let Some(pad_idx) = self.padding_idx {
                    let pad_tensor = Tensor::full_with_dtype(ids.dims(), pad_idx as i64, DType::I64, ids.device())?;  // padding idx 张量
                    let mask = ids.eq(&pad_tensor)?;                        // bool mask: true if pad
                    let mask_expanded = mask.unsqueeze(D::Minus1)?;                   // shape: [B, T, 1]
                    let zeros = Tensor::zeros_like(&emb)?;                     // same shape as emb
                    mask_expanded.where_cond(&zeros, &emb)?                   // 替换 padding 位置为零
                } else {
                    emb
                }
            },
            (None, Some(embeds)) => embeds.clone(),
            _ => return Err(anyhow!("Must specify either input_ids or inputs_embeds")),
        };

        // 缓存处理（from_legacy_cache逻辑）
        let (mut past_key_values, use_legacy_cache) = match past_key_values {
            Some(cache) => (cache, false),
            None => (Box::new(DynamicCache::from_legacy_cache(None)) as Box<dyn Cache>, true),
        };

        // 处理 cache_position
        let default_cache_position = {
            let past_seen_tokens = past_key_values.get_seq_length();
            Tensor::arange(
                past_seen_tokens as f32,
                past_seen_tokens as f32 + inputs_embeds.dims()[1] as f32,
                &inputs_embeds.device(),
            )?
        };
        let cache_position = cache_position.unwrap_or(&default_cache_position);

        // 处理 position_ids
        let position_ids = position_ids.unwrap_or(&cache_position.unsqueeze(0)?);

        // 生成因果掩码（关键修正）
        let causal_mask = self._update_causal_mask(
            attention_mask,
            &inputs_embeds,
            &cache_position,
            Some(&*past_key_values), // 传递不可变引用
        )?;

        let mut hidden_states = inputs_embeds;

        // 层循环
        let mut next_decoder_cache: Option<Box<dyn Cache>> = None;
        let mut next_prefetch_expert_list: Option<Vec<usize>> = None;

        for i in 0..self.config.num_hidden_layers {
            let next_layer = if i < self.config.num_hidden_layers - 1 {
                Some(&self.layers[i + 1])
            } else {
                None
            };

            let past_kv = Some(past_key_values.as_mut());

            let (output, cache, prefetch) = self.layers[i].forward(
                &hidden_states,
                causal_mask.as_ref(),
                Some(position_ids),
                past_kv,
                Some(&cache_position),
                next_prefetch_expert_list.as_ref(),
                next_layer,
            )?;

            hidden_states = output;
            next_decoder_cache = Some(past_key_values.boxed_clone());
            next_prefetch_expert_list = prefetch;
        }

        // 归一化和缓存转换
        hidden_states = self.norm.forward(&hidden_states)?;
        let next_cache = if use_legacy_cache {
            next_decoder_cache.and_then(|c| c.to_legacy_cache())
        } else {
            next_decoder_cache
        };

        Ok((hidden_states, next_cache))
    }

    pub fn _update_causal_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_tensor: &Tensor,
        cache_position: &Tensor,
        past_key_values: Option<&dyn Cache>,
    ) -> Result<Option<Tensor>> {
        let past_seen_tokens = past_key_values
            .map(|cache| cache.get_usable_length(0, 0))
            .unwrap_or(0);
        
        let using_static_cache = past_key_values
            .map(|cache| cache.is_static())
            .unwrap_or(false);

        // SDPA逻辑：仅当使用sdpa且非静态缓存时检查
        if self.config._attn_implementation == Some("sdpa".to_string()) && !using_static_cache {
            if self._ignore_causal_mask_sdpa(attention_mask, input_tensor, past_seen_tokens)? {
                return Ok(None);
            }
        }

        let dtype = input_tensor.dtype();
        let device = input_tensor.device();
        let min_dtype = match dtype {
            DType::F16 => f16::NEG_INFINITY.to_f32(),
            DType::F32 => f32::NEG_INFINITY,
            DType::F64 => f64::NEG_INFINITY as f32,
            _ => return Err(anyhow!("Unsupported dtype for causal mask")),
        };

        let sequence_length = input_tensor.dims()[1];
        let target_length = if using_static_cache {
            past_key_values.unwrap().get_max_length()
        } else {
            attention_mask
                .as_ref()
                .map(|m| m.dims()[3])
                .unwrap_or(past_seen_tokens + sequence_length + 1)
        };

        let batch_size = input_tensor.dims()[0];
        let causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length,
            target_length,
            dtype,
            device,
            min_dtype,
            cache_position,
            batch_size,
        )?;

        // SDPA后处理：仅当使用sdpa且存在attention_mask时处理
        if self.config._attn_implementation == Some("sdpa".to_string()) && attention_mask.is_some() {
            let causal_mask = self._unmask_unattended(&causal_mask.unwrap(), min_dtype)?;
            return Ok(Some(causal_mask));
        }

        Ok(causal_mask)
    }

    fn _ignore_causal_mask_sdpa(
        &self,
        attention_mask: Option<&Tensor>,
        _input_tensor: &Tensor,
        _past_seen_tokens: usize,
    ) -> Result<bool> {
        // Python逻辑：如果attention_mask为None且is_training为False则忽略
        // 由于Rust配置中无is_training，假设始终为推理模式（is_training=false）
        Ok(attention_mask.is_none())
    }

    fn _unmask_unattended(
        &self,
        expanded_mask: &Tensor,
        min_dtype: f32,
    ) -> Result<Tensor> {
        use candle_core::{DType, Tensor, D};

        // 布尔检查
        if expanded_mask.dtype() == DType::U8 {
            return Err(anyhow!("Expected float expanded_mask, got BoolTensor"));
        }

        // 创建 min_dtype 标量 tensor
        let device = expanded_mask.device();
        let min_dtype_tensor = Tensor::new(min_dtype, device)?;

        // 找出等于 min 的元素
        let is_min = expanded_mask.eq(&min_dtype_tensor)?;

        // 判断每一行是否都是 min
        let last_dim = is_min.dims()[is_min.rank() - 1] as u32;
        let row_sum = is_min.sum(D::Minus1)?;
        let all_rows_are_min = row_sum.eq(&Tensor::new(last_dim, device)?)?;

        // 取反，保留非全 min 的行
        let all_rows_are_min = all_rows_are_min.to_dtype(DType::U8)?;
        let ones = Tensor::ones(all_rows_are_min.dims(), DType::U8, device)?;
        let mask = ones.sub(&all_rows_are_min)?.unsqueeze(D::Minus1)?;

        // 转换 dtype
        let mask = mask.to_dtype(expanded_mask.dtype())?;

        // 应用 mask
        let result = expanded_mask.broadcast_mul(&mask)?;

        Ok(result)
    }


}


pub struct Qwen2MoeForCausalLM {
    config: Qwen2MoeConfig,
    device: Device,
    min_length: usize,
    max_length: usize,
    early_stopping: bool,
    path: PathBuf,
    model: Qwen2MoeModel,
    lm_head: Linear,
}


impl Qwen2MoeForCausalLM {
    pub fn new(args: &Args) -> Result<Self> {
        let config = get_qwen_config(&args.model)
            .map_err(|e| anyhow!("Invalid model name: {}", e))?;
        
        let device = args.device.clone();
        let mut config = config;
        config.device = device.clone();
        config._attn_implementation = Some("sdpa".to_string());
        
        let (offload_map, quan_map) = memory_cost_qwen(&config, args.memory_budget)?;
        config.offload_map = Some(offload_map);
        config.quan_map = Some(quan_map);
        println!("offload: {:?}", offload_map);
        println!("quan_map: {:?}", quan_map);

        let model = Qwen2MoeModel::new(&config)?;
        let lm_head = linear_no_bias_from_device(config.hidden_size, config.vocab_size, &device)?;

        Ok(Self {
            config,
            device,
            min_length: args.min_length,
            max_length: args.max_length,
            early_stopping: args.early_stopping,
            path: args.path.clone(),
            model,
            lm_head,
        })
    }

    pub fn init_weights(&mut self) -> Result<()> {
        let expanded_path = self.path.join("Qwen").join("Qwen1.5-MoE-A2.7B");
        let check_path = expanded_path.join("original/lm_head.weight");

        if !check_path.exists() {
            bail!("Weight file not found");
        }

        let expanded_path_str = expanded_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path"))?;

        let check_path_str = check_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path"))?;

        self.model.init_weights(expanded_path_str)?;

        let device = self.config.device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device not set in config"))?;

        let new_weight = load_file(check_path_str, device)?;
        self.lm_head = Linear::new(new_weight, None);

        Ok(())
    }

    pub fn get_input_embeddings(&self) -> &Embedding {
        self.model.get_input_embeddings()
    }

    pub fn set_input_embeddings(&mut self, value: Embedding) {
        self.model.set_input_embeddings(value);
    }

    pub fn get_output_embeddings(&self) -> &Linear {
        &self.lm_head
    }

    pub fn set_output_embeddings(&mut self, new_embeddings: Linear) {
        self.lm_head = new_embeddings;
    }

    pub fn set_decoder(&mut self, decoder: Qwen2MoeModel) {
        self.model = decoder;
    }

    pub fn get_decoder(&self) -> &Qwen2MoeModel {
        &self.model
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_values: Option<Box<dyn Cache>>,
        inputs_embeds: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        num_logits_to_keep: usize,
    ) -> Result<(Tensor, Option<Box<dyn Cache>>)> {
        let (hidden_states, next_cache) = self.model.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            cache_position,
        )?;

        let logits = self.lm_head.forward(
            &hidden_states.narrow(1, hidden_states.dims()[1].saturating_sub(num_logits_to_keep), num_logits_to_keep)?
        )?;

        Ok((logits, next_cache))
    }

    #[allow(unused_variables)] // 暂不使用梯度，可添加candle的no_grad宏
    pub fn generate(
        &self,
        mut input_ids: Tensor,
        mut attention_mask: Option<Tensor>,
        experiment_mode: Option<&str>,
    ) -> Result<(Tensor, f64)> {
        let batch_size = input_ids.dims()[0];
        let mut output = None;
        let mut prefill_time = 0.0;

        // Prefill timer
        if let Some("decoding") = experiment_mode {
            let start = Instant::now();
            prefill_time = start.elapsed().as_secs_f64(); // 实际计时应在 forward 前后设置
        }

        let seq_len = input_ids.dims()[1];
        let mut past_key_values: Box<dyn Cache> = Box::new(DynamicCache::new());

        if attention_mask.is_none() {
            attention_mask = Some(Tensor::ones(
                &[batch_size, seq_len],
                DType::I64,
                &self.device,
            )?);
        }

        let mut attention_mask = attention_mask.unwrap();
        let mut position_ids = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .unsqueeze(0)?
            .to_dtype(DType::I64)?;
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(DType::I64)?;

        for i in 0..1024 {
            let (logits, new_cache) = self.forward(
                Some(&input_ids),
                Some(&attention_mask),
                Some(&position_ids),
                Some(past_key_values),
                None,
                Some(&cache_position),
                1, // 只保留最后一个 token 的 logits
            )?;
            past_key_values = new_cache.unwrap();

            let logits = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;
            let next_token = logits.argmax(logits.dims().len() - 1)?; // shape: [batch, 1]

            output = Some(if let Some(prev) = output {
                Tensor::cat(&[&prev, &next_token], 1)?
            } else {
                next_token.clone()
            });

            // 提前终止
            let eos_id = self.config.eos_token_id as i64;
            if next_token.to_scalar::<i64>()? == eos_id {
                return Ok((output.unwrap(), prefill_time));
            }

            input_ids = next_token.clone(); // 用于下一步 forward

            // === 更新 position_ids ===
            let last_pos_index = position_ids.dims()[1] - 1;
            let last_pos = position_ids.i((.., last_pos_index))?; // shape: [batch]
            let one = Tensor::new(1i64, &self.device)?;
            position_ids = last_pos.broadcast_add(&one)?.unsqueeze(position_ids.rank())?; // shape: [batch, 1]

            // === 更新 cache_position ===
            let last_cache_index = cache_position.dims()[0] - 1;
            let last_cache_pos = cache_position.i(last_cache_index)?; // shape: []
            cache_position = last_cache_pos.broadcast_add(&one)?.unsqueeze(0)?; // shape: [1]

            // === 更新 attention_mask ===
            let new_mask = Tensor::ones(&[1, 1], DType::I64, &self.device)?; // [batch, 1]
            attention_mask = Tensor::cat(&[&attention_mask, &new_mask], 1)?;
        }

        Ok((output.unwrap(), prefill_time))
    }
}
