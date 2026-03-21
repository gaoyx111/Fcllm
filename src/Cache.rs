use std::collections::HashMap;
use candle_core::{DType, Device, Result, Tensor, IndexOp, D};
use crate::configuration_qwen::Qwen2MoeConfig;

#[derive(Debug, Clone)]
pub struct Cache {
    pub masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub cos: Tensor,
    pub sin: Tensor,
    pub device: Device,
}

pub fn calculate_default_inv_freq(cfg: &Qwen2MoeConfig) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
/*     pub fn new(use_kv_cache: bool, dtype: DType, config: &Qwen2MoeConfig, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    } */

    pub fn new(use_kv_cache: bool, dtype: DType, config: &Qwen2MoeConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;

        // 和 LLaMA 一样的 theta 推导方式
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (config.rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();

        // 创建 [max_seq_len, head_dim/2] 的位置矩阵和频率乘积
        let freqs: Vec<f32> = (0..max_seq_len)
            .flat_map(|pos| {
                inv_freq.iter().map(move |f| pos as f32 * f)
            })
            .collect();

        let freqs_tensor = Tensor::from_vec(freqs, (max_seq_len, inv_freq.len()), &config.device)?/* .to_dtype(dtype)? */;

        // 构建 cos 和 sin 缓存：[max_seq_len, head_dim/2]
        let cos = freqs_tensor.cos()?;
        let sin = freqs_tensor.sin()?;

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
            device: config.device.clone(),
        })
    }

    pub fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn get_usable_length(&self, layer_idx: usize, current_len: usize) -> Result<usize> {
        if let Some((k, _)) = &self.kvs[layer_idx] {
            Ok(k.dim(2)?)
        } else {
            Ok(0)
        }
    }

/*     pub fn update_kv(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        sin: &Tensor,
        cos: &Tensor,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if self.kvs[layer_idx].is_none() {
            self.kvs[layer_idx] = Some((key.clone(), value.clone()));
            return Ok((key.clone(), value.clone()));
        }

        let (mut k_cache, mut v_cache) = self.kvs[layer_idx].clone().unwrap();

        if let Some(pos) = cache_position {
            let dims = key.shape().dims(); // [B, H, T, D]
            let pos_expanded = pos
                .reshape(&[1, 1, dims[2], 1])? // [1, 1, T, 1]
                .expand(&[dims[0], dims[1], dims[2], 1])?
                .contiguous()?;

            println!("key shape: {:?}", key.shape());
            println!("value shape: {:?}", value.shape());
            println!("pos shape: {:?}", pos.shape());
            println!("pos_expanded shape: {:?}", pos_expanded.shape());

            let key = key.contiguous()?;
            let value = value.contiguous()?;

            k_cache = k_cache.scatter(&pos_expanded, &key, 2)?;
            v_cache = v_cache.scatter(&pos_expanded, &value, 2)?;
        } else {
            k_cache = Tensor::cat(&[k_cache, key.clone()], 2)?;
            v_cache = Tensor::cat(&[v_cache, value.clone()], 2)?;
        }

        self.kvs[layer_idx] = Some((k_cache.clone(), v_cache.clone()));
        Ok((k_cache, v_cache))
    } */

   pub fn update_kv(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        sin: &Tensor,
        cos: &Tensor,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if self.kvs[layer_idx].is_none() {
            self.kvs[layer_idx] = Some((key.clone(), value.clone()));
            return Ok((key.clone(), value.clone()));
        }

        let (mut k_cache, mut v_cache) = self.kvs[layer_idx].clone().unwrap();

        let mut k = Tensor::cat(&[k_cache, key.clone()], 2)?.contiguous()?;
        let mut v = Tensor::cat(&[v_cache, value.clone()], 2)?.contiguous()?;
        // k shape: [B, H, S, D] — dims()[2] is the seq_len dimension
        let k_seq_len = k.dims()[2];
        if k_seq_len > 8192 {
            k = k
                .narrow(2, k_seq_len - 8192, 8192)?
                .contiguous()?
        }
        let v_seq_len = v.dims()[2];
        if v_seq_len > 8192 {
            v = v
                .narrow(2, v_seq_len - 8192, 8192)?
                .contiguous()?
        }

        self.kvs[layer_idx] = Some((k.clone(), v.clone()));
        Ok((k, v))
    }

    /// 获取当前已缓存的 token 数（即 sequence length）
    pub fn get_seq_length(&self) -> usize {
        for kv in &self.kvs {
            if let Some((k, _)) = kv {
                if let Ok(dim) = k.dim(2) {
                    return dim;
                }
            }
        }
        0
    }
}