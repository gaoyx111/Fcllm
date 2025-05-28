use std::collections::HashMap;
use anyhow::{Result, bail};
use nvml_wrapper::Nvml;
use candle_core::{Tensor, DType, IndexOp};
use candle_core::utils::{BitAndOp, BitOrOp, ShlOp, ShrOp};
use crate::models::Qwen::configuration_qwen::Qwen2MoeConfig;

const MB: f64 = 1024.0 * 1024.0;

pub fn str2bool(v: &str) -> Result<bool> {
    match v.to_lowercase().as_str() {
        "yes" | "true" | "t" | "y" | "1" => Ok(true),
        "no"  | "false"| "f" | "n" | "0" => Ok(false),
        _ => Err(anyhow!("Boolean value expected.")),
    }
}

pub fn memory_cost_qwen(config: &Qwen2MoeConfig, memory_budget_gb: usize) -> Result<(HashMap<usize, usize>, HashMap<usize, usize>)> {
    let nvml = Nvml::init()?;
    let device_str = config.device.as_ref().ok_or_else(|| anyhow::anyhow!("Device not specified"))?;
    let device_str = format!("{:?}", device_str); // 将 Device 转成 String
    let device_index: u32 = device_str
        .strip_prefix("cuda:")
        .ok_or_else(|| anyhow::anyhow!("Invalid device string, expected prefix 'cuda:'"))?
        .parse()
        .map_err(|_| anyhow::anyhow!("Failed to parse device index"))?;
    let device_handle = nvml.device_by_index(device_index)?;
    let total_memory = device_handle.memory_info()?.total as f64;
    let used_memory = device_handle.memory_info()?.used as f64;

    let mut memory_budget_mb = if memory_budget_gb == 0 {
        (total_memory - used_memory) / MB
    } else {
        (memory_budget_gb as f64) * 1024.0
    };

    let seq_len = 1024.0;
    let num_hidden_layers = config.num_hidden_layers as f64;
    let hidden_size = config.hidden_size as f64;
    let vocab_size = config.vocab_size as f64;
    let moe_intermediate_size = config.moe_intermediate_size as f64;
    let shared_expert_intermediate_size = config.shared_expert_intermediate_size as f64;
    let num_experts = config.num_experts as f64;
    let num_heads = config.num_attention_heads as f64;
    let head_dim = hidden_size / num_heads;
    let num_key_value_heads = config.num_key_value_heads as f64;
    let max_position_embeddings = config.max_position_embeddings as f64;

    let embed = (vocab_size * hidden_size * 2.0 * 2.0) / MB;
    let attention = (2.0 * (hidden_size * num_heads * head_dim + hidden_size * num_key_value_heads * head_dim) * num_hidden_layers * 2.0) / MB;
    let attn_bias = ((num_heads * head_dim + 2.0 * num_key_value_heads * head_dim) * num_hidden_layers * 2.0) / MB;
    let rotary_embedding = ((head_dim / 2.0 + head_dim * max_position_embeddings * 2.0) * num_hidden_layers * 4.0) / MB;
    let norm = ((2.0 * hidden_size * num_hidden_layers + hidden_size) * 2.0) / MB;
    let shared_expert = (3.0 * hidden_size * shared_expert_intermediate_size * 2.0) / MB;
    let shared_expert_gate = (hidden_size * num_hidden_layers * num_hidden_layers * 2.0) / MB;
    let expert = (3.0 * hidden_size * moe_intermediate_size * 2.0) / MB;
    let expert_gate = (hidden_size * num_experts * num_hidden_layers * 2.0) / MB;
    let kv = (2.0 * seq_len * num_hidden_layers * hidden_size * 2.0) / MB;
    let hidden = (2.0 * seq_len * hidden_size * 2.0) / MB;

    let mut available_memory = memory_budget_mb - embed - attention - attn_bias - rotary_embedding - norm - shared_expert_gate - expert_gate - kv - hidden;
    available_memory -= shared_expert * 24.0;

    let meta_data = 0.3 * (3.0 * 60.0 + 3.0 + 4.0 + 2.0 + 2.0 + 2.0) * 24.0;
    available_memory -= meta_data + 1000.0;

    if available_memory < 0.0 {
        bail!("{} memory is not enough for dense.", available_memory);
    }

    let zero_scale = (2.0 * ((moe_intermediate_size / 64.0) * hidden_size) * 1.0 * 4.0) / MB;
    let expert_int4 = expert / 4.0 + 3.0 * zero_scale;

    let mut quan_map = HashMap::new();
    let mut offload_map = HashMap::new();

    for i in 0..config.num_hidden_layers {
        quan_map.insert(i, 0);
        offload_map.insert(i, 0);
    }

    if available_memory > num_hidden_layers * 60.0 * expert {
        Ok((offload_map, quan_map))
    } else if available_memory > num_hidden_layers * 60.0 * expert_int4 {
        available_memory -= num_hidden_layers * 60.0 * expert_int4;
        let fp16_layers = (available_memory / (60.0 * expert - 60.0 * expert_int4)).floor() as usize;
        for i in 0..config.num_hidden_layers {
            if i >= fp16_layers {
                quan_map.insert(i, 4);
            }
        }
        Ok((offload_map, quan_map))
    } else {
        let cache_num = (available_memory / expert_int4).floor();
        let all_cache_layers = (cache_num / 60.0).floor() as usize;

        if all_cache_layers < 4 {
            for i in 0..config.num_hidden_layers {
                quan_map.insert(i, 4);
                if i < all_cache_layers {
                    offload_map.insert(i, 0);
                } else if i == all_cache_layers {
                    let offload_value = (60.0 - (cache_num - 60.0 * (all_cache_layers as f64))) as usize;
                    offload_map.insert(i, offload_value);
                } else {
                    offload_map.insert(i, 60);
                }
            }
        } else {
            let cache_deep = ((cache_num - 4.0 * 60.0) / (num_hidden_layers - 4.0)).floor();
            for i in 0..config.num_hidden_layers {
                quan_map.insert(i, 4);
                if i < 4 {
                    offload_map.insert(i, 0);
                } else {
                    let offload_value = (60.0 - cache_deep) as usize;
                    offload_map.insert(i, offload_value);
                }
            }
        }
        Ok((offload_map, quan_map))
    }
}


pub fn pack_8bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    w.to_dtype(DType::U8)
}

pub fn unpack_8bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    w.to_dtype(DType::U8)
}

pub fn pack_4bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    let w = w.to_dtype(DType::U8)?;
    let shape = w.shape();
    let step = shape.dims()[0] / 2;
    let high = w.i(..step)?.shl(4)?;
    let low = w.i(step..)?.clone();
    high.bitor(&low)
}

pub fn unpack_4bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    let shape = w.shape();
    let step = shape.dims()[0];
    let high = w.bitand(0b11110000u8)?.shr(4)?;
    let low = w.bitand(0b00001111u8)?;
    Tensor::cat(&[&high, &low], 0)
}

pub fn pack_2bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    use candle_numpy::ops::{BitOrOp, ShlOp};

    let w = w.to_dtype(DType::U8)?;
    let shape = w.shape();
    let step = shape.dims()[0] / 4;
    let a = w.i(0..step)?.shl(6)?;
    let b = w.i(step..2 * step)?.shl(4)?;
    let c = w.i(2 * step..3 * step)?.shl(2)?;
    let d = w.i(3 * step..)?.clone();
    a.bitor(&b)?.bitor(&c)?.bitor(&d)
}

pub fn unpack_2bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    use candle_numpy::ops::{BitAndOp, ShrOp};

    let a = w.bitand(0b11000000u8)?.shr(6)?;
    let b = w.bitand(0b00110000u8)?.shr(4)?;
    let c = w.bitand(0b00001100u8)?.shr(2)?;
    let d = w.bitand(0b00000011u8)?;
    Tensor::cat(&[&a, &b, &c, &d], 0)
}

pub fn pack_3bit_32(w: &Tensor) -> candle_core::Result<Tensor> {
    let shape = w.shape();
    if shape.rank() != 2 {
        bail!("Expected 2D tensor for 3-bit pack, got shape {:?}", shape);
    }

    let (n, d) = (shape.dims()[0], shape.dims()[1]);
    let padded_n = ((n + 9) / 10) * 10;
    let device = w.device();

    let mut padded = Tensor::zeros((padded_n, d), DType::U32, device)?;
    padded.i(0..n)?.copy_(&w.to_dtype(DType::U32)?)?;

    let step = padded_n / 10;
    let mut packed = padded.i(0..step)?.shl(27)?;
    for i in 1..10 {
        let shift = 27 - 3 * i;
        let chunk = padded.i(i * step..(i + 1) * step)?;
        packed = packed.bitor(&chunk.shl(shift)?)?;
    }

    Ok(packed)
}

pub fn unpack_3bit_32(packed: &Tensor) -> candle_core::Result<Tensor> {
    let shape = packed.shape();
    if shape.rank() != 2 {
        return Err(candle_core::Error::Msg("Expected 2D packed tensor".into()));
    }

    let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
    let mut parts = Vec::with_capacity(10);
    for i in 0..10 {
        let shift = 27 - 3 * i;
        let mask = 0b111 << shift;
        parts.push(packed.bitand_scalar(mask)?.shr_scalar(shift)?);
    }

    let stacked = Tensor::stack(&parts, 0)?;
    stacked.permute((1, 0, 2))?.reshape((rows * 10, cols))
}

pub fn pack_1bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    let w = w.to_dtype(DType::U8)?;
    let step = w.shape().dims()[0] / 8;
    let mut packed = w.i(0..step)?.shl(7)?;
    for i in 1..8 {
        packed = packed.bitor(&w.i(i * step..(i + 1) * step)?.shl(7 - i as u32)?)?;
    }
    Ok(packed)
}

pub fn unpack_1bit_u8(w: &Tensor) -> candle_core::Result<Tensor> {
    let step = w.shape().dims()[0];
    let mut result = vec![];
    for i in 0..8 {
        let mask = 1 << (7 - i);
        result.push(w.bitand(mask)?.shr((7 - i) as u32)?);
    }
    Tensor::cat(&result.iter().collect::<Vec<_>>(), 0)
}
