use crate::configuration_qwen::Qwen2MoeConfig;
use candle_core::{DType, Error, Result, Tensor};
use nvml_wrapper::Nvml;
use std::collections::HashMap;

const CPU_DEQUANT_CACHE_MIN_FREE_SYSTEM_MB: usize = 4 * 1024;
const AUTO_MIN_ONE_LAYER_CACHE_FREE_GPU_MB: f64 = 5_200.0;
const AUTO_MIN_TWO_LAYER_CACHE_FREE_GPU_MB: f64 = 5_900.0;

pub fn pack_8bit_u8(w_q: &Tensor) -> Result<Tensor> {
    w_q.to_dtype(DType::U8)
}

pub fn unpack_8bit_u8(w_q: &Tensor, dtype: DType) -> Result<Tensor> {
    w_q.to_dtype(dtype)
}

pub fn pack_4bit_u8(w_q: &Tensor) -> Result<Tensor> {
    let w_q = w_q.to_dtype(DType::U8)?;
    let len = w_q.shape().dims()[0];
    let step = len / 2;

    let vec = w_q.to_vec1::<u8>()?;

    let mut packed = Vec::with_capacity(step);

    for i in 0..step {
        let high = vec[i] << 4;
        let low = vec[i + step];
        packed.push(high | low);
    }

    let device = w_q.device();
    Tensor::from_vec(packed, &[step], device)
}

pub fn unpack_4bit_u8(w_q: &Tensor, dtype: DType) -> Result<Tensor> {
    let shape = w_q.shape();
    let dims = shape.dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::Msg(
            "unpack_4bit_u8 expects a 2D tensor".into(),
        ));
    }

    let step = dims[0];
    let cols = dims[1];
    let vec = w_q.to_vec2::<u8>()?;

    let mut high = Vec::with_capacity(step * cols);
    let mut low = Vec::with_capacity(step * cols);

    for row in &vec {
        for &val in row {
            high.push((val >> 4) & 0x0F);
        }
    }
    for row in &vec {
        for &val in row {
            low.push(val & 0x0F);
        }
    }

    let mut unpacked = Vec::with_capacity(2 * step * cols);
    unpacked.extend(high);
    unpacked.extend(low);

    let new_shape = (2 * step, cols);
    Tensor::from_vec(unpacked, new_shape, w_q.device())?.to_dtype(dtype)
}

pub fn pack_2bit_u8(w_q: &Tensor) -> Result<Tensor> {
    let w_q = w_q.to_dtype(DType::U8)?;
    let shape = w_q.shape();
    let len = shape.dims()[0];
    if len % 4 != 0 {
        return Err(candle_core::Error::Msg(
            "Length of tensor must be divisible by 4 for 2-bit packing".into(),
        ));
    }

    let vec = w_q.to_vec1::<u8>()?;
    let step = len / 4;
    let mut packed = Vec::with_capacity(step);

    for i in 0..step {
        let a = vec[i] & 0x03;
        let b = vec[i + step] & 0x03;
        let c = vec[i + 2 * step] & 0x03;
        let d = vec[i + 3 * step] & 0x03;

        let byte = (a << 6) | (b << 4) | (c << 2) | d;
        packed.push(byte);
    }
    Tensor::from_vec(packed, &[step], w_q.device())
}

pub fn unpack_2bit_u8(w_q: &Tensor, dtype: DType) -> Result<Tensor> {
    let shape = w_q.shape();
    let dims = shape.dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::Msg(
            "unpack_2bit_u8 expects a 2D tensor".into(),
        ));
    }

    let step = dims[0];
    let cols = dims[1];
    let vec = w_q.to_vec2::<u8>()?;

    // unpack 4x，每个字节包含 4 个 2-bit
    let mut unpacked = vec![0u8; 4 * step * cols];

    for (i, row) in vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let idx_base = i + j * step;
            unpacked[0 * step * cols + idx_base] = (val >> 6) & 0x03;
            unpacked[1 * step * cols + idx_base] = (val >> 4) & 0x03;
            unpacked[2 * step * cols + idx_base] = (val >> 2) & 0x03;
            unpacked[3 * step * cols + idx_base] = val & 0x03;
        }
    }
    Tensor::from_vec(unpacked, (4 * step, cols), w_q.device())?.to_dtype(dtype)
}

pub fn pack_3bit_32(w_q_in: &Tensor) -> Result<Tensor> {
    let shape = w_q_in.shape();
    let rows = shape.dims()[0];
    let cols = shape.dims()[1];

    let padded_len = ((rows + 9) / 10) * 10; // 向上取整到10的倍数
    let step = padded_len / 10;

    // 构造 pad 后的 (padded_len, cols) 张量
    let mut w_q = vec![0i64; padded_len * cols];
    let orig = w_q_in.to_vec2::<i64>()?;

    // 拷贝原始值
    for i in 0..rows {
        for j in 0..cols {
            w_q[i * cols + j] = orig[i][j];
        }
    }

    // 每个输出行是 10 个 3-bit 数拼接成 1 个 i64
    let mut packed = vec![0i64; step * cols];
    for c in 0..cols {
        for i in 0..step {
            let mut val = 0i64;
            for k in 0..10 {
                let idx = (k * step + i) * cols + c;
                let v = w_q[idx] & 0x7; // 取 3-bit
                val |= v << (27 - 3 * k);
            }
            packed[i * cols + c] = val;
        }
    }
    Tensor::from_vec(packed, (step, cols), w_q_in.device())
}

pub fn unpack_3bit_32(w_q: &Tensor, dtype: DType) -> Result<Tensor> {
    let shape = w_q.shape();
    let step = shape.dims()[0];
    let cols = shape.dims()[1];

    let vec = w_q.to_vec2::<i64>()?;
    let mut unpacked = vec![0u8; step * cols * 10];

    for row in 0..step {
        for col in 0..cols {
            let val = vec[row][col];
            let base = (row * 10) * cols + col;

            unpacked[base + 0 * cols] = ((val >> 27) & 0x7) as u8;
            unpacked[base + 1 * cols] = ((val >> 24) & 0x7) as u8;
            unpacked[base + 2 * cols] = ((val >> 21) & 0x7) as u8;
            unpacked[base + 3 * cols] = ((val >> 18) & 0x7) as u8;
            unpacked[base + 4 * cols] = ((val >> 15) & 0x7) as u8;
            unpacked[base + 5 * cols] = ((val >> 12) & 0x7) as u8;
            unpacked[base + 6 * cols] = ((val >> 9) & 0x7) as u8;
            unpacked[base + 7 * cols] = ((val >> 6) & 0x7) as u8;
            unpacked[base + 8 * cols] = ((val >> 3) & 0x7) as u8;
            unpacked[base + 9 * cols] = (val & 0x7) as u8;
        }
    }
    Tensor::from_vec(unpacked, (step * 10, cols), w_q.device())?.to_dtype(dtype)
}

pub fn pack_1bit_u8(w_q: &Tensor) -> Result<Tensor> {
    let w_q = w_q.to_dtype(DType::U8)?;
    let len = w_q.shape().dims()[0];
    assert!(len % 8 == 0, "Length must be divisible by 8");
    let step = len / 8;

    // 转成 Vec<u8> 方便操作
    let vec = w_q.to_vec1::<u8>()?;
    let mut packed = Vec::with_capacity(step);

    for i in 0..step {
        // 对应 Python 代码中的按位左移和或操作
        let mut byte = 0u8;
        byte |= vec[i] << 7;
        byte |= vec[i + step] << 6;
        byte |= vec[i + 2 * step] << 5;
        byte |= vec[i + 3 * step] << 4;
        byte |= vec[i + 4 * step] << 3;
        byte |= vec[i + 5 * step] << 2;
        byte |= vec[i + 6 * step] << 1;
        byte |= vec[i + 7 * step];
        packed.push(byte);
    }
    Tensor::from_vec(packed, (step,), w_q.device())
}

pub fn unpack_1bit_u8(w_q: &Tensor, dtype: DType) -> Result<Tensor> {
    let shape = w_q.shape();
    let dims = shape.dims();
    if dims.len() != 2 {
        return Err(candle_core::Error::Msg(
            "unpack_1bit_u8 expects a 2D tensor".into(),
        ));
    }
    let step = dims[0];
    let cols = dims[1];

    let vec = w_q.to_vec2::<u8>()?;
    let mut unpacked = Vec::with_capacity(8 * step * cols);

    for bit_pos in (0..8).rev() {
        for row in &vec {
            for &val in row {
                // 按位提取对应 bit
                let bit = (val >> bit_pos) & 1;
                unpacked.push(bit);
            }
        }
    }
    Tensor::from_vec(unpacked, (8 * step, cols), w_q.device())?.to_dtype(dtype)
}

pub const SUPPORTED_BITS: &[usize] = &[8, 4, 3, 2, 1];

pub fn pack(w_q: &Tensor, bits: usize) -> Result<Tensor> {
    match bits {
        8 => pack_8bit_u8(w_q),
        4 => pack_4bit_u8(w_q),
        3 => pack_3bit_32(w_q),
        2 => pack_2bit_u8(w_q),
        1 => pack_1bit_u8(w_q),
        _ => Err(Error::Msg(format!("Unsupported bits: {}", bits))),
    }
}

pub fn unpack(w_q: &Tensor, bits: usize, dtype: DType) -> Result<Tensor> {
    match bits {
        8 => unpack_8bit_u8(w_q, dtype),
        4 => unpack_4bit_u8(w_q, dtype),
        3 => unpack_3bit_32(w_q, dtype),
        2 => unpack_2bit_u8(w_q, dtype),
        1 => unpack_1bit_u8(w_q, dtype),
        _ => Err(Error::Msg(format!("Unsupported bits: {}", bits))),
    }
}

pub const KB: usize = 1 << 10; // 1024
pub const MB: usize = 1 << 20; // 1024 * 1024
pub const GB: usize = 1 << 30; // 1024 * 1024 * 1024
pub const T: f64 = 1e12; // 万亿级常量，用于浮点计算

// 获取指定 cuda 设备编号的显存使用和总显存（单位 MB）
pub fn get_gpu_memory_usage(device_id: u32) -> Result<(usize, usize)> {
    let nvml =
        Nvml::init().map_err(|e| candle_core::Error::Msg(format!("NVML init error: {e}")))?;
    let device = nvml
        .device_by_index(device_id)
        .map_err(|e| candle_core::Error::Msg(format!("Device access error: {e}")))?;
    let mem_info = device
        .memory_info()
        .map_err(|e| candle_core::Error::Msg(format!("Memory info error: {e}")))?;
    Ok((mem_info.used as usize / MB, mem_info.total as usize / MB))
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(target_os = "windows")]
unsafe extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
}

#[cfg(target_os = "windows")]
pub fn system_memory_usage_mb() -> Option<(usize, usize)> {
    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };

    let ok = unsafe { GlobalMemoryStatusEx(&mut status as *mut MemoryStatusEx) };
    if ok == 0 || status.ull_total_phys == 0 {
        return None;
    }

    let total_mb = (status.ull_total_phys as usize) / MB;
    let free_mb = (status.ull_avail_phys as usize) / MB;
    Some((total_mb.saturating_sub(free_mb), total_mb))
}

#[cfg(not(target_os = "windows"))]
pub fn system_memory_usage_mb() -> Option<(usize, usize)> {
    None
}

pub fn cpu_dequant_cache_has_memory_headroom() -> bool {
    let Some((used_mb, total_mb)) = system_memory_usage_mb() else {
        return true;
    };
    total_mb.saturating_sub(used_mb) > CPU_DEQUANT_CACHE_MIN_FREE_SYSTEM_MB
}

pub fn memory_cost_qwen(
    config: &Qwen2MoeConfig,
    memory_budget: usize,
) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mb_f64 = MB as f64;

    let auto_memory_budget = memory_budget == 0;
    let memory_budget = if auto_memory_budget {
        let device_id = 0u32;
        match get_gpu_memory_usage(device_id) {
            Ok((used_mb, total_mb)) => {
                if total_mb > used_mb {
                    (total_mb - used_mb) as f64
                } else {
                    panic!("GPU memory usage anomaly: used > total");
                }
            }
            Err(e) => {
                eprintln!("Failed to query GPU memory: {:?}", e);
                8_000.0
            }
        }
    } else {
        (memory_budget as f64) * 1024.0
    };

    let seq_len = 1024_f64;
    let num_hidden_layers = config.num_hidden_layers as f64;
    let hidden_size = config.hidden_size as f64;
    let vocab_size = config.vocab_size as f64;
    let moe_intermediate_size = config.moe_intermediate_size as f64;
    let shared_expert_intermediate_size = config.shared_expert_intermediate_size as f64;
    let num_experts = config.num_experts as f64;

    let num_heads = config.num_attention_heads as f64;
    let head_dim = hidden_size / num_heads;
    let num_key_value_heads = config.num_key_value_heads as f64;

    let embed = (vocab_size * hidden_size * 2.0 * 2.0) / mb_f64;
    let attention = (2.0
        * (hidden_size * num_heads * head_dim + hidden_size * num_key_value_heads * head_dim)
        * num_hidden_layers
        * 2.0)
        / mb_f64;
    let attn_bias =
        ((num_heads * head_dim + 2.0 * num_key_value_heads * head_dim) * num_hidden_layers * 2.0)
            / mb_f64;
    let rotary_embedding = shared_rotary_embedding_cache_mb(config);
    let norm = ((2.0 * hidden_size * num_hidden_layers + hidden_size) * 2.0) / mb_f64;

    let shared_expert = (3.0 * hidden_size * shared_expert_intermediate_size * 2.0) / mb_f64;
    let shared_expert_gate = (hidden_size * num_hidden_layers * num_hidden_layers * 2.0) / mb_f64;

    let expert = (3.0 * hidden_size * moe_intermediate_size * 2.0) / mb_f64;
    let expert_gate = (hidden_size * num_experts * num_hidden_layers * 2.0) / mb_f64;

    let kv = (2.0 * seq_len * num_hidden_layers * hidden_size * 2.0) / mb_f64;
    let hidden = (2.0 * seq_len * hidden_size * 2.0) / mb_f64;

    let mut available_memory = memory_budget
        - embed
        - attention
        - attn_bias
        - rotary_embedding
        - norm
        - shared_expert_gate
        - expert_gate
        - kv
        - hidden;

    available_memory -= shared_expert * config.num_hidden_layers as f64;

    let meta_data = 0.3
        * (3.0 * config.num_experts as f64 + 3.0 + 4.0 + 2.0 + 2.0 + 2.0)
        * config.num_hidden_layers as f64;
    available_memory = available_memory - meta_data - 1000.0;

    let zero_scale = (2.0 * (moe_intermediate_size / 64.0 * hidden_size) * 1.0 * 4.0) / mb_f64;
    let expert_int4 = expert / 4.0 + 3.0 * zero_scale;
    let mut quan_map: HashMap<usize, usize> = HashMap::new();
    let mut offload_map: HashMap<usize, usize> = HashMap::new();

    for i in 0..config.num_hidden_layers {
        quan_map.insert(i, 0);
        offload_map.insert(i, 0);
    }

    if available_memory < 0.0 {
        let minimum_resident_layers =
            minimum_auto_resident_cache_layers(auto_memory_budget, memory_budget, config);
        eprintln!(
            "[memory] available expert memory is negative ({available_memory:.2} MB); preserving {minimum_resident_layers} early layer(s) for ARC expert cache"
        );
        for i in 0..config.num_hidden_layers {
            quan_map.insert(i, 4);
            if i < minimum_resident_layers {
                offload_map.insert(i, 0);
            } else {
                offload_map.insert(i, config.num_experts);
            }
        }
        return (offload_map, quan_map);
    }

    let planned_layers = config.num_hidden_layers as f64;

    if available_memory > planned_layers * num_experts * expert {
        return (offload_map, quan_map);
    } else if available_memory > planned_layers * num_experts * expert_int4 {
        let remain = available_memory - planned_layers * num_experts * expert_int4;
        let fp16_layers = (remain / (num_experts * (expert - expert_int4))) as usize;

        for i in 0..config.num_hidden_layers {
            if i >= fp16_layers {
                quan_map.insert(i, 4);
            }
        }
        return (offload_map, quan_map);
    } else {
        let cache_num = (available_memory / expert_int4) as usize;
        let all_cache_layers = cache_num / config.num_experts;
        let partial_cache = cache_num.saturating_sub(config.num_experts * all_cache_layers);

        if all_cache_layers < 4 {
            for i in 0..config.num_hidden_layers {
                quan_map.insert(i, 4);
                if i < all_cache_layers {
                    offload_map.insert(i, 0);
                } else if i == all_cache_layers && partial_cache > 0 {
                    offload_map.insert(i, config.num_experts - partial_cache);
                } else {
                    offload_map.insert(i, config.num_experts);
                }
            }
        } else {
            let tail_layers = config.num_hidden_layers.saturating_sub(4).max(1);
            let cache_deep =
                ((cache_num - 4 * config.num_experts) / tail_layers).min(config.num_experts);
            for i in 0..config.num_hidden_layers {
                quan_map.insert(i, 4);
                if i < 4 {
                    offload_map.insert(i, 0);
                } else {
                    offload_map.insert(i, config.num_experts - cache_deep);
                }
            }
        }
        return (offload_map, quan_map);
    }
}

fn shared_rotary_embedding_cache_mb(config: &Qwen2MoeConfig) -> f64 {
    let head_dim = config.hidden_size as f64 / config.num_attention_heads as f64;
    ((head_dim / 2.0 + head_dim * config.max_position_embeddings as f64 * 2.0) * 4.0) / MB as f64
}

fn minimum_auto_resident_cache_layers(
    auto_memory_budget: bool,
    free_gpu_mb: f64,
    config: &Qwen2MoeConfig,
) -> usize {
    if !auto_memory_budget {
        return 0;
    }

    let layers = if free_gpu_mb >= AUTO_MIN_TWO_LAYER_CACHE_FREE_GPU_MB {
        2
    } else if free_gpu_mb >= AUTO_MIN_ONE_LAYER_CACHE_FREE_GPU_MB {
        1
    } else {
        0
    };

    layers.min(config.num_hidden_layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_planner_can_quantize_layer0_when_budget_is_tight() {
        let config = Qwen2MoeConfig::new();
        let (offload_map, quan_map) = memory_cost_qwen(&config, 8);

        assert_eq!(quan_map.get(&0), Some(&4));
        assert!(offload_map[&0] <= config.num_experts);
        assert_eq!(quan_map.len(), config.num_hidden_layers);
        assert_eq!(offload_map.len(), config.num_hidden_layers);
    }

    #[test]
    fn auto_memory_planner_keeps_small_resident_cache_when_formula_is_pessimistic() {
        let config = Qwen2MoeConfig::new();

        assert_eq!(
            minimum_auto_resident_cache_layers(true, 6_500.0, &config),
            2
        );
        assert_eq!(
            minimum_auto_resident_cache_layers(true, 6_100.0, &config),
            2
        );
        assert_eq!(
            minimum_auto_resident_cache_layers(true, 5_600.0, &config),
            1
        );
        assert_eq!(
            minimum_auto_resident_cache_layers(true, 5_000.0, &config),
            0
        );
        assert_eq!(
            minimum_auto_resident_cache_layers(false, 8_000.0, &config),
            0
        );
    }

    #[test]
    fn shared_rotary_budget_does_not_scale_with_layer_count() {
        let mut small = Qwen2MoeConfig::new();
        small.num_hidden_layers = 1;
        let mut large = small.clone();
        large.num_hidden_layers = 24;

        assert_eq!(
            shared_rotary_embedding_cache_mb(&small),
            shared_rotary_embedding_cache_mb(&large)
        );
    }
}
