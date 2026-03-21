use super::*;
use crate::load::ExpertTensorLoader;
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;
use crate::RotaryEmbedding::Qwen2MoeRotaryEmbedding;
use crate::configuration_qwen::Qwen2MoeConfig;

pub fn test_rotary_embedding() -> Result<()> {
    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    println!("device: {:?}", device);

    let config = Qwen2MoeConfig::new();
    let mut rotary_emb = Qwen2MoeRotaryEmbedding::new(config.hidden_size/config.num_attention_heads,config.max_position_embeddings, config.rope_theta, DType::F32, &device)?;
    println!("rotary_emb: {:?}", rotary_emb.inv_freq.shape());
    
    Ok(())
}