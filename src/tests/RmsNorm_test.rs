use super::*;
use crate::load::ExpertTensorLoader;
use candle_core::{Device, Tensor, Result};
use std::path::Path;
use crate::RmsNorm::Qwen2MoeRMSNorm;
use crate::configuration_qwen::Qwen2MoeConfig;

pub fn test_rms_norm() -> Result<()> {
    // let path = format!("model.layers.{}.mlp.experts.{}.weight", layer_id, expert_id);

    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    println!("device: {:?}", device);

    let config = Qwen2MoeConfig::new();
    let mut norm = Qwen2MoeRMSNorm::new(config.hidden_size, device, config.rms_norm_eps);
    let expert_loader = ExpertTensorLoader::new("E:\\Rust\\model_weights\\Qwen\\Qwen1.5-MoE-A2.7B\\original");
    let tensor = expert_loader
        .pp("model")
        .pp("layers")
        .pp("0")      // 层编号
        .pp("input_layernorm")
        .pp("weight");
    println!("tensor: {:?}", tensor.root);
    let weight_path = "E:\\Rust\\model_weights\\Qwen\\Qwen1.5-MoE-A2.7B\\original\\model.layers.0.input_layernorm.weight";

    norm.init_weights(weight_path);
    println!("norm: {:?}", norm.weight.shape());

    Ok(())
}