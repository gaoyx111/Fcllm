use load::ExpertTensorLoader;
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;
use RmsNorm::Qwen2MoeRMSNorm;
use configuration_qwen::Qwen2MoeConfig;
use RotaryEmbedding::Qwen2MoeRotaryEmbedding;
use Attention::Qwen2MoeAttention;
use MLP::Qwen2MoeMLP;


pub fn test_mlp() -> Result<()> {
    // let path = format!("model.layers.{}.mlp.experts.{}.weight", layer_id, expert_id);

    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    println!("device: {:?}", device);

    let config = Qwen2MoeConfig::new();

    let mut mlp = Qwen2MoeMLP::new(&config, 0, true);
    mlp.init_weights("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/", None, None)?;

    Ok(())
}
