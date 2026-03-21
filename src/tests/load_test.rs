// src/tests/load_test.rs
use super::*;
use crate::load::ExpertTensorLoader;
use candle_core::{Device, Tensor, Result};
use std::path::Path;

pub fn run_model_loader_test() -> Result<()> {
    //let expert_loader = ExpertTensorLoader::new("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/original");
    let expert_loader = ExpertTensorLoader::new("E:\\Rust\\model_weights\\Qwen\\Qwen1.5-MoE-A2.7B\\original");

    let path = Path::new("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/original");
    println!("Exists? {}", path.exists());

    // let path = format!("model.layers.{}.mlp.experts.{}.weight", layer_id, expert_id);

    let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    println!("device: {:?}", device);

    // let tensor = expert_loader
    //     .pp("model")
    //     .pp("layers")
    //     .pp("0")      // 层编号
    //     .pp("mlp")
    //     .pp("experts")
    //     .pp("0")      // expert 编号
    //     .pp("weight");
    let tensor = expert_loader
        .pp("model")
        .pp("layers")
        .pp("0")      // 层编号
        .pp("self_attn")
        .pp("q_proj")      // expert 编号
        .pp("weight");
    println!("tensor: {:?}", tensor.root);
        
    let tensor = tensor.load_tensor("tensor", &device)?;
    //let tensor = tensor.load_only_tensor(&device)?;
    println!("tensor shape: {:?}", tensor.shape());
    Ok(())
}