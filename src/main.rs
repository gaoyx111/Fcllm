mod configuration_qwen;
mod load;
mod RmsNorm;
mod RotaryEmbedding;
mod Attention;
mod linear;
mod Cache;
mod MLP;
mod quantizer;
mod utils;
mod expert_ARC_cahce;
mod SparseMoeBlock;
mod DecoderLayer;
mod Model;
mod nn_embedding;
mod ForCausalLM;
mod args;


use candle_core::{shape, DType, Device, IndexOp, Result, Tensor, D, Error, CudaDevice};
use std::path::PathBuf;
use tokenizers::{Tokenizer, Encoding};
use std::time::Instant;

use load::ExpertTensorLoader;
use RmsNorm::Qwen2MoeRMSNorm;
use configuration_qwen::Qwen2MoeConfig;
use RotaryEmbedding::Qwen2MoeRotaryEmbedding;
use Attention::Qwen2MoeAttention;
use MLP::Qwen2MoeMLP;
use SparseMoeBlock::Qwen2MoeSparseMoeBlock;
use args::Args;
use clap::Parser;
use ForCausalLM::Qwen2MoeForCausalLM;



fn main() -> Result<()> {
    // let path = format!("model.layers.{}.mlp.experts.{}.weight", layer_id, expert_id);

    // let device = Device::cuda_if_available(0).unwrap_or_else(|_| Device::Cpu);
    // println!("device: {:?}", device);

    // let config = Qwen2MoeConfig::new();
    // let mut att = Qwen2MoeAttention::new(&config, Some(0))?;
    // att.init_weights("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/")?;

    // let mut mlp = Qwen2MoeMLP::new(&config, 0, true);
    // mlp.init_weights("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/", None, None)?;

    // let mut sparse_moe = Qwen2MoeSparseMoeBlock::new(&config, 0)?;
    // sparse_moe.init_weights("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/")?;

    // let loader = ExpertTensorLoader::new("E:/Rust/model_weights/Qwen/Qwen1.5-MoE-A2.7B/original/model.embed_tokens.weight");
    // let embed_tokens = loader.load_tensor("tensor", &device)?;
    // println!("embed_tokens: {:?}, shape: {:?}", embed_tokens, embed_tokens.shape());

    let args = Args::parse();
    let mut model = Qwen2MoeForCausalLM::new(args)?;
    let device = model.device.clone();
    model.init_weights()?;

    let tokenizer_path = PathBuf::from("model_weights/Qwen/Qwen1.5-MoE-A2.7B/tokenizer")
        .join("models--Qwen--Qwen1.5-MoE-A2.7B")
        .join("snapshots")
        .join("1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
        .join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer");

     // 编码输入
    let prompt = "Hey, are you conscious? Can you talk to me?";
    let encoding: Encoding = tokenizer.encode(prompt, true).unwrap();
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    // 转换为 Tensor
    let input_ids_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)?.unsqueeze(0)?;

    // 正式生成
    let start = Instant::now();
    let (output_ids_tensor, prefill_time) =
        model.generate(&input_ids_tensor, Some(attention_mask_tensor), Some("decoding"))?;
    let end = Instant::now();
    println!("latency = {:.2?}", end - start);

    // output_ids_tensor shape: [1, N] — squeeze to 1D first
    let output_flat = output_ids_tensor.squeeze(0)?;
    let output_ids: Vec<u32> = output_flat.to_vec1::<u32>()?;
    let decoded = tokenizer.decode(&output_ids, true)
    .map_err(|e| candle_core::Error::Msg(format!("decode error: {e}")))?;
    println!("Output = {}", decoded);

    Ok(())
}
