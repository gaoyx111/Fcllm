use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use candle_core::{Tensor, Device, DType, Error};
use crate::models::Qwen::modeling_qwen_moe::Qwen2MoeForCausalLM;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    #[arg(long, default_value = "Qwen/Qwen1.5-MoE-A2.7B")]
    pub model: String,

    #[arg(long, default_value = "./model_weights")]
    pub path: String,

    #[arg(long, default_value_t = true)]
    pub early_stopping: bool,

    #[arg(long, default_value_t = 1)]
    pub min_length: usize,

    #[arg(long, default_value_t = 256)]
    pub max_length: usize,

    #[arg(long, default_value_t = true)]
    pub pin_weight: bool,

    #[arg(long, default_value_t = 0)]
    pub memory_budget: usize,

    #[arg(long, default_value = "cuda:0")]
    pub device: String,

    #[arg(long, default_value_t = true)]
    pub overlap: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化 device
    let device = match args.device.as_str() {
        "cuda:0" => Device::cuda_if_available(0)?,
        _ => Device::Cpu,
    };

    // 加载 tokenizer
    let tokenizer_path = PathBuf::from(&args.path)
        .join("Qwen")
        .join("Qwen1.5-MoE-A2.7B")
        .join("tokenizer")
        .join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 创建模型
    let mut model = Qwen2MoeForCausalLM::new(&args)?;
    model.init_weights()?; // 加载权重
    // model.eval(); // 如果你有对应方法

    // 构造输入
    let input_prompt = "Hey, are you conscious? Can you talk to me?";
    let encoding = tokenizer.encode(input_prompt, true)?;
    let input_ids = Tensor::new(&encoding.get_ids(), &device)?.unsqueeze(0)?; // shape: [1, T]
    let attention_mask = Tensor::ones(&[1, input_ids.dims()[1]], DType::Int64, &device)?;

    // warmup
    let _ = model.generate(&input_ids, Some(&attention_mask), None)?;

    // 推理计时
    let start = Instant::now();
    let (output_ids, prefill_time) = model.generate(&input_ids, Some(&attention_mask), Some("decoding"))?;
    let latency = start.elapsed().as_secs_f64();
    println!("latency = {}", latency);

    // 解码输出
    let output_tokens = output_ids.to_vec1::<i64>()?[0].clone();
    let output = tokenizer.decode(output_tokens, true)?;
    println!("{}", output);

    Ok(())
}
