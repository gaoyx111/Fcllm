use anyhow::{Result, bail};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use clap::Parser;
use std::{path::{PathBuf}, time::Instant, fs};
use tokenizers::Tokenizer;

/// 命令行参数
#[derive(Parser, Debug)]
struct Args {
    /// 模型名称（本地文件夹名称，如 Qwen/Qwen1.5-MoE-A2.7B）
    #[arg(long, default_value = "Qwen/Qwen1.5-MoE-A2.7B")]
    model: String,

    /// 本地模型权重根路径
    #[arg(long, default_value = "model_weights")]
    path: String,

    /// 推理设备
    #[arg(long, default_value = "cuda:0")]
    device: String,
}

fn load_local_model_and_tokenizer(model_name: &str, model_base_path: &str, device: &Device)
    -> Result<(Tokenizer, candle_transformers::models::qwen::Qwen)> {
    let base_dir = PathBuf::from(model_base_path).join(model_name);
    let original_dir = base_dir.join("original");
    let tokenizer_path = base_dir.join("tokenizer.json");
    let config_path = base_dir.join("config.json");

    // 加载 tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    // 加载 config
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: candle_transformers::models::qwen::Config = serde_json::from_str(&config_str)?;

    // 找到所有 *.safetensors 权重文件
    let mut safetensor_paths = vec![];
    for entry in fs::read_dir(&original_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            safetensor_paths.push(path);
        }
    }

    if safetensor_paths.is_empty() {
        bail!("未找到原始权重文件，请确认路径: {}", original_dir.display());
    }

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_paths, DType::F32, device)?
    };
    let model = candle_transformers::models::qwen::Qwen::load(vb, &config)?;
    Ok((tokenizer, model))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.device.starts_with("cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    let (tokenizer, model) = load_local_model_and_tokenizer(&args.model, &args.path, &device)?;

    let input_text = "Hey, are you conscious? Can you talk to me?";
    let encoded = tokenizer.encode(input_text, true).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = Tensor::new(encoded.get_ids(), &device)?.unsqueeze(0)?; // [1, seq_len]

    println!("Running inference...");
    let start = Instant::now();
    let output = model.forward(&input_ids)?;
    let duration = start.elapsed();

    println!("Inference took: {:?}", duration);
    println!("Output dims: {:?}", output.dims());

    Ok(())
}
