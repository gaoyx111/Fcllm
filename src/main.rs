use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use candle_core::{Tensor, Device, DType, Error};
//use crate::models::Qwen::modeling_qwen_moe::Qwen2MoeForCausalLM;
use anyhow::Result;
use std::process::{Command, Stdio};
use std::io::{BufReader, BufRead};

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

#[cfg(feature = "version_min_than_one")]
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


fn main() {
    // 获取项目根目录并转换为 PathBuf
    let project_root = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("无法获取项目根目录")
    );
    
    // 构建 Python 脚本的绝对路径
    let script_path = project_root.join("Fate").join("main.py");
    let script_path_str = script_path.to_str().expect("无法转换脚本路径为字符串");

    // 创建子进程并捕获标准输出和错误
    let mut child = Command::new(r#"C:\Users\Lenovo\anaconda3\envs\especially_for_LLM\python.exe"#)
        .arg(script_path_str)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("无法启动 Python 进程");

    // 获取标准输出和错误的句柄
    let stdout = child.stdout.take().expect("无法获取标准输出");
    let stderr = child.stderr.take().expect("无法获取标准错误");

    // 使用线程分别处理标准输出和错误
    let stdout_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line) = line {
                println!("{}", line);
            }
        }
    });

    let stderr_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(line) = line {
                eprintln!("{}", line); // 使用 eprintln! 打印错误输出
            }
        }
    });

    // 等待子进程完成
    let status = child.wait().expect("无法等待 Python 进程结束");
    
    // 等待输出处理线程完成
    stdout_handle.join().expect("标准输出线程异常退出");
    stderr_handle.join().expect("标准错误线程异常退出");

    // 检查退出状态
    if !status.success() {
        eprintln!("Python 脚本执行失败，退出状态：{:?}", status);
    }
}