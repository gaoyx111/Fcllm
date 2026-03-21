use clap::Parser;
use std::path::PathBuf;


/// 参数结构体
#[derive(Debug, Parser)]
#[command(name = "Qwen2MoeForCausalLM")]
#[command(about = "Run Qwen2MoeForCausalLM inference", long_about = None)]
pub struct Args {
    /// 模型名称
    #[arg(long, default_value = "Qwen/Qwen1.5-MoE-A2.7B")]
    pub model: String,

    /// 权重路径
    #[arg(long, default_value_os_t = default_model_path())]
    pub path: PathBuf,

    /// 是否启用 early stopping
    #[arg(long, default_value_t = true)]
    pub early_stopping: bool,

    /// 最小生成长度
    #[arg(long, default_value_t = 1)]
    pub min_length: usize,

    /// 最大生成长度
    #[arg(long, default_value_t = 256)]
    pub max_length: usize,

    /// 是否将权重固定到内存
    #[arg(long = "pin-weight", default_value_t = true)]
    pub pin_weight: bool,

    /// 内存预算（单位：GB）
    #[arg(long, default_value_t = 0)]
    pub memory_budget: usize,

    /// 使用设备（如 cuda:0）
    #[arg(long, default_value = "cuda:0")]
    pub device: String,

    /// 是否使用 overlap 推理
    #[arg(long, default_value_t = true)]
    pub overlap: bool,
}

// /// 默认路径函数
// fn default_model_path() -> PathBuf {
//     let current_exe = std::env::current_exe().unwrap_or_else(|_| ".".into());
//     let base = current_exe.parent().unwrap_or_else(|| ".".as_ref());
//     base.join("model_weights")
//     // 打印出来是：Path: "E:\\Rust\\Fcllm\\target\\debug\\model_weights"
// }

pub fn find_project_root() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();

    // 向上找包含 Cargo.toml 的目录
    while let Some(parent) = path.parent() {
        if parent.join("Cargo.toml").exists() {
            return parent.to_path_buf();
        }
        path = parent.to_path_buf();
    }

    // fallback
    PathBuf::from(".")
}

pub fn default_model_path() -> PathBuf {
    find_project_root().join("model_weights")
    // 打印出来是：Path: "E:\\Rust\\Fcllm\\model_weights"
}