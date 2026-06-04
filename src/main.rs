mod Attention;
mod Cache;
mod DecoderLayer;
mod ForCausalLM;
mod MLP;
mod Model;
mod RmsNorm;
mod RotaryEmbedding;
mod SparseMoeBlock;
mod args;
mod configuration_qwen;
mod expert_ARC_cahce;
mod linear;
mod load;
mod nn_embedding;
mod quantizer;
mod server;
mod utils;

use candle_core::{Result, Tensor};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::{Encoding, Tokenizer};

use ForCausalLM::Qwen2MoeForCausalLM;
use args::Args;
use clap::Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartMode {
    Cli,
    Desktop,
    HeadlessServer,
}

fn start_mode(args: &Args) -> StartMode {
    if args.cli {
        StartMode::Cli
    } else if args.no_ui {
        StartMode::HeadlessServer
    } else {
        StartMode::Desktop
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    match start_mode(&args) {
        // 命令行推理模式（原有功能），现在需显式使用 --cli
        StartMode::Cli => run_cli(args),
        // 服务器 + 桌面窗口模式（默认行为）
        StartMode::Desktop => {
            println!("正在加载模型，请稍候...");
            let mut model = Qwen2MoeForCausalLM::new(args.clone())?;
            model.init_weights()?;
            let tokenizer = Arc::new(load_tokenizer(&args)?);
            println!("模型加载完成！正在启动桌面窗口...");
            // launch_desktop_app 返回 !（永不返回），会强制关闭进程
            launch_desktop_app(args, model, tokenizer)
        }
        // 服务器 + 无界面模式（--no-ui）
        StartMode::HeadlessServer => run_server_headless(args),
    }
}

// ─────────────────────────────────────────────────────────────
//  CLI 模式（原有命令行推理逻辑）
// ─────────────────────────────────────────────────────────────

fn run_cli(args: Args) -> Result<()> {
    let mut model = Qwen2MoeForCausalLM::new(args.clone())?;
    let device = model.device.clone();
    model.init_weights()?;

    let tokenizer = load_tokenizer(&args)?;

    // 编码输入
    let prompt = "Hey, are you conscious? Can you talk to me?";
    let encoding: Encoding = tokenizer.encode(prompt, true).unwrap();
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as i64)
        .collect();

    let input_ids_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)?.unsqueeze(0)?;

    // 生成
    let start = Instant::now();
    let (output_ids_tensor, _prefill_time) = model.generate(
        &input_ids_tensor,
        Some(attention_mask_tensor),
        Some("decoding"),
    )?;
    println!("latency = {:.2?}", start.elapsed());

    let output_flat = output_ids_tensor.squeeze(0)?;
    let output_ids: Vec<u32> = output_flat.to_vec1::<u32>()?;
    let decoded = tokenizer
        .decode(&output_ids, true)
        .map_err(|e| candle_core::Error::Msg(format!("decode error: {e}")))?;
    println!("Output = {}", decoded);

    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  桌面原生窗口模式（默认 server 行为）
//  模型在工作线程运行，tao 事件循环在主线程运行
// ─────────────────────────────────────────────────────────────

/// 启动桌面 App（原生 WebView 窗口）
/// 此函数永不返回（tao event loop 发散）
fn launch_desktop_app(args: Args, model: Qwen2MoeForCausalLM, tokenizer: Arc<Tokenizer>) -> ! {
    use tao::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    };
    use wry::WebViewBuilder;

    let port = args.port;

    // 1. 创建推理请求通道
    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<server::InferRequest>(4);

    // 2. HTTP 服务器就绪信号通道
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<u16>();

    // 3. 模型推理工作线程（model 需要 Send，candle Tensor 实现了 unsafe Send）
    std::thread::spawn(move || {
        server::model_worker_loop(model, tokenizer, req_rx);
    });

    // 4. HTTP 服务器线程
    let frontend_dir = resolve_frontend_dir(args.frontend_dir.clone());
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("无法创建 tokio 运行时")
            .block_on(async move {
                let state = server::AppState::new(req_tx);
                let app = server::create_router(state, frontend_dir);
                let addr = format!("0.0.0.0:{port}");
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .unwrap_or_else(|e| panic!("无法绑定端口 {port}: {e}"));
                // 通知主线程：服务已就绪
                let _ = ready_tx.send(port);
                axum::serve(listener, app)
                    .await
                    .expect("HTTP 服务器异常退出");
            });
    });

    // 5. 等待 HTTP 服务器绑定完成
    let bound_port = ready_rx.recv().expect("HTTP 服务器启动失败");
    let url = format!("http://localhost:{bound_port}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  服务已启动，正在打开桌面窗口...");
    println!("  也可以浏览器直接访问：{url}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 6. 在主线程创建 tao 窗口 + wry WebView
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Fcllm Chat")
        .with_inner_size(LogicalSize::new(1280.0_f64, 860.0_f64))
        .build(&event_loop)
        .expect("无法创建桌面窗口");

    let _webview = WebViewBuilder::new()
        .with_url(&url)
        .build(&window)
        .expect("无法创建 WebView（请确认 Microsoft Edge WebView2 已安装）");

    // 7. 运行事件循环（此处永不返回）
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        if let Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } = event
        {
            // 关闭窗口时退出整个进程（包括模型线程和 HTTP 服务器）
            std::process::exit(0);
        }
    })
}

// ─────────────────────────────────────────────────────────────
//  无界面服务器模式（--no-ui，模型在主线程）
// ─────────────────────────────────────────────────────────────

fn run_server_headless(args: Args) -> Result<()> {
    println!("正在加载模型权重，请稍候...");
    let mut model = Qwen2MoeForCausalLM::new(args.clone())?;
    model.init_weights()?;
    println!("模型加载完成！");

    let tokenizer = Arc::new(load_tokenizer(&args)?);

    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<server::InferRequest>(4);

    let port = args.port;
    let frontend_dir = resolve_frontend_dir(args.frontend_dir.clone());
    let req_tx_clone = req_tx.clone();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("无法创建 tokio 运行时")
            .block_on(async move {
                let state = server::AppState::new(req_tx_clone);
                let app = server::create_router(state, frontend_dir);
                let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
                    .await
                    .unwrap_or_else(|e| panic!("无法绑定端口 {port}: {e}"));
                let local_url = format!("http://localhost:{port}");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("  无界面模式，请用浏览器访问：{local_url}");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                if let Err(e) = open::that(&local_url) {
                    eprintln!("无法自动打开浏览器: {e}");
                }
                axum::serve(listener, app)
                    .await
                    .expect("HTTP 服务器异常退出");
            });
    });

    // 主线程运行模型推理（无需 Send）
    server::model_worker_loop(model, tokenizer, req_rx);
    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  辅助函数
// ─────────────────────────────────────────────────────────────

fn resolve_frontend_dir(frontend_dir: Option<String>) -> Option<String> {
    if let Some(dir) = frontend_dir.filter(|dir| !dir.trim().is_empty()) {
        return Some(dir);
    }

    for candidate in frontend_dir_candidates() {
        if candidate.join("index.html").exists() {
            let dir = candidate.to_string_lossy().to_string();
            println!("使用前端目录：{dir}");
            return Some(dir);
        }
    }

    eprintln!(
        "未找到前端 dist 目录；将只启动 API 服务。可用 --frontend-dir 指定 chatgpt-web\\dist。"
    );
    None
}

fn frontend_dir_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("dist"));
            candidates.push(exe_dir.join("frontend").join("dist"));
        }
    }

    let project_root = args::find_project_root();
    candidates.push(project_root.join("dist"));
    candidates.push(project_root.join("frontend").join("dist"));
    candidates.push(project_root.join("chatgpt-web").join("dist"));

    if let Some(parent) = project_root.parent() {
        candidates.push(parent.join("chatgpt-web").join("dist"));
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join("chatgpt-web").join("dist"));
        }
    }

    candidates.push(PathBuf::from(r"E:\chatgpt-web\dist"));
    candidates
}

/// 推断 tokenizer.json 的路径
/// 优先使用 --tokenizer-path 参数，否则按默认目录结构查找
fn resolve_tokenizer_path(args: &Args) -> PathBuf {
    if let Some(ref p) = args.tokenizer_path {
        return p.clone();
    }

    // 默认路径结构：
    // {model_weights}/Qwen/Qwen1.5-MoE-A2.7B/tokenizer/
    //   models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/<hash>/tokenizer.json
    let base = args
        .path
        .join("Qwen")
        .join("Qwen1.5-MoE-A2.7B")
        .join("tokenizer")
        .join("models--Qwen--Qwen1.5-MoE-A2.7B")
        .join("snapshots")
        .join("1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
        .join("tokenizer.json");

    // 若上面路径不存在，尝试相对路径（兼容旧的启动方式）
    if !base.exists() {
        let fallback = PathBuf::from("model_weights/Qwen/Qwen1.5-MoE-A2.7B/tokenizer")
            .join("models--Qwen--Qwen1.5-MoE-A2.7B")
            .join("snapshots")
            .join("1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
            .join("tokenizer.json");
        if fallback.exists() {
            return fallback;
        }
    }

    base
}

fn load_tokenizer(args: &Args) -> Result<Tokenizer> {
    let tokenizer_path = resolve_tokenizer_path(args);
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer 加载失败: {e}")))?;

    // Qwen 的 chat template 依赖 <|im_start|>/<|im_end|> 被编码为
    // 151644/151645。tokenizers 里这个开关语义很反直觉：
    // false 才会把 special token 字面量抽取为 special token id；
    // true 会把它们继续当普通文本编码。
    tokenizer.set_encode_special_tokens(false);
    let marker_encoding = tokenizer
        .encode("<|im_start|><|im_end|>", true)
        .map_err(|e| candle_core::Error::Msg(format!("tokenizer special token 校验失败: {e}")))?;
    let marker_ids = marker_encoding.get_ids();
    if !marker_ids.contains(&151_644) || !marker_ids.contains(&151_645) {
        return Err(candle_core::Error::Msg(format!(
            "tokenizer special token 校验失败: <|im_start|>/<|im_end|> 编码为 {:?}，不是 151644/151645",
            marker_ids
        )));
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn tokenizer_encodes_qwen_chat_markers_as_special_tokens() {
        let args = Args::parse_from(["Fcllm"]);
        let tokenizer = load_tokenizer(&args).expect("tokenizer should load from model_weights");
        let encoding = tokenizer
            .encode("<|im_start|>assistant\n好的<|im_end|>", true)
            .expect("prompt should encode");
        let ids = encoding.get_ids();

        assert!(ids.contains(&151_644), "<|im_start|> must be token 151644");
        assert!(ids.contains(&151_645), "<|im_end|> must be token 151645");
    }

    #[test]
    fn default_start_mode_is_desktop_window() {
        let args = Args::parse_from(["Fcllm"]);

        assert_eq!(start_mode(&args), StartMode::Desktop);
    }

    #[test]
    fn no_ui_start_mode_is_headless_server() {
        let args = Args::parse_from(["Fcllm", "--no-ui"]);

        assert_eq!(start_mode(&args), StartMode::HeadlessServer);
    }

    #[test]
    fn cli_flag_keeps_single_inference_mode() {
        let args = Args::parse_from(["Fcllm", "--cli"]);

        assert_eq!(start_mode(&args), StartMode::Cli);
    }
}
