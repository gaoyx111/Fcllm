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
mod server;


use candle_core::{Result, Tensor};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::{Tokenizer, Encoding};
use std::time::Instant;

use args::Args;
use clap::Parser;
use ForCausalLM::Qwen2MoeForCausalLM;



fn main() -> Result<()> {
    let args = Args::parse();

    match (args.server, args.no_ui) {
        // 命令行推理模式（原有功能）
        (false, _) => run_cli(args),
        // 服务器 + 桌面窗口模式（默认 server 行为）
        (true, false) => {
            println!("正在加载模型，请稍候...");
            let mut model = Qwen2MoeForCausalLM::new(args.clone())?;
            model.init_weights()?;
            let tokenizer = Arc::new(
                Tokenizer::from_file(resolve_tokenizer_path(&args))
                    .map_err(|e| candle_core::Error::Msg(format!("tokenizer 加载失败: {e}")))?,
            );
            println!("模型加载完成！正在启动桌面窗口...");
            // launch_desktop_app 返回 !（永不返回），会强制关闭进程
            launch_desktop_app(args, model, tokenizer)
        }
        // 服务器 + 无界面模式（--no-ui）
        (true, true) => run_server_headless(args),
    }
}

// ─────────────────────────────────────────────────────────────
//  CLI 模式（原有命令行推理逻辑）
// ─────────────────────────────────────────────────────────────

fn run_cli(args: Args) -> Result<()> {
    let mut model = Qwen2MoeForCausalLM::new(args.clone())?;
    let device = model.device.clone();
    model.init_weights()?;

    let tokenizer_path = resolve_tokenizer_path(&args);
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("failed to load tokenizer: {e}")))?;

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
    let attention_mask_tensor =
        Tensor::new(attention_mask.as_slice(), &device)?.unsqueeze(0)?;

    // 生成
    let start = Instant::now();
    let (output_ids_tensor, _prefill_time) =
        model.generate(&input_ids_tensor, Some(attention_mask_tensor), Some("decoding"))?;
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
fn launch_desktop_app(
    args: Args,
    model: Qwen2MoeForCausalLM,
    tokenizer: Arc<Tokenizer>,
) -> ! {
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
    let frontend_dir = args.frontend_dir.clone();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("无法创建 tokio 运行时")
            .block_on(async move {
                let state = server::AppState { req_tx };
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

    let tokenizer = Arc::new(
        Tokenizer::from_file(resolve_tokenizer_path(&args))
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer 加载失败: {e}")))?,
    );

    let (req_tx, req_rx) = tokio::sync::mpsc::channel::<server::InferRequest>(4);

    let port = args.port;
    let frontend_dir = args.frontend_dir.clone();
    let req_tx_clone = req_tx.clone();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("无法创建 tokio 运行时")
            .block_on(async move {
                let state = server::AppState { req_tx: req_tx_clone };
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
                axum::serve(listener, app).await.expect("HTTP 服务器异常退出");
            });
    });

    // 主线程运行模型推理（无需 Send）
    server::model_worker_loop(model, tokenizer, req_rx);
    Ok(())
}

// ─────────────────────────────────────────────────────────────
//  辅助函数
// ─────────────────────────────────────────────────────────────

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
