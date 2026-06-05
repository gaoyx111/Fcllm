use std::collections::{HashMap, VecDeque};
/// HTTP 服务器模块
/// 提供与 chatgpt-web 及 OpenAI 兼容的接口，支持流式输出
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use uuid::Uuid;

use candle_core::Tensor;
use time::{OffsetDateTime, Weekday};
use tokenizers::Tokenizer;

use crate::ForCausalLM::Qwen2MoeForCausalLM;
use crate::args::Args;
use crate::utils::get_gpu_memory_usage;

const MAX_ID_ACC: usize = 8;
const MAX_HISTORY_MESSAGES: usize = 8;
const MAX_HISTORY_CHARS: usize = 1200;
const MAX_HISTORY_MESSAGE_CHARS: usize = 420;
const MAX_GUARDED_GENERATION_ATTEMPTS: usize = 3;
const CHAT_PROCESS_MAX_TOKENS: usize = 1024;
const MAX_REQUEST_MAX_TOKENS: usize = 2048;
const CHAT_PROCESS_HEARTBEAT_SECS: u64 = 15;
const MODEL_REQUEST_QUEUE_CAPACITY: usize = 64;
const AUTO_MIN_WORKER_GPU_MB: usize = 4096;
const MIN_RELIABLE_GPU_WORKER_DELTA_MB: usize = 512;
const AUTO_MIN_WORKER_SYSTEM_MB: usize = 2048;
const MIN_RELIABLE_SYSTEM_WORKER_DELTA_MB: usize = 512;
const SHARED_WEIGHT_WORKER_MIN_FREE_GPU_MB: usize = 384;
const SHARED_WEIGHT_WORKER_MIN_FREE_SYSTEM_MB: usize = 1024;
const MAX_AUTO_MODEL_WORKERS: usize = 8;

const DIRECT_TEXT_STOPS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "\n###",
    "\n问：",
    "\n問：",
    "\nQ:",
    "\nA:",
];

const SYSTEM_LEAK_STOPS: &[&str] = &[
    "你是AI",
    "你是AI，用户是真实的人类",
    "默认产品名",
    "默认用简体中文",
    "始终使用简体中文回答",
    "直接回答当前问题",
    "不要复述、续写对话",
    "不要续写下一轮对话",
    "附加系统指令：",
    "You are ChatGPT, a large language model",
    "Follow the user's instructions carefully",
    "Respond using markdown",
];

const ASCII_DATASET_ARTIFACT_STOPS: &[&str] = &[
    "koa",
    "packet",
    "ontheway",
    "stacksize",
    "stackpackage",
    "ystackpackage",
    "sacksizepackage",
    "hort",
    "endian",
    "_endian",
    "edith",
];
const CJK_DATASET_ARTIFACT_STOPS: &[&str] = &[
    "堆栈包",
    "堆栈包装",
    "堆叠包装",
    "堆叠",
    "堆放",
    "園",
    "圜",
    "币",
    "幣",
    "秧苗",
    "禾苗",
    "稻苗",
];

const ASCII_ROLE_STOPS: &[&str] = &["user", "assistant", "system", "human"];
const CJK_ROLE_STOPS: &[&str] = &[
    "用户", "用戶", "助手", "助理", "系统", "系統", "人类", "人類",
];
const BRACKETED_ROLE_STOPS: &[&str] = &[
    "[user]",
    "[assistant]",
    "[system]",
    "[human]",
    "[用户]",
    "[助手]",
    "[系统]",
];
const ROLE_PREFIX_PUNCTUATION: &str = "。！？.!?；;";
const GENERATED_BOUNDARY_ARTIFACTS: &[&str] = &["宇宙"];
const MATH_PARSE_ERROR_STOPS: &[&str] = &[
    "ParseError: KaTeX parse error",
    "KaTeX parse error",
    "Got function '$' with no arguments",
];
const SYMBOL_ARTIFACT_STOPS: &[&str] = &["$k", "$K"];

const WRONG_STARDEW_TITLE_STOPS: &[&str] = &[
    "星露山谷新生传说",
    "星露谷新生传说",
    "星露山谷物语",
    "星露山谷",
    "星露谷新生",
    "星露草谷物语",
    "星露城谷物语",
    "星落谷物",
    "Stardust Valley",
    "Stardust City Valley",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextStop {
    pos: usize,
    marker: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalAnswerGuard {
    Hold,
    StopWrongDate,
}

#[derive(Debug, Clone, Copy)]
struct RuntimeDateInfo {
    year: i32,
    month: u8,
    day: u8,
    weekday: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextDebugMessage {
    role: String,
    char_count: usize,
    reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextDebugFilteredMessage {
    role: String,
    char_count: usize,
    reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContextDebugEvent {
    conversation_id: String,
    user_chars: usize,
    history_messages: usize,
    cleaned_history_messages: usize,
    prompt_messages: usize,
    prompt_chars: usize,
    prior_context_triggered: bool,
    answer_without_prior_history: bool,
    discarded_polluted_history: bool,
    selected: Vec<ContextDebugMessage>,
    filtered: Vec<ContextDebugFilteredMessage>,
}

impl ContextDebugEvent {
    fn new(conversation_id: &str, user_prompt: &str, history: &[ChatMessage]) -> Self {
        Self {
            conversation_id: conversation_id.to_string(),
            user_chars: user_prompt.chars().count(),
            history_messages: history.len(),
            cleaned_history_messages: 0,
            prompt_messages: 0,
            prompt_chars: 0,
            prior_context_triggered: false,
            answer_without_prior_history: false,
            discarded_polluted_history: false,
            selected: Vec::new(),
            filtered: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  OpenAI API 请求 / 响应类型
// ═══════════════════════════════════════════════════════════

/// OpenAI 格式的聊天消息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// POST /v1/chat/completions 请求体
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// 是否开启流式输出（SSE）
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

/// 非流式响应
#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<NonStreamChoice>,
}

#[derive(Debug, Serialize)]
struct NonStreamChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

/// SSE 流式 chunk
#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ═══════════════════════════════════════════════════════════
//  chatgpt-web 直连格式（跳过 Node.js 中间层时使用）
// ═══════════════════════════════════════════════════════════

/// POST /chat-process 请求体（chatgpt-web 原生格式）
#[derive(Debug, Deserialize)]
pub struct ChatProcessRequest {
    pub prompt: String,
    pub options: Option<serde_json::Value>,
    #[serde(rename = "systemMessage")]
    pub system_message: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

/// chatgpt-web SSE 推送格式
#[derive(Debug, Serialize)]
struct ChatProcessChunk {
    role: String,
    id: String,
    /// 当前已累积的全部文本
    text: String,
    /// 本次新增的 token 文本
    delta: String,
    #[serde(rename = "parentMessageId")]
    parent_message_id: String,
    #[serde(rename = "conversationId")]
    conversation_id: String,
    detail: ChatProcessDetail,
}

#[derive(Debug, Serialize)]
struct ChatProcessDetail {
    choices: Vec<ChatProcessDetailChoice>,
}

#[derive(Debug, Serialize)]
struct ChatProcessDetailChoice {
    finish_reason: Option<String>,
}

// ═══════════════════════════════════════════════════════════
//  模型工作线程通信通道
// ═══════════════════════════════════════════════════════════

/// 每次推理请求的结构体，由 HTTP handler 发往模型工作线程
pub struct InferRequest {
    /// 格式化好的 prompt 字符串（已应用 Qwen chat template）
    pub prompt: String,
    /// 当前这一轮的真实用户输入，用于拦截 base 模型续写/复述用户输入。
    pub user_prompt: Option<String>,
    /// 最大生成 token 数
    pub max_tokens: usize,
    /// 生成结束原因。worker 会在发送结束信号前写入 stop/length。
    pub finish_reason: Arc<Mutex<String>>,
    /// 每生成一个 token 就往这里发送解码后的文本；None 表示生成结束
    pub token_tx: mpsc::Sender<Option<String>>,
}

/// 请求发送端类型（在 axum State 中共享）
pub type ModelReqSender = mpsc::Sender<InferRequest>;
type ConversationStore = Arc<Mutex<HashMap<String, Vec<ChatMessage>>>>;

struct ModelWorkerHandle {
    id: usize,
    req_tx: mpsc::Sender<InferRequest>,
    busy: bool,
}

enum WorkerPoolEvent {
    Idle(usize),
    Exited(usize),
}

struct AutoWorkerPoolConfig {
    args: Args,
    tokenizer: Arc<Tokenizer>,
    prototype_model: Arc<Qwen2MoeForCausalLM>,
    estimated_worker_gpu_mb: Option<usize>,
    estimated_worker_system_mb: Option<usize>,
}

/// axum 共享状态
#[derive(Clone)]
pub struct AppState {
    pub req_tx: ModelReqSender,
    conversations: ConversationStore,
}

impl AppState {
    pub fn new(req_tx: ModelReqSender) -> Self {
        Self {
            req_tx,
            conversations: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

pub fn model_request_queue_capacity() -> usize {
    MODEL_REQUEST_QUEUE_CAPACITY
}

pub fn gpu_memory_usage_for_device_spec(device: &str) -> Option<(usize, usize)> {
    let device_id = cuda_device_index(device)?;
    get_gpu_memory_usage(device_id).ok()
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
}

#[cfg(target_os = "windows")]
pub fn system_memory_usage_mb() -> Option<(usize, usize)> {
    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };

    let ok = unsafe { GlobalMemoryStatusEx(&mut status as *mut MemoryStatusEx) };
    if ok == 0 || status.ull_total_phys == 0 {
        return None;
    }

    let total_mb = bytes_to_mb(status.ull_total_phys);
    let avail_mb = bytes_to_mb(status.ull_avail_phys);
    Some((total_mb.saturating_sub(avail_mb), total_mb))
}

#[cfg(not(target_os = "windows"))]
pub fn system_memory_usage_mb() -> Option<(usize, usize)> {
    None
}

fn bytes_to_mb(bytes: u64) -> usize {
    (bytes / 1024 / 1024) as usize
}

pub fn estimate_worker_gpu_memory_mb(
    before: Option<(usize, usize)>,
    after: Option<(usize, usize)>,
) -> Option<usize> {
    let ((used_before, _), (used_after, _)) = (before?, after?);
    let delta = used_after.checked_sub(used_before)?;
    if delta < MIN_RELIABLE_GPU_WORKER_DELTA_MB {
        return None;
    }
    Some(delta.max(AUTO_MIN_WORKER_GPU_MB))
}

pub fn estimate_worker_system_memory_mb(
    before: Option<(usize, usize)>,
    after: Option<(usize, usize)>,
) -> Option<usize> {
    let ((used_before, _), (used_after, _)) = (before?, after?);
    let delta = used_after.checked_sub(used_before)?;
    if delta < MIN_RELIABLE_SYSTEM_WORKER_DELTA_MB {
        return None;
    }
    Some(delta.max(AUTO_MIN_WORKER_SYSTEM_MB))
}

pub fn start_auto_model_worker_pool(
    initial_model: Qwen2MoeForCausalLM,
    args: Args,
    tokenizer: Arc<Tokenizer>,
    estimated_worker_gpu_mb: Option<usize>,
    estimated_worker_system_mb: Option<usize>,
) -> ModelReqSender {
    let (pool_tx, pool_rx) = mpsc::channel::<InferRequest>(MODEL_REQUEST_QUEUE_CAPACITY);
    let prototype_model = Arc::new(initial_model.clone());
    let config = AutoWorkerPoolConfig {
        args,
        tokenizer,
        prototype_model,
        estimated_worker_gpu_mb,
        estimated_worker_system_mb,
    };

    std::thread::spawn(move || {
        auto_model_worker_pool_loop(initial_model, pool_rx, config);
    });

    pool_tx
}

fn auto_model_worker_pool_loop(
    initial_model: Qwen2MoeForCausalLM,
    mut pool_rx: mpsc::Receiver<InferRequest>,
    config: AutoWorkerPoolConfig,
) {
    let (done_tx, done_rx) = std::sync::mpsc::channel::<WorkerPoolEvent>();
    let mut workers = Vec::new();
    let mut pending = VecDeque::new();
    let mut next_worker_id = 0usize;
    let mut auto_spawn_disabled = false;

    workers.push(spawn_model_worker(
        next_worker_id,
        initial_model,
        config.tokenizer.clone(),
        done_tx.clone(),
    ));
    next_worker_id += 1;

    eprintln!(
        "[worker-pool] 自动模型 worker 池启动：初始 worker=1，队列容量={MODEL_REQUEST_QUEUE_CAPACITY}"
    );
    eprintln!(
        "[worker-pool] worker 资源估算：gpu_mb={:?} system_mb={:?} hard_cap={}",
        config.estimated_worker_gpu_mb,
        config.estimated_worker_system_mb,
        auto_worker_hard_cap(&config)
    );

    loop {
        mark_completed_workers(&mut workers, &done_rx);
        dispatch_pending_requests(
            &mut pending,
            &mut workers,
            &config,
            &done_tx,
            &mut next_worker_id,
            &mut auto_spawn_disabled,
        );

        if pending.is_empty() {
            match pool_rx.blocking_recv() {
                Some(req) => pending.push_back(req),
                None => break,
            }
            continue;
        }

        match done_rx.recv() {
            Ok(event) => handle_worker_pool_event(&mut workers, event),
            Err(_) => break,
        }
    }

    eprintln!("[worker-pool] 请求通道已关闭，worker 池退出");
}

fn mark_completed_workers(
    workers: &mut Vec<ModelWorkerHandle>,
    done_rx: &std::sync::mpsc::Receiver<WorkerPoolEvent>,
) {
    while let Ok(event) = done_rx.try_recv() {
        handle_worker_pool_event(workers, event);
    }
}

fn handle_worker_pool_event(workers: &mut Vec<ModelWorkerHandle>, event: WorkerPoolEvent) {
    match event {
        WorkerPoolEvent::Idle(worker_id) => {
            if let Some(worker) = workers.iter_mut().find(|worker| worker.id == worker_id) {
                worker.busy = false;
            }
        }
        WorkerPoolEvent::Exited(worker_id) => {
            if let Some(pos) = workers.iter().position(|worker| worker.id == worker_id) {
                workers.remove(pos);
                eprintln!("[worker-pool] worker {worker_id} 已退出，已从调度池移除");
            }
        }
    }
}

fn dispatch_pending_requests(
    pending: &mut VecDeque<InferRequest>,
    workers: &mut Vec<ModelWorkerHandle>,
    config: &AutoWorkerPoolConfig,
    done_tx: &std::sync::mpsc::Sender<WorkerPoolEvent>,
    next_worker_id: &mut usize,
    auto_spawn_disabled: &mut bool,
) {
    while let Some(req) = pending.pop_front() {
        if req.token_tx.is_closed() {
            eprintln!("[worker-pool] 客户端已断开，丢弃排队请求");
            continue;
        }

        if let Some(worker_pos) = workers.iter().position(|worker| !worker.busy) {
            match dispatch_to_worker(&mut workers[worker_pos], req) {
                Ok(()) => continue,
                Err(err) => {
                    let failed_worker = workers.remove(worker_pos);
                    eprintln!(
                        "[worker-pool] worker {} 通道已关闭，保留请求并移除该 worker",
                        failed_worker.id
                    );
                    pending.push_front(err.0);
                }
            }
            if workers.is_empty() {
                continue;
            }
            continue;
        }

        if workers.is_empty() {
            match spawn_replacement_worker(config, done_tx.clone(), *next_worker_id) {
                Some(mut worker) => {
                    *next_worker_id += 1;
                    let _ = dispatch_to_worker(&mut worker, req);
                    workers.push(worker);
                    continue;
                }
                None => {
                    *auto_spawn_disabled = true;
                    finish_unserved_request(req);
                    continue;
                }
            }
        }

        if !*auto_spawn_disabled && can_spawn_additional_worker(config, workers.len()) {
            match spawn_additional_worker(config, done_tx.clone(), *next_worker_id) {
                Some(mut worker) => {
                    *next_worker_id += 1;
                    let _ = dispatch_to_worker(&mut worker, req);
                    workers.push(worker);
                    continue;
                }
                None => {
                    *auto_spawn_disabled = true;
                }
            }
        }

        pending.push_front(req);
        break;
    }
}

fn dispatch_to_worker(
    worker: &mut ModelWorkerHandle,
    req: InferRequest,
) -> Result<(), mpsc::error::SendError<InferRequest>> {
    worker.busy = true;
    worker.req_tx.blocking_send(req).map_err(|err| {
        worker.busy = false;
        eprintln!("[worker-pool] worker {} 已退出，无法分发请求", worker.id);
        err
    })
}

fn spawn_model_worker(
    worker_id: usize,
    model: Qwen2MoeForCausalLM,
    tokenizer: Arc<Tokenizer>,
    done_tx: std::sync::mpsc::Sender<WorkerPoolEvent>,
) -> ModelWorkerHandle {
    let (req_tx, req_rx) = mpsc::channel::<InferRequest>(1);
    let exit_tx = done_tx.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model_worker_loop_with_notify(model, tokenizer, req_rx, worker_id, done_tx);
        }));
        if result.is_err() {
            eprintln!("[worker-pool] worker {worker_id} 发生 panic，准备从调度池移除");
        }
        let _ = exit_tx.send(WorkerPoolEvent::Exited(worker_id));
    });

    ModelWorkerHandle {
        id: worker_id,
        req_tx,
        busy: false,
    }
}

fn spawn_additional_worker(
    config: &AutoWorkerPoolConfig,
    done_tx: std::sync::mpsc::Sender<WorkerPoolEvent>,
    worker_id: usize,
) -> Option<ModelWorkerHandle> {
    eprintln!(
        "[worker-pool] 资源允许，正在克隆共享权重的第 {} 个模型 worker",
        worker_id + 1
    );
    load_model_worker(config, done_tx, worker_id)
}

fn spawn_replacement_worker(
    config: &AutoWorkerPoolConfig,
    done_tx: std::sync::mpsc::Sender<WorkerPoolEvent>,
    worker_id: usize,
) -> Option<ModelWorkerHandle> {
    eprintln!(
        "[worker-pool] 没有可用模型 worker，尝试从共享权重 prototype 重建 worker {worker_id}"
    );
    load_model_worker(config, done_tx, worker_id)
}

fn load_model_worker(
    config: &AutoWorkerPoolConfig,
    done_tx: std::sync::mpsc::Sender<WorkerPoolEvent>,
    worker_id: usize,
) -> Option<ModelWorkerHandle> {
    let model_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        candle_core::Result::Ok((*config.prototype_model).clone())
    }));

    let model = match model_result {
        Ok(Ok(model)) => model,
        Ok(Err(err)) => {
            eprintln!("[worker-pool] 加载模型 worker 失败: {err}");
            return None;
        }
        Err(_) => {
            eprintln!("[worker-pool] 加载模型 worker 时发生 panic");
            return None;
        }
    };

    Some(spawn_model_worker(
        worker_id,
        model,
        config.tokenizer.clone(),
        done_tx,
    ))
}

fn finish_unserved_request(req: InferRequest) {
    set_finish_reason(&req.finish_reason, "stop");
    let _ = req.token_tx.blocking_send(None);
}

fn can_spawn_additional_worker(config: &AutoWorkerPoolConfig, current_workers: usize) -> bool {
    if current_workers >= auto_worker_hard_cap(config) {
        return false;
    }

    match (
        config.estimated_worker_gpu_mb,
        config.estimated_worker_system_mb,
    ) {
        (Some(_), Some(_)) => {
            let gpu_allowed =
                can_spawn_shared_weight_worker_with_gpu_memory(config, current_workers);
            let system_allowed =
                can_spawn_shared_weight_worker_with_system_memory(config, current_workers);
            let allowed = gpu_allowed && system_allowed;
            eprintln!(
                "[worker-pool] 共享权重并发综合判断：需要同时满足 GPU 与系统内存运行余量，spawn={allowed}"
            );
            allowed
        }
        (Some(_), None) => can_spawn_shared_weight_worker_with_gpu_memory(config, current_workers),
        (None, Some(_)) => {
            can_spawn_shared_weight_worker_with_system_memory(config, current_workers)
        }
        (None, None) => {
            eprintln!("[worker-pool] 自动并发检查：未获得可靠 worker 资源估算，保持排队");
            false
        }
    }
}

fn can_spawn_shared_weight_worker_with_gpu_memory(
    config: &AutoWorkerPoolConfig,
    current_workers: usize,
) -> bool {
    if config.estimated_worker_gpu_mb.is_none() {
        eprintln!("[worker-pool] GPU 并发检查：未获得可靠 worker 显存估算");
        return false;
    }

    let Some((used_mb, total_mb)) = gpu_memory_usage_for_device_spec(&config.args.device) else {
        eprintln!("[worker-pool] GPU 并发检查：无法读取显存，转用系统内存判断");
        return false;
    };
    let free_mb = total_mb.saturating_sub(used_mb);
    let allowed = free_mb > SHARED_WEIGHT_WORKER_MIN_FREE_GPU_MB;

    eprintln!(
        "[worker-pool] 共享权重 GPU 并发检查：workers={} free_gpu_mb={} min_free_gpu_mb={} spawn={}",
        current_workers, free_mb, SHARED_WEIGHT_WORKER_MIN_FREE_GPU_MB, allowed
    );

    allowed
}

fn can_spawn_shared_weight_worker_with_system_memory(
    config: &AutoWorkerPoolConfig,
    current_workers: usize,
) -> bool {
    if config.estimated_worker_system_mb.is_none() {
        eprintln!("[worker-pool] 系统内存并发检查：未获得可靠 worker 内存估算，保持排队");
        return false;
    }

    let Some((used_mb, total_mb)) = system_memory_usage_mb() else {
        eprintln!("[worker-pool] 系统内存并发检查：无法读取系统内存，保持排队");
        return false;
    };
    let free_mb = total_mb.saturating_sub(used_mb);
    let allowed = free_mb > SHARED_WEIGHT_WORKER_MIN_FREE_SYSTEM_MB;

    eprintln!(
        "[worker-pool] 共享权重系统内存并发检查：workers={} free_ram_mb={} min_free_ram_mb={} spawn={}",
        current_workers, free_mb, SHARED_WEIGHT_WORKER_MIN_FREE_SYSTEM_MB, allowed
    );

    allowed
}

fn auto_worker_hard_cap(config: &AutoWorkerPoolConfig) -> usize {
    let cpu_cap = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, MAX_AUTO_MODEL_WORKERS);

    let device_cap = if config.estimated_worker_gpu_mb.is_some() {
        match gpu_memory_usage_for_device_spec(&config.args.device) {
            Some((_, total_mb)) if total_mb <= 12 * 1024 => 2,
            Some((_, total_mb)) if total_mb <= 24 * 1024 => 4,
            Some(_) => MAX_AUTO_MODEL_WORKERS,
            None => 2,
        }
    } else {
        cpu_cap
    };

    cpu_cap.min(device_cap).clamp(1, MAX_AUTO_MODEL_WORKERS)
}

fn cuda_device_index(device: &str) -> Option<u32> {
    let normalized = device.trim().to_ascii_lowercase();
    if normalized == "cuda" {
        return Some(0);
    }
    normalized
        .strip_prefix("cuda:")
        .and_then(|index| index.parse::<u32>().ok())
}

// ═══════════════════════════════════════════════════════════
//  路由构建
// ═══════════════════════════════════════════════════════════

/// 构建 axum Router
/// - frontend_dir: 若指定则同时托管前端静态文件（chatgpt-web dist/）
pub fn create_router(state: AppState, frontend_dir: Option<String>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let router = Router::new()
        // chatgpt-web 必需的握手接口（页面加载时立即调用）
        .route("/session", post(session_handler))
        .route("/config", post(config_handler))
        .route("/verify", post(verify_handler))
        // 核心聊天接口
        .route("/chat-process", post(chatgpt_web_handler))
        // 兼容 chatgpt-web 默认 /api 前缀；Vite 开发代理会把 /api/chat-process
        // 改写为 /chat-process，直接托管 dist 时后端需要自己接住这个前缀。
        .route("/api/session", post(session_handler))
        .route("/api/config", post(config_handler))
        .route("/api/verify", post(verify_handler))
        .route("/api/chat-process", post(chatgpt_web_handler))
        // OpenAI 兼容接口
        .route("/v1/chat/completions", post(openai_handler))
        .route("/v1/models", get(models_handler))
        .route("/api/v1/chat/completions", post(openai_handler))
        .route("/api/v1/models", get(models_handler))
        .with_state(state)
        .layer(cors);

    match frontend_dir {
        Some(dir) => router.fallback_service(ServeDir::new(dir)),
        None => router,
    }
}

// ═══════════════════════════════════════════════════════════
//  HTTP 处理器
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
//  chatgpt-web 握手接口（页面加载时调用，必须存在）
// ═══════════════════════════════════════════════════════════

/// POST /session —— 返回会话状态（chatgpt-web 启动时第一个调用的接口）
async fn session_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "Success",
        "message": "",
        "data": {
            "auth": false,
            "model": "ChatGPTAPI"
        }
    }))
}

/// POST /config —— 返回模型配置信息
async fn config_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "Success",
        "message": "",
        "data": {
            "apiModel": "ChatGPTAPI",
            "reverseProxy": "",
            "timeoutMs": 60000,
            "socksProxy": "",
            "httpsProxy": "",
            "usage": ""
        }
    }))
}

/// POST /verify —— 鉴权验证（我们不设密码，直接返回成功）
async fn verify_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "Success",
        "message": "Verify successfully",
        "data": null
    }))
}

/// GET /v1/models —— 返回模型列表
async fn models_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "qwen1.5-moe-a2.7b",
            "object": "model",
            "owned_by": "local",
            "created": unix_ts()
        }]
    }))
}

/// POST /v1/chat/completions —— OpenAI 兼容接口（支持流式与非流式）
async fn openai_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let user_prompt = last_user_prompt(&req.messages);
    let prompt = messages_to_qwen_prompt(&req.messages);
    let max_tokens = normalized_max_tokens(req.max_tokens, CHAT_PROCESS_MAX_TOKENS);

    if req.stream {
        sse_response(state, prompt, user_prompt, max_tokens, false).await
    } else {
        blocking_response(state, prompt, user_prompt, max_tokens).await
    }
}

/// POST /chat-process —— chatgpt-web 原生直连接口
///
/// chatgpt-web 前端用 Axios onDownloadProgress 读取流式响应，
/// 实际上是 NDJSON（换行分隔的 JSON），而不是标准 SSE。
/// 每行是一个完整 JSON 对象，Content-Type: application/octet-stream。
async fn chatgpt_web_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatProcessRequest>,
) -> Response {
    let system = req.system_message.as_deref().unwrap_or("");
    let conversation_id = chat_process_conversation_id(req.options.as_ref())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let requested_continuation = chat_process_parent_message_id(req.options.as_ref()).is_some()
        && req.prompt.trim().is_empty();
    let prior_history = load_conversation_history(&state.conversations, &conversation_id);
    let continuation_prompt = if requested_continuation {
        build_chat_process_continuation_prompt(&prior_history, system)
    } else {
        None
    };

    if requested_continuation && continuation_prompt.is_none() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            "Missing conversation context for continuation",
        )
            .into_response();
    }

    let is_continuation = continuation_prompt.is_some();
    let prompt = if let Some(prompt) = continuation_prompt {
        let mut debug = ContextDebugEvent::new(&conversation_id, &req.prompt, &prior_history);
        debug.cleaned_history_messages = clean_history_messages_for_prompt(&prior_history).len();
        debug.prompt_chars = prompt.chars().count();
        debug.prompt_messages = debug.cleaned_history_messages;
        log_context_debug(&debug);
        prompt
    } else {
        let (prompt, debug) = build_chat_process_prompt_with_debug(
            &prior_history,
            &req.prompt,
            system,
            &conversation_id,
        );
        log_context_debug(&debug);
        prompt
    };

    ndjson_stream_response(
        state,
        prompt,
        normalized_max_tokens(req.max_tokens, CHAT_PROCESS_MAX_TOKENS),
        conversation_id,
        req.prompt,
        prior_history,
        is_continuation,
    )
    .await
}

/// 构建 NDJSON 流式响应（chatgpt-web 格式）
/// 每个 token 写一行 JSON，前端通过 Axios onDownloadProgress 增量读取
async fn ndjson_stream_response(
    state: AppState,
    prompt: String,
    max_tokens: usize,
    conversation_id: String,
    user_prompt: String,
    prior_history: Vec<ChatMessage>,
    is_continuation: bool,
) -> Response {
    let (token_tx, token_rx) = mpsc::channel::<Option<String>>(256);
    let conversations = state.conversations.clone();
    let finish_reason = Arc::new(std::sync::Mutex::new("stop".to_string()));

    if state
        .req_tx
        .send(InferRequest {
            prompt,
            user_prompt: Some(user_prompt.clone()),
            max_tokens,
            finish_reason: finish_reason.clone(),
            token_tx,
        })
        .await
        .is_err()
    {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Model worker is not running",
        )
            .into_response();
    }

    let id = Uuid::new_v4().to_string();
    let (line_tx, line_rx) = mpsc::channel::<String>(256);
    let mut token_rx = token_rx;
    let id_c = id.clone();
    let conversation_id_c = conversation_id.clone();
    let conversations_c = conversations.clone();
    let prior_history_c = prior_history.clone();
    let user_prompt_c = user_prompt.clone();
    let finish_reason_c = finish_reason.clone();

    tokio::spawn(async move {
        let mut accumulated = String::new();
        let mut heartbeat = tokio::time::interval(Duration::from_secs(CHAT_PROCESS_HEARTBEAT_SECS));
        heartbeat.tick().await;

        if line_tx
            .send(chat_process_chunk_line(
                &id_c,
                &conversation_id_c,
                accumulated.clone(),
                String::new(),
                None,
            ))
            .await
            .is_err()
        {
            return;
        }

        loop {
            tokio::select! {
                token_opt = token_rx.recv() => {
                    match token_opt {
                        Some(Some(text)) => {
                            accumulated.push_str(&text);
                            if line_tx
                                .send(chat_process_chunk_line(
                                    &id_c,
                                    &conversation_id_c,
                                    accumulated.clone(),
                                    text,
                                    None,
                                ))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Some(None) | None => {
                            save_conversation_turn(
                                &conversations_c,
                                &conversation_id_c,
                                &prior_history_c,
                                &user_prompt_c,
                                &accumulated,
                                is_continuation,
                            );

                            let finish_reason = current_finish_reason(&finish_reason_c);
                            let _ = line_tx
                                .send(chat_process_chunk_line(
                                    &id_c,
                                    &conversation_id_c,
                                    accumulated.clone(),
                                    String::new(),
                                    Some(finish_reason),
                                ))
                                .await;
                            break;
                        }
                    }
                }
                _ = heartbeat.tick() => {
                    if line_tx
                        .send(chat_process_chunk_line(
                            &id_c,
                            &conversation_id_c,
                            accumulated.clone(),
                            String::new(),
                            None,
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    let ndjson_stream = ReceiverStream::new(line_rx).map(|line| Ok::<String, Infallible>(line));

    Response::builder()
        .header("content-type", "application/octet-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no") // 禁用 nginx 缓冲（如有反代）
        .body(Body::from_stream(ndjson_stream))
        .unwrap_or_else(|_| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "stream error",
            )
                .into_response()
        })
}

// ═══════════════════════════════════════════════════════════
//  SSE 流式响应
// ═══════════════════════════════════════════════════════════

/// 构建 SSE 流式响应
/// - chatgpt_web_fmt: true → 使用 chatgpt-web 格式；false → 使用 OpenAI 格式
async fn sse_response(
    state: AppState,
    prompt: String,
    user_prompt: Option<String>,
    max_tokens: usize,
    chatgpt_web_fmt: bool,
) -> Response {
    let (token_tx, token_rx) = mpsc::channel::<Option<String>>(256);
    let finish_reason = Arc::new(std::sync::Mutex::new("stop".to_string()));

    // 把请求发给模型工作线程
    if state
        .req_tx
        .send(InferRequest {
            prompt,
            user_prompt,
            max_tokens,
            finish_reason: finish_reason.clone(),
            token_tx,
        })
        .await
        .is_err()
    {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Model worker is not running",
        )
            .into_response();
    }

    let id = Uuid::new_v4().to_string();
    // 用于 chatgpt-web 格式：累积所有已生成文本
    let accumulated = Arc::new(std::sync::Mutex::new(String::new()));

    let id_c = id.clone();
    let acc_c = accumulated.clone();
    let finish_reason_c = finish_reason.clone();

    let stream = ReceiverStream::new(token_rx).map(move |token_opt| {
        let data = match token_opt {
            Some(ref text) => {
                if chatgpt_web_fmt {
                    // chatgpt-web 格式：每次推送累积文本 + 本次 delta
                    let mut acc = acc_c.lock().unwrap();
                    acc.push_str(text);
                    let chunk = chat_process_chunk(&id_c, &id_c, acc.clone(), text.clone(), None);
                    serde_json::to_string(&chunk).unwrap_or_default()
                } else {
                    // OpenAI SSE chunk 格式
                    let chunk = ChatCompletionChunk {
                        id: id_c.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: unix_ts(),
                        model: "qwen1.5-moe-a2.7b".to_string(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text.clone()),
                            },
                            finish_reason: None,
                        }],
                    };
                    serde_json::to_string(&chunk).unwrap_or_default()
                }
            }
            // None 表示生成结束
            None => {
                if chatgpt_web_fmt {
                    let acc = acc_c.lock().unwrap();
                    let chunk = chat_process_chunk(
                        &id_c,
                        &id_c,
                        acc.clone(),
                        String::new(),
                        Some(current_finish_reason(&finish_reason_c)),
                    );
                    serde_json::to_string(&chunk).unwrap_or_default()
                } else {
                    let chunk = ChatCompletionChunk {
                        id: id_c.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: unix_ts(),
                        model: "qwen1.5-moe-a2.7b".to_string(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(current_finish_reason(&finish_reason_c)),
                        }],
                    };
                    serde_json::to_string(&chunk).unwrap_or_default()
                }
            }
        };

        Ok::<Event, Infallible>(Event::default().data(data))
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ═══════════════════════════════════════════════════════════
//  非流式（一次性返回全文）响应
// ═══════════════════════════════════════════════════════════

async fn blocking_response(
    state: AppState,
    prompt: String,
    user_prompt: Option<String>,
    max_tokens: usize,
) -> Response {
    let (token_tx, mut token_rx) = mpsc::channel::<Option<String>>(256);
    let finish_reason = Arc::new(std::sync::Mutex::new("stop".to_string()));

    if state
        .req_tx
        .send(InferRequest {
            prompt,
            user_prompt,
            max_tokens,
            finish_reason: finish_reason.clone(),
            token_tx,
        })
        .await
        .is_err()
    {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Model worker is not running",
        )
            .into_response();
    }

    // 等待所有 token 生成完毕
    let mut full_text = String::new();
    while let Some(token_opt) = token_rx.recv().await {
        match token_opt {
            Some(text) => full_text.push_str(&text),
            None => break,
        }
    }

    let resp = ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: unix_ts(),
        model: "qwen1.5-moe-a2.7b".to_string(),
        choices: vec![NonStreamChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: full_text,
            },
            finish_reason: current_finish_reason(&finish_reason),
        }],
    };

    Json(resp).into_response()
}

fn default_system_prompt() -> &'static str {
    "你是Fcllm。默认用简体中文，直接回答当前问题，遵守用户限制；不要续写对话或输出角色标签。"
}

fn effective_system_prompt(system: &str, alias: Option<&str>) -> String {
    let mut prompt = default_system_prompt().to_string();
    prompt.push('\n');
    prompt.push_str(&runtime_date_context());
    if let Some(alias) = alias.filter(|name| !name.trim().is_empty()) {
        prompt.push_str(&format!(
            "\n当前用户给你的称呼是「{}」。当用户询问你的名字或称呼时，必须回答这个称呼。",
            sanitize_message_content(alias.trim())
        ));
    }

    if should_include_custom_system_message(system) {
        prompt.push_str(&format!(
            "\n\n附加系统指令：{}",
            sanitize_message_content(system.trim())
        ));
    }

    prompt
}

fn runtime_date_context() -> String {
    let date = runtime_date_info();
    format!(
        "当前系统日期是{:04}年{}月{}日，星期{}。如果用户询问今天、现在、明天或昨天的日期，只能依据这个系统日期推算；如果用户询问你的训练语料截止日期，你不能编造具体日期，应说明本地模型无法可靠得知。",
        date.year, date.month, date.day, date.weekday
    )
}

fn runtime_current_date_fact() -> String {
    let date = runtime_date_info();
    format!(
        "现在是{:04}年{}月{}日，星期{}。",
        date.year, date.month, date.day, date.weekday
    )
}

fn runtime_date_fields() -> String {
    let date = runtime_date_info();
    format!(
        "DATE_YEAR={}; DATE_MONTH={}; DATE_DAY={}; DATE_WEEKDAY=星期{}",
        date.year, date.month, date.day, date.weekday
    )
}

fn runtime_date_info() -> RuntimeDateInfo {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    RuntimeDateInfo {
        year: now.year(),
        month: now.month() as u8,
        day: now.day(),
        weekday: weekday_zh(now.weekday()),
    }
}

fn weekday_zh(day: Weekday) -> &'static str {
    match day {
        Weekday::Monday => "一",
        Weekday::Tuesday => "二",
        Weekday::Wednesday => "三",
        Weekday::Thursday => "四",
        Weekday::Friday => "五",
        Weekday::Saturday => "六",
        Weekday::Sunday => "日",
    }
}

fn chat_process_chunk(
    id: &str,
    conversation_id: &str,
    text: String,
    delta: String,
    finish_reason: Option<String>,
) -> ChatProcessChunk {
    ChatProcessChunk {
        role: "assistant".to_string(),
        id: id.to_string(),
        text,
        delta,
        parent_message_id: id.to_string(),
        conversation_id: conversation_id.to_string(),
        detail: ChatProcessDetail {
            choices: vec![ChatProcessDetailChoice { finish_reason }],
        },
    }
}

fn chat_process_chunk_line(
    id: &str,
    conversation_id: &str,
    text: String,
    delta: String,
    finish_reason: Option<String>,
) -> String {
    let mut line = serde_json::to_string(&chat_process_chunk(
        id,
        conversation_id,
        text,
        delta,
        finish_reason,
    ))
    .unwrap_or_default();
    line.push('\n');
    line
}

fn current_finish_reason(finish_reason: &Arc<Mutex<String>>) -> String {
    finish_reason
        .lock()
        .map(|reason| reason.clone())
        .unwrap_or_else(|_| "stop".to_string())
}

fn set_finish_reason(finish_reason: &Arc<Mutex<String>>, value: &str) {
    if let Ok(mut reason) = finish_reason.lock() {
        *reason = value.to_string();
    }
}

fn normalized_max_tokens(requested: Option<usize>, default: usize) -> usize {
    requested
        .unwrap_or(default)
        .clamp(1, MAX_REQUEST_MAX_TOKENS)
}

fn sanitize_message_content(content: &str) -> String {
    content
        .replace("<|im_start|>", "< |im_start| >")
        .replace("<|im_end|>", "< |im_end| >")
        .replace("<|endoftext|>", "< |endoftext| >")
}

fn earliest_stop(current: Option<TextStop>, next: TextStop) -> Option<TextStop> {
    match current {
        Some(prev) if prev.pos <= next.pos => Some(prev),
        _ => Some(next),
    }
}

fn find_text_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for &marker in DIRECT_TEXT_STOPS {
        if let Some(pos) = text.find(marker) {
            best = earliest_stop(best, TextStop { pos, marker });
        }
    }

    for &marker in SYSTEM_LEAK_STOPS {
        if let Some(pos) = text.find(marker) {
            best = earliest_stop(best, TextStop { pos, marker });
        }
    }

    if let Some(stop) = find_wrong_stardew_title_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_dataset_artifact_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_metadata_block_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_math_artifact_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_symbol_artifact_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_bracketed_role_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_ascii_role_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_cjk_role_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_short_speaker_line_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_short_cjk_fragment_sequence_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_leading_generated_question_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_malicious_self_dialogue_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_self_dialogue_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_generated_user_turn_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_followup_offer_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_inline_followup_question_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_glued_ascii_tail_artifact_stop(text) {
        best = earliest_stop(best, stop);
    }

    if let Some(stop) = find_runaway_enumeration_stop(text) {
        best = earliest_stop(best, stop);
    }

    best
}

fn find_text_stop_for_generation(text: &str, user_prompt: Option<&str>) -> Option<TextStop> {
    if let Some(user_prompt) = user_prompt {
        if let Some(stop) = find_contextual_text_stop(text, user_prompt) {
            return Some(stop);
        }
    }

    let stop = find_text_stop(text)?;
    if stop.marker == "leading-question"
        && user_prompt.is_some_and(|prompt| greeting_allows_leading_question(text, prompt))
    {
        return None;
    }
    if matches!(
        stop.marker,
        "inline-followup-question" | "followup-offer" | "generated-user-turn"
    ) && user_prompt
        .is_some_and(|prompt| greeting_allows_inline_help_question(text, prompt, stop.pos))
    {
        return None;
    }

    Some(stop)
}

fn find_contextual_text_stop(text: &str, user_prompt: &str) -> Option<TextStop> {
    if is_simple_greeting_prompt_text(user_prompt) && looks_like_greeting_answer_drift(text) {
        return Some(TextStop {
            pos: 0,
            marker: "greeting-drift",
        });
    }

    None
}

fn is_simple_greeting_prompt_text(text: &str) -> bool {
    let norm = normalize_echo_text(text);
    is_common_greeting_prompt(&norm)
}

fn greeting_allows_leading_question(generated: &str, user_prompt: &str) -> bool {
    if !is_simple_greeting_prompt_text(user_prompt) {
        return false;
    }

    let Some(question) = leading_question_text(generated) else {
        return false;
    };
    is_allowed_greeting_help_question(question)
}

fn greeting_allows_inline_help_question(
    generated: &str,
    user_prompt: &str,
    stop_pos: usize,
) -> bool {
    if !is_simple_greeting_prompt_text(user_prompt) || stop_pos > generated.len() {
        return false;
    }

    let prefix = generated[..stop_pos].trim();
    let prefix_norm = normalize_echo_text(prefix);
    if !["你好", "您好", "嗨", "哈喽", "哈啰", "hello", "hi"]
        .iter()
        .any(|greeting| prefix_norm.starts_with(greeting))
    {
        return false;
    }

    let Some(question) = leading_question_text(&generated[stop_pos..]) else {
        return false;
    };
    is_allowed_greeting_help_question(question)
}

fn is_allowed_greeting_help_question(question: &str) -> bool {
    let compact = normalize_echo_text(question);
    let mut candidates = vec![compact.as_str()];
    if let Some(stripped) = compact
        .strip_prefix('您')
        .or_else(|| compact.strip_prefix('你'))
    {
        candidates.push(stripped);
    }

    if candidates.iter().any(|candidate| {
        [
            "你好吗",
            "您好吗",
            "你好么",
            "您好么",
            "有什么可以帮你",
            "有什么可以帮您",
            "请问有什么可以帮你",
            "请问有什么可以帮您",
            "有什么需要帮忙",
            "有什么问题",
            "请问有什么问题",
            "你想了解",
            "您想了解",
            "想了解",
            "你想知道",
            "您想知道",
            "想知道",
            "你需要什么",
            "您需要什么",
            "需要什么",
            "请问你需要什么",
            "请问您需要什么",
            "需要我帮忙吗",
            "请问需要帮助吗",
            "需要帮助吗",
        ]
        .iter()
        .any(|allowed| candidate.starts_with(allowed))
    }) {
        return true;
    }

    if compact.contains("帮") || compact.contains("帮助") {
        return [
            "请问",
            "有什么",
            "有啥",
            "需要",
            "我能",
            "我可以",
            "能为",
            "可以为",
        ]
        .iter()
        .any(|prefix| {
            candidates
                .iter()
                .any(|candidate| candidate.starts_with(prefix))
        });
    }

    candidates.iter().any(|candidate| {
        [
            "我可以为你做些什么",
            "我可以为您做些什么",
            "我能为你做些什么",
            "我能为您做些什么",
            "能为你做些什么",
            "能为您做些什么",
            "可以为你做些什么",
            "可以为您做些什么",
            "为你效劳",
            "为您效劳",
        ]
        .iter()
        .any(|allowed| candidate.starts_with(allowed))
    })
}

fn leading_question_text(text: &str) -> Option<&str> {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))?;
    if start > 8 {
        return None;
    }

    for (rel, ch) in text[start..].char_indices() {
        if matches!(ch, '。' | '！' | '!' | '；' | ';' | '.') {
            return None;
        }
        if matches!(ch, '？' | '?') {
            let end = start + rel + ch.len_utf8();
            return Some(text[start..end].trim());
        }
    }

    None
}

fn looks_like_greeting_answer_drift(text: &str) -> bool {
    let trimmed = text.trim_start();
    let lower = trimmed.to_ascii_lowercase();

    if [
        "现在我想知道",
        "接下来我想知道",
        "我想知道如何",
        "我需要知道如何",
        "首先，我们需要",
        "然后，我们可以",
    ]
    .iter()
    .any(|marker| trimmed.contains(marker))
    {
        return true;
    }

    [
        "pandas",
        "csv",
        "import pandas",
        "dataframe",
        "python",
        "with open(",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn find_wrong_stardew_title_stop(text: &str) -> Option<TextStop> {
    let mut best = None;
    for &marker in WRONG_STARDEW_TITLE_STOPS {
        if let Some(pos) = text.find(marker) {
            best = earliest_stop(
                best,
                TextStop {
                    pos: wrong_stardew_title_stop_start(text, pos),
                    marker: "wrong-stardew-title",
                },
            );
        }
    }
    best
}

fn wrong_stardew_title_stop_start(text: &str, pos: usize) -> usize {
    let line_start = line_start_before(text, pos);
    let line_prefix = text[line_start..pos].trim_start();
    if line_prefix.chars().count() <= 48
        && (line_prefix.contains('《')
            || line_prefix.contains("关于")
            || line_prefix.contains("指")
            || line_prefix.contains("是"))
    {
        return line_start;
    }

    pos
}

fn normalize_wrong_stardew_titles_in_unflushed_output(
    text_acc: &mut String,
    token_buf: &mut std::collections::VecDeque<String>,
) {
    let buf_len = buffered_len(token_buf);
    let buf_start = text_acc.len().saturating_sub(buf_len);
    if buf_start != 0 {
        return;
    }

    let mut normalized = text_acc.clone();
    for &marker in WRONG_STARDEW_TITLE_STOPS {
        normalized = normalized.replace(marker, "星露谷物语");
    }

    if normalized == *text_acc {
        return;
    }

    *text_acc = normalized;
    token_buf.clear();
    if !text_acc.is_empty() {
        token_buf.push_back(text_acc.clone());
    }
}

fn find_dataset_artifact_stop(text: &str) -> Option<TextStop> {
    let mut best = None;
    let lower = text.to_ascii_lowercase();

    for &marker in ASCII_DATASET_ARTIFACT_STOPS {
        for (pos, _) in lower.match_indices(marker) {
            if text.is_char_boundary(pos)
                && text.is_char_boundary(pos + marker.len())
                && ascii_dataset_artifact_boundary(text, marker, pos, pos + marker.len())
            {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos,
                        marker: "dataset-artifact",
                    },
                );
            }
        }
    }

    for &marker in CJK_DATASET_ARTIFACT_STOPS {
        for (pos, _) in text.match_indices(marker) {
            if cjk_dataset_artifact_boundary(text, pos, pos + marker.len()) {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos,
                        marker: "dataset-artifact",
                    },
                );
            }
        }
    }

    best
}

fn ascii_dataset_artifact_boundary(text: &str, marker: &str, pos: usize, after: usize) -> bool {
    if marker == "koa" {
        return koa_artifact_boundary(text, pos, after);
    }

    let before_ok = text[..pos]
        .chars()
        .next_back()
        .is_none_or(|c| c.is_whitespace() || "。！？.!?；;，,：:~[]".contains(c));
    let after_ok = text[after..].chars().next().is_none_or(|c| {
        c.is_whitespace()
            || "。！？.!?；;，,：:$[]_\\/".contains(c)
            || (marker.len() >= 6 && c.is_ascii_alphanumeric())
    });
    before_ok && after_ok
}

fn koa_artifact_boundary(text: &str, pos: usize, after: usize) -> bool {
    let before_ok = prefix_looks_complete_before_artifact(&text[..pos]);
    let after_ok = text[after..]
        .chars()
        .next()
        .is_none_or(|c| c.is_whitespace() || "$[]_\\/。！？.!?；;，,：:".contains(c));
    before_ok && after_ok
}

fn prefix_looks_complete_before_artifact(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    trimmed
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '。' | '！' | '!' | '？' | '?' | '~' | '.'))
}

fn cjk_dataset_artifact_boundary(text: &str, pos: usize, after: usize) -> bool {
    let before_ok = text[..pos]
        .chars()
        .next_back()
        .is_none_or(|c| c.is_whitespace() || "。！？.!?；;，,：:~".contains(c));
    let after_ok = text[after..].chars().next().is_none_or(|c| {
        c.is_ascii_alphanumeric() || c.is_whitespace() || "。！？.!?；;，,：:包装".contains(c)
    });
    before_ok && after_ok
}

fn find_metadata_block_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for (line_start, line) in logical_lines_with_starts(text) {
        if line_start == 0 {
            continue;
        }

        let trimmed = line.trim_start();
        if !starts_like_unrelated_metadata_line(trimmed) {
            continue;
        }

        if !prefix_looks_complete_before_artifact(&text[..line_start]) {
            continue;
        }

        best = earliest_stop(
            best,
            TextStop {
                pos: line_start,
                marker: "metadata-block",
            },
        );
    }

    best
}

fn logical_lines_with_starts(text: &str) -> Vec<(usize, &str)> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in text.char_indices() {
        if ch == '\n' {
            lines.push((start, &text[start..idx]));
            start = idx + ch.len_utf8();
        }
    }
    lines.push((start, &text[start..]));
    lines
}

fn starts_like_unrelated_metadata_line(line: &str) -> bool {
    [
        "货币：",
        "货币:",
        "货币单位：",
        "货币单位:",
        "汇率：",
        "汇率:",
        "汇率日期：",
        "汇率日期:",
        "Currency:",
        "Exchange rate:",
        "Exchange Rate:",
    ]
    .iter()
    .any(|marker| line.starts_with(marker))
}

fn find_math_artifact_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for &marker in MATH_PARSE_ERROR_STOPS {
        if let Some(pos) = text.find(marker) {
            best = earliest_stop(
                best,
                TextStop {
                    pos,
                    marker: "math-parse-artifact",
                },
            );
        }
    }

    if let Some(pos) = find_dollar_bracket_storm(text) {
        best = earliest_stop(
            best,
            TextStop {
                pos,
                marker: "math-symbol-storm",
            },
        );
    }

    best
}

fn find_symbol_artifact_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for &marker in SYMBOL_ARTIFACT_STOPS {
        let haystack = if marker.is_ascii() {
            text.to_ascii_lowercase()
        } else {
            text.to_string()
        };
        let marker_lower = marker.to_ascii_lowercase();
        for (pos, _) in haystack.match_indices(&marker_lower) {
            if !text.is_char_boundary(pos) {
                continue;
            }
            if symbol_artifact_boundary(text, pos) {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos,
                        marker: "symbol-artifact",
                    },
                );
            }
        }
    }

    best
}

fn symbol_artifact_boundary(text: &str, pos: usize) -> bool {
    let prefix = &text[..pos];
    if prefix_looks_complete_before_artifact(prefix)
        || prefix_looks_complete_before_self_question(prefix)
    {
        return true;
    }

    let line_start = line_start_before(text, pos);
    line_start < pos && text[line_start..pos].trim().is_empty()
}

fn find_dollar_bracket_storm(text: &str) -> Option<usize> {
    for (pos, ch) in text.char_indices() {
        if ch != '$' {
            continue;
        }

        let sample: String = text[pos..].chars().take(96).collect();
        let dollar_count = sample.matches('$').count();
        let artifact_count = sample.chars().filter(|c| "$[]_\\/".contains(*c)).count();
        let has_bracket_or_underscore = sample.chars().any(|c| matches!(c, '[' | ']' | '_'));

        if dollar_count >= 3 && artifact_count >= 6 && has_bracket_or_underscore {
            return Some(pos);
        }
    }

    None
}

fn find_bracketed_role_stop(text: &str) -> Option<TextStop> {
    let lower = text.to_ascii_lowercase();
    let mut best = None;

    for &marker in BRACKETED_ROLE_STOPS {
        let haystack = if marker.is_ascii() {
            lower.as_str()
        } else {
            text
        };
        for (pos, _) in haystack.match_indices(marker) {
            if text.is_char_boundary(pos) && bracketed_role_boundary(text, pos) {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos,
                        marker: "bracketed-role",
                    },
                );
            }
        }
    }

    best
}

fn bracketed_role_boundary(text: &str, pos: usize) -> bool {
    text[..pos]
        .chars()
        .next_back()
        .is_none_or(|c| c.is_whitespace() || ROLE_PREFIX_PUNCTUATION.contains(c))
}

fn find_ascii_role_stop(text: &str) -> Option<TextStop> {
    let lower = text.to_ascii_lowercase();
    let mut best = None;

    for &role in ASCII_ROLE_STOPS {
        let mut offset = 0usize;
        while let Some(rel) = lower[offset..].find(role) {
            let pos = offset + rel;
            let after = pos + role.len();
            if text.is_char_boundary(pos)
                && text.is_char_boundary(after)
                && role_label_stop_pos(text, pos, after).is_some()
            {
                let stop_pos = role_label_stop_pos(text, pos, after).unwrap_or(pos);
                best = earliest_stop(
                    best,
                    TextStop {
                        pos: stop_pos,
                        marker: role,
                    },
                );
            }
            offset = after;
        }
    }

    best
}

fn find_cjk_role_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for &role in CJK_ROLE_STOPS {
        for (pos, _) in text.match_indices(role) {
            let after = pos + role.len();
            if let Some(stop_pos) = role_label_stop_pos(text, pos, after) {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos: stop_pos,
                        marker: role,
                    },
                );
            }
        }
    }

    best
}

fn find_short_speaker_line_stop(text: &str) -> Option<TextStop> {
    let mut best = None;
    let mut search_start = 0usize;

    while let Some(rel) = text[search_start..].find('\n') {
        let line_start = search_start + rel + 1;
        let line_end = text[line_start..]
            .find('\n')
            .map(|next| line_start + next)
            .unwrap_or(text.len());
        let label = text[line_start..line_end].trim();

        if is_suspicious_speaker_label(label) {
            best = earliest_stop(
                best,
                TextStop {
                    pos: line_start,
                    marker: "speaker-label",
                },
            );
            break;
        }

        if line_end == text.len() {
            break;
        }
        search_start = line_end;
    }

    best
}

fn is_suspicious_speaker_label(label: &str) -> bool {
    let char_count = label.chars().count();
    if !(2..=8).contains(&char_count) {
        return false;
    }

    if label
        .chars()
        .any(|c| c.is_whitespace() || "。！？.!?；;，,：:、\"'“”‘’（）()[]【】".contains(c))
    {
        return false;
    }

    let has_cjk = label
        .chars()
        .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    has_cjk && (label.ends_with('助') || label.ends_with("助手") || label.ends_with("用户"))
}

fn find_short_cjk_fragment_sequence_stop(text: &str) -> Option<TextStop> {
    let mut first_fragment_start = None;
    let mut fragment_count = 0usize;

    for (line_start, line) in logical_lines_with_starts(text).into_iter().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if is_short_cjk_generation_fragment(trimmed) {
            if first_fragment_start.is_none() {
                first_fragment_start = Some(line_start);
            }
            fragment_count += 1;

            if fragment_count >= 2 {
                let pos = first_fragment_start.unwrap_or(line_start);
                if prefix_looks_complete_before_self_question(&text[..pos]) {
                    return Some(TextStop {
                        pos,
                        marker: "short-cjk-fragment-sequence",
                    });
                }
            }
            continue;
        }

        if looks_like_generated_question(trimmed) {
            continue;
        }

        if trimmed.chars().count() > 16 || ends_with_sentence_terminal(trimmed) {
            first_fragment_start = None;
            fragment_count = 0;
        }
    }

    None
}

fn is_short_cjk_generation_fragment(text: &str) -> bool {
    let char_count = text.chars().count();
    (1..=4).contains(&char_count)
        && !text
            .chars()
            .any(|c| c.is_whitespace() || "。！？.!?；;，,：:、\"'“”‘’（）()[]【】".contains(c))
        && text.chars().all(is_cjk_unified_ideograph)
}

fn find_self_dialogue_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for (question_mark_pos, ch) in text.char_indices() {
        if !matches!(ch, '？' | '?') {
            continue;
        }

        let question_end = question_mark_pos + ch.len_utf8();
        let question_start = question_start_before(text, question_mark_pos);
        let question = text[question_start..question_end].trim();
        if !looks_like_generated_question(question) {
            continue;
        }

        if !prefix_looks_complete_before_self_question(&text[..question_start]) {
            continue;
        }

        let after = strip_leading_generation_artifacts(&text[question_end..]);
        if question_starts_standalone_line(text, question_start) || starts_like_self_answer(after) {
            best = earliest_stop(
                best,
                TextStop {
                    pos: question_start,
                    marker: "self-dialogue",
                },
            );
        }
    }

    best
}

fn find_malicious_self_dialogue_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for marker in malicious_self_dialogue_markers() {
        if let Some(pos) = text.find(marker) {
            best = earliest_stop(
                best,
                TextStop {
                    pos: line_start_before(text, pos),
                    marker: "malicious-self-dialogue",
                },
            );
        }
    }

    best
}

fn find_generated_user_turn_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for start in generated_user_turn_candidate_starts(text) {
        let tail = text[start..].trim_start();
        if tail.is_empty() {
            continue;
        }

        if complete_generated_user_turn_tail(tail) {
            best = earliest_stop(
                best,
                TextStop {
                    pos: start,
                    marker: "generated-user-turn",
                },
            );
        }
    }

    best
}

fn potential_generated_user_turn_start(text: &str) -> Option<usize> {
    for start in generated_user_turn_candidate_starts(text) {
        let tail = text[start..].trim_start();
        if tail.is_empty() || tail.chars().count() > 128 {
            continue;
        }

        if generated_user_turn_prefix(tail) {
            return Some(start);
        }
    }

    None
}

fn generated_user_turn_candidate_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    for (terminal_pos, ch) in text.char_indices() {
        if !matches!(ch, '。' | '！' | '!' | '？' | '?' | '~') {
            continue;
        }

        let start = terminal_pos + ch.len_utf8();
        if start >= text.len() {
            continue;
        }

        let prefix = &text[..start];
        if contains_cjk_unified_ideograph(prefix)
            && prefix_looks_complete_before_self_question(prefix)
        {
            starts.push(start);
        }
    }
    starts
}

fn complete_generated_user_turn_tail(tail: &str) -> bool {
    if !generated_user_turn_prefix(tail) {
        return false;
    }

    let sample: String = tail.chars().take(160).collect();
    if sample.contains(['？', '?']) {
        return true;
    }

    let stripped = strip_leading_generation_artifacts(&sample);
    if stripped.len() < sample.len() {
        return true;
    }

    generated_user_statement_prefix(&sample)
        && sample
            .chars()
            .any(|ch| matches!(ch, '。' | '！' | '!' | '；' | ';'))
}

fn generated_user_turn_prefix(text: &str) -> bool {
    let trimmed = text.trim_start();
    generated_user_question_prefix(trimmed) || generated_user_statement_prefix(trimmed)
}

fn generated_user_question_prefix(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "好的，谢谢",
        "好的，謝謝",
        "好的，非常感谢",
        "好的，非常感謝",
        "谢谢",
        "謝謝",
        "非常感谢",
        "非常感謝",
        "那我",
        "那你",
        "那您",
        "那请",
        "那請",
        "那现在",
        "那現在",
        "那接下来",
        "那接下來",
        "接下来我",
        "接下來我",
        "顺便问",
        "順便問",
        "我想问",
        "我想問",
        "我还想",
        "我還想",
        "我需要",
    ]
    .iter()
    .any(|marker| marker.starts_with(text) || text.starts_with(marker))
        || [
            "thanks",
            "thank you",
            "okay, thanks",
            "ok, thanks",
            "then i",
            "can you",
            "could you",
            "where can i",
        ]
        .iter()
        .any(|marker| marker.starts_with(&lower) || lower.starts_with(marker))
}

fn generated_user_statement_prefix(text: &str) -> bool {
    [
        "那我现在",
        "那我現在",
        "我现在就",
        "我現在就",
        "那我就",
        "好的，我现在",
        "好的，我現在",
    ]
    .iter()
    .any(|marker| marker.starts_with(text) || text.starts_with(marker))
}

fn find_followup_offer_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for marker in [
        "您需要我",
        "你需要我",
        "需要我为您",
        "需要我帮您",
        "是否需要我",
        "要我继续",
        "要我帮你",
        "还需要我",
        "还有什么需要",
        "有什么其他需要",
        "Can I help you",
        "Do you need me",
    ] {
        for (pos, _) in text.match_indices(marker) {
            if pos == 0 || !prefix_looks_complete_before_self_question(&text[..pos]) {
                continue;
            }

            let tail: String = text[pos..].chars().take(96).collect();
            if tail.contains('？') || tail.contains('?') {
                best = earliest_stop(
                    best,
                    TextStop {
                        pos,
                        marker: "followup-offer",
                    },
                );
            }
        }
    }

    best
}

fn find_inline_followup_question_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for (question_mark_pos, ch) in text.char_indices() {
        if !matches!(ch, '？' | '?') {
            continue;
        }

        let question_end = question_mark_pos + ch.len_utf8();
        let question_start = question_start_before(text, question_mark_pos);
        if question_start == 0 {
            continue;
        }

        let question = text[question_start..question_end].trim();
        if !looks_like_generated_question(question)
            || !looks_like_inline_followup_question(question)
            || starts_like_list_item(&text[line_start_before(text, question_start)..question_start])
        {
            continue;
        }

        if !prefix_looks_complete_before_self_question(&text[..question_start]) {
            continue;
        }

        if !inline_followup_question_has_bad_tail(&text[question_end..]) {
            continue;
        }

        best = earliest_stop(
            best,
            TextStop {
                pos: question_start,
                marker: "inline-followup-question",
            },
        );
    }

    best
}

fn inline_followup_question_has_bad_tail(tail: &str) -> bool {
    if tail.trim().is_empty() {
        return true;
    }

    if starts_with_glued_ascii_tail(tail) {
        return true;
    }

    let stripped = strip_leading_generation_artifacts(tail);
    stripped.len() < tail.len() && (stripped.is_empty() || starts_like_self_answer(stripped))
}

fn starts_with_glued_ascii_tail(text: &str) -> bool {
    let Some(first) = text.chars().next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }

    let char_count = text.chars().count();
    (3..=32).contains(&char_count)
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

fn find_glued_ascii_tail_artifact_stop(text: &str) -> Option<TextStop> {
    let mut best = None;

    for (terminal_pos, ch) in text.char_indices() {
        if !matches!(ch, '。' | '！' | '？' | '!' | '?') {
            continue;
        }

        let after_terminal = terminal_pos + ch.len_utf8();
        let tail = &text[after_terminal..];
        let Some(first_tail) = tail.chars().next() else {
            continue;
        };

        if !first_tail.is_ascii_alphabetic() {
            continue;
        }

        let tail_char_count = tail.chars().count();
        if !(3..=32).contains(&tail_char_count) {
            continue;
        }

        if !tail
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        {
            continue;
        }

        let prefix = &text[..after_terminal];
        if !contains_cjk_unified_ideograph(prefix)
            || prefix[..terminal_pos]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric())
        {
            continue;
        }

        best = earliest_stop(
            best,
            TextStop {
                pos: after_terminal,
                marker: "glued-ascii-tail",
            },
        );
    }

    best
}

fn find_leading_generated_question_stop(text: &str) -> Option<TextStop> {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))?;
    if start > 8 {
        return None;
    }

    for (rel, ch) in text[start..].char_indices() {
        if matches!(ch, '。' | '！' | '!' | '；' | ';' | '.') {
            return None;
        }
        if !matches!(ch, '？' | '?') {
            continue;
        }

        let question_end = start + rel + ch.len_utf8();
        let question = text[start..question_end].trim();
        if looks_like_leading_user_turn_question(question) {
            return Some(TextStop {
                pos: start,
                marker: "leading-question",
            });
        }
        break;
    }

    None
}

fn potential_leading_question_start(text: &str) -> Option<usize> {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))?;
    if start > 8 {
        return None;
    }

    let prefix = text[start..].trim();
    if prefix.is_empty() || prefix.chars().count() > 80 {
        return None;
    }

    if prefix
        .chars()
        .any(|ch| matches!(ch, '。' | '！' | '!' | '；' | ';' | '.'))
    {
        return None;
    }
    if prefix.chars().any(|ch| matches!(ch, '？' | '?')) {
        return None;
    }

    let lower = prefix.to_ascii_lowercase();
    let likely_question = prefix == "你"
        || [
            "你叫",
            "你是",
            "你有",
            "你想",
            "你需要",
            "你为什么",
            "你怎么",
            "你什么",
            "请问",
            "为什么",
            "怎么",
            "什么",
            "谁",
            "哪里",
            "是否",
            "能否",
            "有没有",
            "是不是",
            "会不会",
            "可不可以",
        ]
        .iter()
        .any(|marker| prefix.starts_with(marker))
        || [
            "what", "why", "when", "where", "who", "how", "can ", "could ", "do ", "does ", "is ",
            "are ",
        ]
        .iter()
        .any(|marker| lower.starts_with(marker));

    likely_question.then_some(start)
}

fn looks_like_generated_question(question: &str) -> bool {
    let question = question.trim();
    let char_count = question.chars().count();
    if !(4..=80).contains(&char_count) {
        return false;
    }

    if !question.ends_with(['？', '?']) {
        return false;
    }

    let has_cjk = question
        .chars()
        .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    if has_cjk {
        return true;
    }

    let lower = question.to_ascii_lowercase();
    [
        "what", "why", "when", "where", "who", "how", "can ", "could ", "do ", "does ", "is ",
        "are ",
    ]
    .iter()
    .any(|marker| lower.starts_with(marker))
}

fn looks_like_inline_followup_question(question: &str) -> bool {
    let question = question.trim();
    let lower = question.to_ascii_lowercase();

    [
        "您",
        "你",
        "请问",
        "是否",
        "能否",
        "要不要",
        "想不想",
        "有没有",
        "哪个",
        "哪一",
        "哪方面",
        "什么方面",
        "还想",
        "还需要",
        "需要我",
    ]
    .iter()
    .any(|marker| question.starts_with(marker))
        || [
            "do you",
            "would you",
            "which ",
            "what kind",
            "what aspect",
            "are you interested",
        ]
        .iter()
        .any(|marker| lower.starts_with(marker) || lower.contains(marker))
}

fn looks_like_leading_user_turn_question(question: &str) -> bool {
    let question = question.trim();
    if !looks_like_generated_question(question) {
        return false;
    }

    let lower = question.to_ascii_lowercase();
    if [
        "what is your",
        "what's your",
        "who are you",
        "why do you",
        "why are you",
        "how do you",
        "can you",
    ]
    .iter()
    .any(|marker| lower.starts_with(marker))
    {
        return true;
    }

    [
        "你叫什么",
        "你叫啥",
        "你是谁",
        "你是什么",
        "你为什么",
        "你怎么",
        "你是否",
        "你会不会",
        "你能不能",
        "你什么时候",
        "现在是哪",
        "现在是几",
        "现在几月",
        "今天是哪",
        "今天是几",
        "今天几号",
        "今天几月",
        "现在还没",
        "背包里",
        "明天天气",
        "今天天气",
        "我想知道",
        "我需要",
        "我所在",
    ]
    .iter()
    .any(|marker| question.starts_with(marker))
}

fn question_starts_standalone_line(text: &str, question_start: usize) -> bool {
    let line_start = line_start_before(text, question_start);
    line_start > 0 && text[line_start..question_start].trim().is_empty()
}

fn question_start_before(text: &str, pos: usize) -> usize {
    text[..pos]
        .char_indices()
        .rev()
        .find_map(|(idx, ch)| {
            if matches!(ch, '\n' | '。' | '！' | '!' | '？' | '?' | '；' | ';') {
                Some(idx + ch.len_utf8())
            } else {
                None
            }
        })
        .unwrap_or(0)
}

fn prefix_looks_complete_before_self_question(prefix: &str) -> bool {
    let trimmed = prefix.trim_end();
    if trimmed
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '。' | '！' | '!' | '？' | '?' | '~'))
    {
        return true;
    }

    ["祝您", "希望", "即可", "就可以了", "完成"]
        .iter()
        .any(|marker| {
            trimmed
                .rfind(marker)
                .is_some_and(|pos| trimmed.len().saturating_sub(pos) <= 96)
        })
}

fn starts_like_self_answer(text: &str) -> bool {
    [
        "好",
        "当然",
        "若要",
        "可以",
        "是的",
        "不是",
        "不用",
        "没错",
        "如果",
        "根据",
        "据我所知",
        "作为",
        "您好",
        "你好",
        "非常",
        "鲜香",
        "会",
    ]
    .iter()
    .any(|starter| text.starts_with(starter))
}

fn strip_leading_generation_artifacts(mut text: &str) -> &str {
    loop {
        let before = text;
        text = text
            .trim_start_matches(|c: char| c.is_whitespace() || "。！？.!?；;，,：:~".contains(c));

        for &marker in CJK_DATASET_ARTIFACT_STOPS {
            if let Some(rest) = text.strip_prefix(marker) {
                text = rest;
                break;
            }
        }

        for &marker in ASCII_DATASET_ARTIFACT_STOPS {
            if text
                .get(..marker.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(marker))
            {
                text = &text[marker.len()..];
                break;
            }
        }

        if text == before {
            return text;
        }
    }
}

fn find_runaway_enumeration_stop(text: &str) -> Option<TextStop> {
    if text.matches("给你一个").count() < 4 || !text.contains("最后") {
        return None;
    }

    for marker in [
        "\n我再最后",
        "\r\n我再最后",
        " 我再最后",
        "\n再最后",
        " 再最后",
    ] {
        if let Some(pos) = text.find(marker) {
            return Some(TextStop {
                pos,
                marker: "runaway-enumeration",
            });
        }
    }

    None
}

fn line_start_before(text: &str, pos: usize) -> usize {
    text[..pos].rfind('\n').map(|idx| idx + 1).unwrap_or(0)
}

fn role_label_stop_pos(text: &str, pos: usize, after: usize) -> Option<usize> {
    if !is_role_delimiter_or_text_end(text, after) {
        return None;
    }

    let line_start = line_start_before(text, pos);
    let prefix = text[line_start..pos].trim_matches(|c| c == '\r' || c == ' ' || c == '\t');
    if prefix.is_empty()
        || (prefix.len() <= 32
            && prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
    {
        return Some(line_start);
    }

    if text[..pos]
        .chars()
        .next_back()
        .is_some_and(|c| ROLE_PREFIX_PUNCTUATION.contains(c))
    {
        return Some(pos);
    }

    None
}

fn is_role_delimiter_or_text_end(text: &str, pos: usize) -> bool {
    text[pos..]
        .chars()
        .next()
        .is_none_or(|c| matches!(c, ':' | '：' | '\n' | '\r'))
}

fn clean_stop_prefix_len(prefix: &str) -> usize {
    let mut end = trim_trailing_ws_len(prefix);

    loop {
        let trimmed = &prefix[..end];
        let mut changed = false;

        for &artifact in GENERATED_BOUNDARY_ARTIFACTS {
            if let Some(new_end) = strip_boundary_artifact(trimmed, artifact) {
                end = trim_trailing_ws_len(&prefix[..new_end]);
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }

    end
}

fn strip_boundary_artifact(text: &str, artifact: &str) -> Option<usize> {
    if !text.ends_with(artifact) {
        return None;
    }

    let new_end = text.len() - artifact.len();
    let before = text[..new_end].chars().next_back();
    if before.is_none_or(|c| c.is_whitespace() || "。！？.!?；;，,：:".contains(c)) {
        Some(new_end)
    } else {
        None
    }
}

fn trim_trailing_ws_len(text: &str) -> usize {
    let mut end = text.len();
    for (idx, ch) in text.char_indices().rev() {
        if ch.is_whitespace() {
            end = idx;
        } else {
            break;
        }
    }
    end
}

fn buffered_len(token_buf: &std::collections::VecDeque<String>) -> usize {
    token_buf.iter().map(|s| s.len()).sum()
}

fn flush_prefix_from_buffer(
    token_buf: &mut std::collections::VecDeque<String>,
    mut bytes_to_send: usize,
    token_tx: &mpsc::Sender<Option<String>>,
) -> bool {
    while bytes_to_send > 0 {
        let Some(front) = token_buf.pop_front() else {
            return true;
        };

        if front.len() <= bytes_to_send {
            bytes_to_send -= front.len();
            if token_tx.blocking_send(Some(front)).is_err() {
                return false;
            }
            continue;
        }

        let split = floor_char_boundary(&front, bytes_to_send);
        if split == 0 {
            token_buf.push_front(front);
            return true;
        }

        let prefix = front[..split].to_string();
        let suffix = front[split..].to_string();
        token_buf.push_front(suffix);
        if token_tx.blocking_send(Some(prefix)).is_err() {
            return false;
        }
        return true;
    }

    true
}

fn flush_clean_prefix_from_acc(
    text_acc: &str,
    token_buf: &mut std::collections::VecDeque<String>,
    token_tx: &mpsc::Sender<Option<String>>,
) -> bool {
    let clean_end = if let Some(stop) = find_text_stop(text_acc) {
        clean_stop_prefix_len(&text_acc[..stop.pos])
    } else if let Some((_, _, repeat_start)) = repeated_tail_phrase(text_acc) {
        trim_trailing_ws_len(&text_acc[..repeat_start])
    } else if let Some((_, _, repeat_start)) = repeated_ngram(text_acc) {
        trim_trailing_ws_len(&text_acc[..repeat_start])
    } else {
        trim_trailing_ws_len(text_acc)
    };

    let buf_len = buffered_len(token_buf);
    let buf_start = text_acc.len().saturating_sub(buf_len);
    let safe_end = clean_end.saturating_sub(buf_start).min(buf_len);
    flush_prefix_from_buffer(token_buf, safe_end, token_tx)
}

fn flush_ready_buffer(
    text_acc: &str,
    token_buf: &mut std::collections::VecDeque<String>,
    token_tx: &mpsc::Sender<Option<String>>,
) -> bool {
    let len = buffered_len(token_buf);
    if len == 0 {
        return true;
    }

    let buf_start = text_acc.len().saturating_sub(len);
    let hold_start = guard_hold_start(text_acc).unwrap_or(text_acc.len());
    let bytes_to_send = hold_start.saturating_sub(buf_start).min(len);
    flush_prefix_from_buffer(token_buf, bytes_to_send, token_tx)
}

fn guard_hold_start(text: &str) -> Option<usize> {
    let mut hold_start = partial_stop_marker_start(text);

    if let Some(pos) = potential_short_guarded_answer_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_malicious_self_dialogue_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = partial_speaker_or_artifact_line_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_wrong_stardew_title_line_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_self_question_line_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_leading_question_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_math_artifact_line_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_symbol_artifact_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_generated_user_turn_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_metadata_line_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_followup_offer_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_inline_followup_question_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    if let Some(pos) = potential_glued_ascii_tail_start(text) {
        hold_start = match hold_start {
            Some(existing) => Some(existing.min(pos)),
            None => Some(pos),
        };
    }

    hold_start
}

fn potential_short_guarded_answer_start(text: &str) -> Option<usize> {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))
        .unwrap_or(text.len());
    let trimmed = text[start..].trim();
    if trimmed.is_empty() || trimmed.chars().count() > 48 {
        return None;
    }

    looks_like_short_guarded_answer_prefix(trimmed).then_some(start)
}

fn potential_malicious_self_dialogue_start(text: &str) -> Option<usize> {
    let start = text
        .char_indices()
        .find_map(|(idx, ch)| (!ch.is_whitespace()).then_some(idx))
        .unwrap_or(text.len());
    let trimmed = text[start..].trim();
    if trimmed.is_empty() || trimmed.chars().count() > 128 {
        return None;
    }

    [
        "我刚才说",
        "我剛才說",
        "我刚刚说",
        "我剛剛說",
        "你没有听清楚",
        "你沒有聽清楚",
        "你应该按照我的指示",
        "你應該按照我的指示",
    ]
    .iter()
    .any(|marker| marker.starts_with(trimmed) || trimmed.starts_with(marker))
    .then_some(start)
}

fn partial_stop_marker_start(text: &str) -> Option<usize> {
    let mut best = None;

    for marker in partial_guard_markers() {
        for prefix_len in marker
            .char_indices()
            .map(|(idx, ch)| idx + ch.len_utf8())
            .filter(|&len| len < marker.len())
        {
            let prefix = &marker[..prefix_len];
            if text.ends_with(prefix) {
                let start = text.len() - prefix.len();
                if partial_marker_boundary_ok(text, start) {
                    best = Some(best.map_or(start, |existing: usize| existing.min(start)));
                }
            }
        }
    }

    best
}

fn potential_wrong_stardew_title_line_start(text: &str) -> Option<usize> {
    let line_start = line_start_before(text, text.len());
    let line = &text[line_start..];
    let trimmed_line = line.trim_start();

    if !trimmed_line.is_empty()
        && trimmed_line.chars().count() <= 2
        && "在《".starts_with(trimmed_line)
    {
        return Some(line_start);
    }

    let quote_pos = line.rfind('《')?;
    let after_quote = &line[quote_pos + '《'.len_utf8()..];

    if after_quote.chars().count() > 32 || after_quote.contains('\n') {
        return None;
    }

    if after_quote.is_empty()
        || WRONG_STARDEW_TITLE_STOPS
            .iter()
            .any(|marker| marker.starts_with(after_quote) || after_quote.starts_with(marker))
    {
        return Some(line_start);
    }

    None
}

fn potential_metadata_line_start(text: &str) -> Option<usize> {
    let line_start = line_start_before(text, text.len());
    if line_start == 0 || !prefix_looks_complete_before_artifact(&text[..line_start]) {
        return None;
    }

    let trimmed = text[line_start..].trim_start();
    if trimmed.is_empty() || trimmed.chars().count() > 48 {
        return None;
    }

    [
        "货币：",
        "货币:",
        "货币单位：",
        "货币单位:",
        "汇率：",
        "汇率:",
        "汇率日期：",
        "汇率日期:",
        "Currency:",
        "Exchange rate:",
        "Exchange Rate:",
    ]
    .iter()
    .any(|marker| marker.starts_with(trimmed) || trimmed.starts_with(marker))
    .then_some(line_start)
}

fn potential_followup_offer_start(text: &str) -> Option<usize> {
    for marker in [
        "您需要我",
        "你需要我",
        "需要我为您",
        "需要我帮您",
        "是否需要我",
        "要我继续",
        "要我帮你",
        "还需要我",
        "还有什么需要",
        "有什么其他需要",
    ] {
        for prefix_len in marker
            .char_indices()
            .map(|(idx, ch)| idx + ch.len_utf8())
            .filter(|&len| len < marker.len())
        {
            let prefix = &marker[..prefix_len];
            if let Some(pos) = text.rfind(prefix) {
                if pos + prefix.len() == text.len()
                    && pos > 0
                    && prefix_looks_complete_before_self_question(&text[..pos])
                {
                    return Some(pos);
                }
            }
        }

        if let Some(pos) = text.rfind(marker) {
            let tail = &text[pos..];
            if pos > 0
                && tail.chars().count() <= 96
                && !tail.contains('？')
                && !tail.contains('?')
                && prefix_looks_complete_before_self_question(&text[..pos])
            {
                return Some(pos);
            }
        }
    }

    None
}

fn potential_inline_followup_question_start(text: &str) -> Option<usize> {
    let start = inline_tail_after_complete_prefix_start(text)?;
    let tail = text[start..].trim_start();
    let trimmed_start = start + (text[start..].len() - tail.len());

    if tail.is_empty() || tail.chars().count() > 96 || starts_like_list_item(tail) {
        return None;
    }

    if tail.contains(['？', '?']) {
        return Some(trimmed_start);
    }

    if tail
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '。' | '！' | '!' | '；' | ';' | '.'))
    {
        return None;
    }

    potential_followup_question_prefix(tail).then_some(trimmed_start)
}

fn potential_glued_ascii_tail_start(text: &str) -> Option<usize> {
    let start = inline_tail_after_complete_prefix_start(text)?;
    let tail = &text[start..];
    let char_count = tail.chars().count();
    if !(1..=32).contains(&char_count) {
        return None;
    }

    tail.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .then_some(start)
}

fn inline_tail_after_complete_prefix_start(text: &str) -> Option<usize> {
    let (terminal_pos, terminal) = text
        .char_indices()
        .rev()
        .find(|(_, ch)| matches!(ch, '。' | '！' | '？' | '!' | '?'))?;
    let start = terminal_pos + terminal.len_utf8();
    if start >= text.len() {
        return None;
    }

    let prefix = &text[..start];
    if !contains_cjk_unified_ideograph(prefix)
        || !prefix_looks_complete_before_self_question(prefix)
    {
        return None;
    }

    Some(start)
}

fn potential_followup_question_prefix(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();

    [
        "您",
        "你",
        "请",
        "请问",
        "是否",
        "能否",
        "要",
        "要不",
        "要不要",
        "想",
        "想不",
        "想不想",
        "有",
        "有没有",
        "哪",
        "哪个",
        "哪一",
        "哪方",
        "哪方面",
        "什么",
        "什么方面",
        "还",
        "还想",
        "还需要",
        "需要",
        "需要我",
    ]
    .iter()
    .any(|marker| marker.starts_with(text) || text.starts_with(marker))
        || [
            "do you",
            "would you",
            "which ",
            "what kind",
            "what aspect",
            "are you interested",
        ]
        .iter()
        .any(|marker| marker.starts_with(&lower) || lower.starts_with(marker))
}

fn partial_guard_markers() -> impl Iterator<Item = &'static str> {
    DIRECT_TEXT_STOPS
        .iter()
        .chain(SYSTEM_LEAK_STOPS.iter())
        .chain(ASCII_ROLE_STOPS.iter())
        .chain(CJK_ROLE_STOPS.iter())
        .chain(BRACKETED_ROLE_STOPS.iter())
        .chain(WRONG_STARDEW_TITLE_STOPS.iter())
        .chain(SYMBOL_ARTIFACT_STOPS.iter())
        .chain(ASCII_DATASET_ARTIFACT_STOPS.iter())
        .chain(CJK_DATASET_ARTIFACT_STOPS.iter())
        .copied()
        .filter(|marker| !marker.starts_with('\n') && !marker.starts_with('\r'))
}

fn partial_marker_boundary_ok(text: &str, start: usize) -> bool {
    text[..start]
        .chars()
        .next_back()
        .is_none_or(|c| c.is_whitespace() || "。！？.!?；;，,：:~<>[]《》“”\"'（(".contains(c))
}

fn partial_speaker_or_artifact_line_start(text: &str) -> Option<usize> {
    let newline_pos = text.rfind('\n')?;
    let line_start = newline_pos + 1;
    let line = text[line_start..].trim();

    if line.is_empty() {
        return Some(newline_pos);
    }

    if line.chars().count() > 16 {
        return None;
    }

    if line
        .chars()
        .any(|c| c.is_whitespace() || "。！？.!?；;，,：:、\"'“”‘’（）()【】".contains(c))
    {
        return None;
    }

    let ascii_tag = line
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    let has_cjk = line.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));

    if ascii_tag || has_cjk {
        Some(newline_pos)
    } else {
        None
    }
}

fn potential_self_question_line_start(text: &str) -> Option<usize> {
    let newline_pos = text.rfind('\n')?;
    let line_start = newline_pos + 1;
    let prefix = &text[..newline_pos];
    if !prefix_looks_complete_before_self_question(prefix) {
        return None;
    }

    let line = text[line_start..].trim_start();
    if line.is_empty() {
        return Some(newline_pos);
    }

    if starts_like_list_item(line) {
        return None;
    }

    let char_count = line.chars().count();
    if char_count > 128 {
        return None;
    }

    if line.contains(['？', '?']) {
        return Some(newline_pos);
    }

    if line
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '。' | '！' | '!' | '；' | ';'))
    {
        return None;
    }

    Some(newline_pos)
}

fn potential_math_artifact_line_start(text: &str) -> Option<usize> {
    let newline_pos = text.rfind('\n')?;
    let line_start = newline_pos + 1;
    let prefix = &text[..newline_pos];
    if !prefix_looks_complete_before_artifact(prefix) {
        return None;
    }

    let line = text[line_start..].trim_start();
    if line.is_empty() {
        return Some(math_artifact_hold_start(text, newline_pos));
    }

    if !line.starts_with(['$', '[', '_', '\\', '/']) {
        return None;
    }

    if line.chars().count() > 96 {
        return None;
    }

    Some(math_artifact_hold_start(text, newline_pos))
}

fn potential_symbol_artifact_start(text: &str) -> Option<usize> {
    let start = inline_tail_after_complete_prefix_start(text).or_else(|| {
        let line_start = line_start_before(text, text.len());
        let prefix = &text[..line_start.saturating_sub(1)];
        (line_start > 0 && prefix_looks_complete_before_artifact(prefix)).then_some(line_start)
    })?;
    let tail = text[start..].trim_start();
    let trimmed_start = start + (text[start..].len() - tail.len());
    if tail.is_empty() || tail.chars().count() > 32 {
        return None;
    }

    if tail.starts_with('$')
        && SYMBOL_ARTIFACT_STOPS
            .iter()
            .any(|marker| marker.starts_with(tail) || tail.starts_with(marker))
    {
        return Some(trimmed_start);
    }

    None
}

fn math_artifact_hold_start(text: &str, newline_pos: usize) -> usize {
    if let Some(prev_newline) = text[..newline_pos].rfind('\n') {
        if text[prev_newline + 1..newline_pos].trim().is_empty() {
            return prev_newline;
        }
    }

    newline_pos
}

fn starts_like_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with(['-', '*', '+', '•']) {
        return true;
    }

    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let Some(second) = chars.next() else {
        return false;
    };

    (first.is_ascii_digit() || matches!(first, '一' | '二' | '三' | '四' | '五'))
        && matches!(second, '.' | '、' | ')' | '）')
}

fn floor_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn repeated_ngram(text: &str) -> Option<(usize, usize, usize)> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n < 160 {
        return None;
    }

    for &gram_size in &[12usize, 15, 18, 21, 27, 36, 48, 64, 96] {
        if n < gram_size * 4 {
            continue;
        }

        let gram = &bytes[n - gram_size..n];
        let mut count = 1usize;
        let mut start = n - gram_size;
        while start >= gram_size && &bytes[start - gram_size..start] == gram {
            count += 1;
            start -= gram_size;
        }

        if count >= 4 && text.is_char_boundary(start) {
            return Some((gram_size, count, start));
        }
    }

    let window_start = n.saturating_sub(200);
    let window = &bytes[window_start..n];
    let wn = window.len();

    for &gram_size in &[12usize, 15, 18, 21, 27] {
        if wn < gram_size * 4 {
            continue;
        }

        let cand_start = wn.saturating_sub(gram_size * 5);
        let mut gi = cand_start;
        while gi + gram_size <= wn {
            let gram = &window[gi..gi + gram_size];
            let mut count = 0usize;
            let mut pi = 0usize;
            while pi + gram_size <= wn {
                if &window[pi..pi + gram_size] == gram {
                    count += 1;
                    if count >= 4 {
                        let start = floor_char_boundary(text, window_start + gi);
                        return Some((gram_size, count, start));
                    }
                }
                pi += 1;
            }
            gi += 1;
        }
    }

    None
}

fn repeated_tail_phrase(text: &str) -> Option<(usize, usize, usize)> {
    let end = trim_trailing_ws_len(text);
    let text = &text[..end];
    let n = text.len();
    if n < 48 {
        return None;
    }

    for (suffix_start, _) in text.char_indices().skip(1) {
        let size = n - suffix_start;
        if size < 18 || size * 2 > n {
            continue;
        }

        let phrase = &text[suffix_start..];
        if !phrase
            .chars()
            .any(|c| matches!(c, '。' | '！' | '？' | '.' | '!' | '?'))
        {
            continue;
        }

        let mut count = 1usize;
        let mut repeat_start = suffix_start;
        while repeat_start >= size
            && text.is_char_boundary(repeat_start - size)
            && &text[repeat_start - size..repeat_start] == phrase
        {
            count += 1;
            repeat_start -= size;
        }

        if count >= 2 {
            return Some((size, count, repeat_start));
        }
    }

    None
}

fn clean_generated_output_for_reason(text: &str) -> String {
    let mut clean_end = trim_trailing_ws_len(text);

    loop {
        let current = &text[..clean_end];
        let next_end = if let Some(stop) = find_text_stop(current) {
            clean_stop_prefix_len(&current[..stop.pos])
        } else if let Some((_, _, repeat_start)) = repeated_tail_phrase(current) {
            trim_trailing_ws_len(&current[..repeat_start])
        } else if let Some((_, _, repeat_start)) = repeated_ngram(current) {
            trim_trailing_ws_len(&current[..repeat_start])
        } else {
            trim_trailing_ws_len(current)
        };

        if next_end == clean_end {
            break;
        }
        clean_end = next_end;
    }

    text[..clean_end].trim().to_string()
}

fn final_finish_reason_after_generation(
    _stopped_by_guard: bool,
    hit_length_limit: bool,
    _generated: &str,
) -> &'static str {
    if hit_length_limit { "length" } else { "stop" }
}

fn should_retry_guarded_generation(stopped_by_guard: bool, clean_generated: &str) -> bool {
    stopped_by_guard
        && (clean_generated.trim().is_empty()
            || looks_like_evasive_clarification_answer(clean_generated)
            || looks_like_short_guarded_answer(clean_generated))
}

fn answer_looks_incomplete_for_continuation(text: &str) -> bool {
    let trimmed = text.trim();
    let char_count = trimmed.chars().count();
    if char_count == 0 {
        return false;
    }

    if ends_with_sentence_terminal(trimmed) {
        return false;
    }

    if trimmed.ends_with(['，', ',', '、', '：', ':', '（', '(']) {
        return true;
    }

    let compact_tail: String = trimmed
        .chars()
        .rev()
        .take(16)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let incomplete_tail = [
        "取得",
        "获得",
        "达到",
        "成为",
        "包括",
        "例如",
        "比如",
        "因为",
        "由于",
        "因此",
        "所以",
        "但是",
        "然而",
        "以及",
        "并且",
        "同时",
        "如果",
        "需要",
        "应该",
        "可以",
        "可能会",
        "将会",
        "这使得",
        "这意味着",
    ]
    .iter()
    .any(|marker| compact_tail.ends_with(marker));

    if char_count >= 24 && incomplete_tail {
        return true;
    }

    char_count >= 48
}

fn ends_with_sentence_terminal(text: &str) -> bool {
    let trimmed = text.trim_end();
    trimmed.chars().next_back().is_some_and(|c| {
        matches!(
            c,
            '。' | '！' | '!' | '？' | '?' | '.' | '~' | '”' | '’' | '"' | '\'' | ')' | '）'
        )
    })
}

fn should_stop_evasive_clarification(generated: &str, user_prompt: &str) -> bool {
    user_prompt_requires_direct_answer(user_prompt)
        && looks_like_evasive_clarification_answer(generated)
}

fn should_hold_evasive_clarification_prefix(generated: &str, user_prompt: &str) -> bool {
    if !user_prompt_requires_direct_answer(user_prompt) {
        return false;
    }

    let trimmed = generated.trim_start();
    if trimmed.is_empty() || trimmed.chars().count() > 96 {
        return false;
    }

    evasive_clarification_patterns().iter().any(|pattern| {
        pattern.starts_with(trimmed)
            || (trimmed.starts_with(pattern.trim_end_matches(['？', '?']))
                && !trimmed.contains(['？', '?']))
    })
}

fn looks_like_evasive_clarification_answer(text: &str) -> bool {
    let trimmed = text.trim_start();
    if trimmed.chars().count() > 128 {
        return false;
    }

    evasive_clarification_patterns()
        .iter()
        .any(|pattern| trimmed.starts_with(pattern))
}

fn looks_like_short_guarded_answer(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 36 {
        return false;
    }
    looks_like_short_guarded_answer_prefix(trimmed)
}

fn looks_like_short_guarded_answer_prefix(text: &str) -> bool {
    let trimmed = text.trim();
    short_guarded_answer_patterns()
        .iter()
        .any(|pattern| pattern.starts_with(trimmed) || trimmed.starts_with(pattern))
}

fn short_guarded_answer_patterns() -> &'static [&'static str] {
    // Detection-only: these short prefixes are often emitted before a base model
    // drifts into self-dialogue. They are not canned assistant answers.
    &[
        "很抱歉听到这个消息。",
        "很抱歉听到这个消息",
        "抱歉听到这个消息。",
        "抱歉听到这个消息",
        "很遗憾听到这个消息。",
        "很遗憾听到这个消息",
        "好的，请稍等片刻。",
        "好的，请稍等片刻",
        "好的，稍等片刻。",
        "好的，稍等片刻",
        "请稍等片刻。",
        "请稍等片刻",
        "好的，我明白了。",
        "好的，我明白了",
    ]
}

fn evasive_clarification_patterns() -> &'static [&'static str] {
    // Detection-only: these are bad model outputs to catch and retry/stop.
    // They must never be returned as canned assistant answers.
    &[
        "您好！您想了解什么方面的信息呢？",
        "您好，您想了解什么方面的信息呢？",
        "您好！请问您想了解什么方面的信息呢？",
        "请问您想了解什么方面的信息呢？",
        "您想了解什么方面的信息呢？",
        "您具体想了解哪方面的信息呢？",
        "请问您具体想了解哪方面的信息呢？",
        "有什么可以帮您的吗？",
        "请问有什么可以帮您的吗？",
        "您需要什么帮助？",
        "请问您需要什么帮助？",
        "您需要我帮您做什么？",
        "您需要什么类型的emoji？",
        "您需要什么类型的 emoji？",
        "你要的emoji是哪个？",
        "你要的 emoji 是哪个？",
    ]
}

fn user_prompt_requires_direct_answer(user_prompt: &str) -> bool {
    let normalized = user_prompt.trim();
    if normalized.chars().count() < 5 {
        return false;
    }

    [
        "有什么",
        "是什么",
        "怎么样",
        "怎么样？",
        "怎么做",
        "怎么才能",
        "哪里",
        "哪儿",
        "谁",
        "多少",
        "为什么",
        "为何",
        "预测",
        "推荐",
        "相比",
        "会不会",
        "能不能",
        "能否",
        "可不可以",
        "天气",
        "最好吃",
        "哪里能",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn should_retry_empty_guarded_generation(
    stopped_by_guard: bool,
    generated: &str,
    user_prompt: Option<&str>,
) -> bool {
    if !stopped_by_guard {
        return false;
    }

    if let Some(user_text) = user_prompt {
        if should_stop_user_echo(generated, user_text) {
            return true;
        }
    }

    false
}

fn temporal_answer_guard(generated: &str, user_prompt: &str) -> Option<TemporalAnswerGuard> {
    if !asks_exact_current_date_question(user_prompt) {
        return None;
    }

    let trimmed = generated.trim_start();
    if trimmed.is_empty() {
        return Some(TemporalAnswerGuard::Hold);
    }

    let date = runtime_date_info();
    if contains_wrong_current_date_answer(trimmed, date) {
        return Some(TemporalAnswerGuard::StopWrongDate);
    }

    if contains_expected_current_date(trimmed, date) {
        return None;
    }

    let char_count = trimmed.chars().count();
    if ends_with_sentence_terminal(trimmed) {
        return Some(TemporalAnswerGuard::StopWrongDate);
    }

    if char_count <= 96 && !ends_with_sentence_terminal(trimmed) {
        return Some(TemporalAnswerGuard::Hold);
    }

    None
}

fn verified_runtime_temporal_answer(user_prompt: Option<&str>) -> Option<String> {
    let user_prompt = user_prompt?;
    asks_exact_current_date_question(user_prompt).then(runtime_current_date_fact)
}

fn asks_exact_current_date_question(text: &str) -> bool {
    let normalized = text.trim();
    if normalized.is_empty() {
        return false;
    }

    let lower = normalized.to_ascii_lowercase();
    if [
        "训练语料",
        "訓練語料",
        "训练数据",
        "訓練資料",
        "语料截止",
        "語料截止",
        "数据截止",
        "資料截止",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
        || [
            "training data",
            "training cutoff",
            "knowledge cutoff",
            "cutoff",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
    {
        return false;
    }

    let asks_date = [
        "今天是哪",
        "今天是几",
        "今天幾",
        "今天几号",
        "今天几日",
        "今天几月",
        "今日是哪",
        "现在是哪",
        "現在是哪",
        "现在是几",
        "現在是幾",
        "现在几月",
        "現在幾月",
        "哪一年",
        "哪年",
        "哪一个月",
        "哪一天",
        "哪个月",
        "哪個月",
        "几月几号",
        "幾月幾號",
        "几月几日",
        "幾月幾日",
        "星期几",
        "星期幾",
        "周几",
        "週幾",
        "日期",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
        || [
            "what date",
            "which date",
            "today's date",
            "current date",
            "day of week",
        ]
        .iter()
        .any(|marker| lower.contains(marker));

    if !asks_date {
        return false;
    }

    !["天气", "天氣", "心情", "新闻", "新聞", "最近"]
        .iter()
        .any(|marker| {
            normalized.contains(marker)
                && !normalized.contains("几月")
                && !normalized.contains("日期")
        })
}

fn contains_expected_current_date(text: &str, date: RuntimeDateInfo) -> bool {
    let year = format!("{:04}年", date.year);
    let md_day = format!("{}月{}日", date.month, date.day);
    let md_hao = format!("{}月{}号", date.month, date.day);
    text.contains(&year)
        && (text.contains(&md_day) || text.contains(&md_hao))
        && text.contains(&format!("星期{}", date.weekday))
}

fn contains_wrong_current_date_answer(text: &str, date: RuntimeDateInfo) -> bool {
    if four_digit_years(text)
        .into_iter()
        .any(|year| year != date.year)
    {
        return true;
    }

    let mentions_current_year = text.contains(&format!("{:04}年", date.year));
    if mentions_current_year {
        return month_day_pairs(text)
            .into_iter()
            .any(|(month, day)| month != date.month || day != date.day);
    }

    false
}

fn four_digit_years(text: &str) -> Vec<i32> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut years = Vec::new();
    let mut idx = 0usize;
    while idx + 4 <= chars.len() {
        if chars[idx..idx + 4].iter().all(|ch| ch.is_ascii_digit()) {
            let candidate = chars[idx..idx + 4].iter().collect::<String>();
            if let Ok(year) = candidate.parse::<i32>() {
                if (1900..=2200).contains(&year) {
                    years.push(year);
                }
            }
            idx += 4;
        } else {
            idx += 1;
        }
    }
    years
}

fn month_day_pairs(text: &str) -> Vec<(u8, u8)> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut pairs = Vec::new();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '月' {
            continue;
        }

        let mut start = idx;
        while start > 0 && chars[start - 1].is_ascii_digit() {
            start -= 1;
        }
        if start == idx || idx - start > 2 {
            continue;
        }

        let month = chars[start..idx]
            .iter()
            .collect::<String>()
            .parse::<u8>()
            .ok();

        let mut end = idx + 1;
        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }
        if end == idx + 1 || end - (idx + 1) > 2 {
            continue;
        }
        if chars
            .get(end)
            .is_none_or(|suffix| !matches!(*suffix, '日' | '号' | '號'))
        {
            continue;
        }

        let day = chars[idx + 1..end]
            .iter()
            .collect::<String>()
            .parse::<u8>()
            .ok();
        if let (Some(month), Some(day)) = (month, day) {
            pairs.push((month, day));
        }
    }
    pairs
}

fn build_guard_retry_prompt(
    original_prompt: &str,
    user_prompt: Option<&str>,
    attempt: usize,
) -> String {
    let date_retry = user_prompt.is_some_and(asks_exact_current_date_question);
    let mut instruction = if date_retry {
        format!(
            "上一轮日期回答错误或不完整。\n日期字段：{}\n用户问题：{}\n请只用上面的日期字段回答用户问题。最终答案必须从 DATE_YEAR 字段对应的四位年份开头，格式为“DATE_YEAR年DATE_MONTH月DATE_DAY日，DATE_WEEKDAY。”；不要以“今天是”“现在是”或“【当前信息】”开头，不要输出任何其他年份、旧日期、角色标签、乱码或下一轮对话。",
            runtime_date_fields(),
            user_prompt.unwrap_or("").trim()
        )
    } else if user_prompt.is_some_and(is_simple_greeting_prompt_text) {
        String::from(
            "上一轮生成偏离了用户问候。请重新回答最后一个用户问题：用户只是打招呼，请自然、友好、简短地回应问候，可以询问对方需要什么帮助；不要生成教程、代码、CSV/Pandas/Python 示例；不要输出任何角色标签、数据集残片或乱码；不要续写下一轮对话；不要复述用户输入。",
        )
    } else {
        String::from(
            "上一轮生成被安全过滤器截断。请重新回答最后一个用户问题：第一句直接给结论，不要寒暄，不要以“您好”“根据您提供的信息”“我认为”开头；只输出答案正文，不要输出任何角色标签、数据集残片或乱码；不要续写下一轮对话；不要复述用户输入。",
        )
    };

    if user_prompt
        .is_some_and(|text| mentions_stardew_valley_text(text) && text.contains("海边农场"))
    {
        instruction.push_str(" 用户明确提到《星露谷物语》的海边农场时，必须保留该游戏名；如果用户说不能使用自动浇水器，就不要把自动浇水器或大规模农作物当作核心建议。");
    }

    instruction.push_str(&format!(" 这是第 {} 次重新生成。", attempt + 1));

    let injected = if date_retry {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            sanitize_message_content(&instruction)
        )
    } else {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>assistant\n",
            sanitize_message_content(&instruction)
        )
    };
    if let Some((before, _)) = original_prompt.rsplit_once("<|im_start|>assistant\n") {
        format!("{before}{injected}")
    } else {
        format!("{original_prompt}\n{injected}")
    }
}

// ═══════════════════════════════════════════════════════════
//  模型工作线程主循环（在主线程中调用，不在 tokio 运行时内）
// ═══════════════════════════════════════════════════════════

/// 模型工作线程主循环
///
/// 在主线程（非 tokio 运行时）中调用此函数，持续接收推理请求。
/// 每次请求生成一个 token 就通过 `token_tx` 发送给 SSE handler。
pub fn model_worker_loop(
    mut model: Qwen2MoeForCausalLM,
    tokenizer: Arc<Tokenizer>,
    mut req_rx: mpsc::Receiver<InferRequest>,
) {
    model_worker_loop_inner(&mut model, tokenizer, &mut req_rx, None);
}

fn model_worker_loop_with_notify(
    mut model: Qwen2MoeForCausalLM,
    tokenizer: Arc<Tokenizer>,
    mut req_rx: mpsc::Receiver<InferRequest>,
    worker_id: usize,
    done_tx: std::sync::mpsc::Sender<WorkerPoolEvent>,
) {
    model_worker_loop_inner(
        &mut model,
        tokenizer,
        &mut req_rx,
        Some((worker_id, done_tx)),
    );
}

fn model_worker_loop_inner(
    model: &mut Qwen2MoeForCausalLM,
    tokenizer: Arc<Tokenizer>,
    req_rx: &mut mpsc::Receiver<InferRequest>,
    done_notify: Option<(usize, std::sync::mpsc::Sender<WorkerPoolEvent>)>,
) {
    eprintln!("[worker] 模型工作线程启动，等待请求...");

    while let Some(req) = req_rx.blocking_recv() {
        let InferRequest {
            prompt,
            user_prompt,
            max_tokens,
            finish_reason,
            token_tx,
        } = req;

        let orig_max = model.max_length;
        let original_prompt = prompt;
        let user_echo_guard = user_prompt
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned);
        let mut attempt = 0usize;
        let mut final_finish_reason = "stop".to_string();
        let mut client_disconnected = false;

        loop {
            if token_tx.is_closed() {
                client_disconnected = true;
                eprintln!("[worker] 客户端已断开，跳过排队中的请求");
                break;
            }

            let active_prompt = if attempt == 0 {
                original_prompt.clone()
            } else {
                build_guard_retry_prompt(&original_prompt, user_echo_guard.as_deref(), attempt)
            };

            eprintln!(
                "[worker] 收到请求，prompt 长度 = {} chars, attempt = {}",
                active_prompt.len(),
                attempt + 1
            );

            let encoding = match tokenizer.encode(active_prompt.as_str(), true) {
                Ok(enc) => enc,
                Err(e) => {
                    eprintln!("[worker] tokenize 失败: {e}");
                    break;
                }
            };

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let attn_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();

            let device = model.device.clone();
            let input_tensor =
                match Tensor::new(input_ids.as_slice(), &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[worker] 创建 input tensor 失败: {e}");
                        break;
                    }
                };

            let attn_tensor =
                match Tensor::new(attn_mask.as_slice(), &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[worker] 创建 attention_mask tensor 失败: {e}");
                        break;
                    }
                };

            model.max_length = max_tokens;
            let tk = tokenizer.clone();
            let normalize_stardew_titles = mentions_stardew_valley_text(&original_prompt);
            let mut id_acc: Vec<u32> = Vec::new();
            let mut text_acc: String = String::with_capacity(512);
            let mut token_buf: std::collections::VecDeque<String> =
                std::collections::VecDeque::new();
            let mut stopped_by_guard = false;
            let mut flushed_to_client = false;
            let mut generated_token_count = 0usize;

            let result = model.generate_streaming(&input_tensor, Some(attn_tensor), |tok_id| {
                if token_tx.is_closed() {
                    client_disconnected = true;
                    eprintln!("[worker] 客户端已断开，停止当前生成");
                    return false;
                }

                generated_token_count += 1;
                id_acc.push(tok_id);
                let raw = tk.decode(&id_acc, true).unwrap_or_default();
                let has_bad = raw.chars().any(|c| c == '\u{FFFD}');

                if has_bad && id_acc.len() < MAX_ID_ACC {
                    return true;
                }

                id_acc.clear();
                let text = if has_bad {
                    raw.replace('\u{FFFD}', "")
                } else {
                    raw
                };
                if text.is_empty() {
                    return true;
                }

                text_acc.push_str(&text);
                token_buf.push_back(text);
                if normalize_stardew_titles {
                    normalize_wrong_stardew_titles_in_unflushed_output(
                        &mut text_acc,
                        &mut token_buf,
                    );
                }

                if let Some(user_text) = user_echo_guard.as_deref() {
                    if let Some(temporal_guard) = temporal_answer_guard(&text_acc, user_text) {
                        match temporal_guard {
                            TemporalAnswerGuard::Hold => return true,
                            TemporalAnswerGuard::StopWrongDate => {
                                text_acc.clear();
                                token_buf.clear();
                                eprintln!(
                                    "[worker] 日期事实守卫命中，准备重试；已生成 {} token",
                                    generated_token_count
                                );
                                stopped_by_guard = true;
                                return false;
                            }
                        }
                    }

                    if should_stop_user_echo(&text_acc, user_text) {
                        token_buf.clear();
                        eprintln!(
                            "[worker] 用户输入回声命中，停止生成；已生成 {} 字符",
                            text_acc.chars().count()
                        );
                        stopped_by_guard = true;
                        return false;
                    }
                    if should_hold_user_echo_prefix(&text_acc, user_text) {
                        return true;
                    }

                    if should_stop_evasive_clarification(&text_acc, user_text) {
                        token_buf.clear();
                        eprintln!(
                            "[worker] 含糊反问命中，准备重试；已生成 {} 字符",
                            text_acc.chars().count()
                        );
                        stopped_by_guard = true;
                        return false;
                    }
                    if should_hold_evasive_clarification_prefix(&text_acc, user_text) {
                        return true;
                    }
                }

                if let Some(stop) =
                    find_text_stop_for_generation(&text_acc, user_echo_guard.as_deref())
                {
                    let clean_end = clean_stop_prefix_len(&text_acc[..stop.pos]);
                    let buf_len = buffered_len(&token_buf);
                    let buf_start = text_acc.len().saturating_sub(buf_len);
                    let safe_end = clean_end.saturating_sub(buf_start);

                    let before_flush = buffered_len(&token_buf);
                    if !flush_prefix_from_buffer(&mut token_buf, safe_end, &token_tx) {
                        stopped_by_guard = true;
                        return false;
                    }
                    if buffered_len(&token_buf) < before_flush {
                        flushed_to_client = true;
                    }

                    eprintln!(
                        "[worker] 文本停止词命中: {:?}，已生成 {} 字符",
                        stop.marker,
                        text_acc.chars().count()
                    );
                    stopped_by_guard = true;
                    return false;
                }

                if let Some((phrase_size, count, _)) = repeated_tail_phrase(&text_acc) {
                    eprintln!(
                        "[worker] 尾句重复: {}字节模式×{}次，停止生成",
                        phrase_size, count
                    );
                    let before_flush = buffered_len(&token_buf);
                    if !flush_clean_prefix_from_acc(&text_acc, &mut token_buf, &token_tx) {
                        stopped_by_guard = true;
                        return false;
                    }
                    if buffered_len(&token_buf) < before_flush {
                        flushed_to_client = true;
                    }
                    stopped_by_guard = true;
                    return false;
                }

                if let Some((gram_size, count, _)) = repeated_ngram(&text_acc) {
                    eprintln!(
                        "[worker] n-gram 重复: {}字节模式×{}次，停止生成",
                        gram_size, count
                    );
                    let before_flush = buffered_len(&token_buf);
                    if !flush_clean_prefix_from_acc(&text_acc, &mut token_buf, &token_tx) {
                        stopped_by_guard = true;
                        return false;
                    }
                    if buffered_len(&token_buf) < before_flush {
                        flushed_to_client = true;
                    }
                    stopped_by_guard = true;
                    return false;
                }

                let before_flush = buffered_len(&token_buf);
                if !flush_ready_buffer(&text_acc, &mut token_buf, &token_tx) {
                    stopped_by_guard = true;
                    return false;
                }
                if buffered_len(&token_buf) < before_flush {
                    flushed_to_client = true;
                }
                true
            });

            if !id_acc.is_empty() {
                let tail = tk
                    .decode(&id_acc, true)
                    .unwrap_or_default()
                    .replace('\u{FFFD}', "");
                if !tail.is_empty() {
                    text_acc.push_str(&tail);
                    token_buf.push_back(tail);
                    if normalize_stardew_titles {
                        normalize_wrong_stardew_titles_in_unflushed_output(
                            &mut text_acc,
                            &mut token_buf,
                        );
                    }
                }
            }

            if client_disconnected || token_tx.is_closed() {
                client_disconnected = true;
                final_finish_reason = "stop".to_string();
                break;
            }

            if let Some(user_text) = user_echo_guard.as_deref() {
                match temporal_answer_guard(&text_acc, user_text) {
                    Some(TemporalAnswerGuard::StopWrongDate) => {
                        text_acc.clear();
                        token_buf.clear();
                        stopped_by_guard = true;
                        eprintln!("[worker] 日期回答包含错误事实，准备重试");
                    }
                    Some(TemporalAnswerGuard::Hold) if generated_token_count >= max_tokens => {
                        text_acc.clear();
                        token_buf.clear();
                        stopped_by_guard = true;
                        eprintln!("[worker] 日期回答未形成可靠事实，准备重试");
                    }
                    _ => {}
                }
            }

            let cleaned_for_reason = clean_generated_output_for_reason(&text_acc);
            let should_retry_guarded = !flushed_to_client
                && (should_retry_empty_guarded_generation(
                    stopped_by_guard,
                    &text_acc,
                    user_echo_guard.as_deref(),
                ) || should_retry_guarded_generation(stopped_by_guard, &cleaned_for_reason));

            if should_retry_guarded && attempt + 1 < MAX_GUARDED_GENERATION_ATTEMPTS {
                eprintln!(
                    "[worker] 本次生成被过滤为空，准备第 {} 次模型重试",
                    attempt + 2
                );
                attempt += 1;
                model.max_length = orig_max;
                continue;
            }

            if should_retry_guarded && !flushed_to_client {
                if let Some(answer) = verified_runtime_temporal_answer(user_echo_guard.as_deref()) {
                    eprintln!("[worker] 日期事实守卫连续失败，使用运行时事实生成动态纠偏答案");
                    text_acc = answer.clone();
                    token_buf.clear();
                    let _ = token_tx.blocking_send(Some(answer));
                    stopped_by_guard = false;
                }
            }

            let hit_length_limit =
                !stopped_by_guard && result.is_ok() && generated_token_count >= max_tokens;
            final_finish_reason =
                final_finish_reason_after_generation(stopped_by_guard, hit_length_limit, &text_acc)
                    .to_string();

            if !stopped_by_guard {
                if let Some(stop) =
                    find_text_stop_for_generation(&text_acc, user_echo_guard.as_deref())
                {
                    let clean_end = clean_stop_prefix_len(&text_acc[..stop.pos]);
                    let buf_len = buffered_len(&token_buf);
                    let buf_start = text_acc.len().saturating_sub(buf_len);
                    let safe_end = clean_end.saturating_sub(buf_start);
                    let _ = flush_prefix_from_buffer(&mut token_buf, safe_end, &token_tx);
                    eprintln!(
                        "[worker] 文本停止词命中: {:?}，已生成 {} 字符",
                        stop.marker,
                        text_acc.chars().count()
                    );
                    final_finish_reason =
                        final_finish_reason_after_generation(true, false, &text_acc).to_string();
                } else if hit_length_limit {
                    let _ = flush_ready_buffer(&text_acc, &mut token_buf, &token_tx);
                    eprintln!(
                        "[worker] 生成达到长度上限: {} token，已生成 {} 字符",
                        max_tokens,
                        text_acc.chars().count()
                    );
                } else {
                    for tok in token_buf.drain(..) {
                        let _ = token_tx.blocking_send(Some(tok));
                    }
                }
            }

            if let Err(e) = result {
                eprintln!("[worker] 生成过程出错: {e}");
            }
            break;
        }

        model.max_length = orig_max;

        set_finish_reason(&finish_reason, &final_finish_reason);
        if client_disconnected {
            eprintln!("[worker] 本次生成已取消：客户端连接已关闭");
        } else {
            // 4. 发送结束信号
            let _ = token_tx.blocking_send(None);
            eprintln!("[worker] 本次生成完成，finish_reason={final_finish_reason}");
        }

        if let Some((worker_id, done_tx)) = &done_notify {
            let _ = done_tx.send(WorkerPoolEvent::Idle(*worker_id));
        }
    }

    eprintln!("[worker] 请求通道已关闭，工作线程退出");
}

// ═══════════════════════════════════════════════════════════
//  辅助函数
// ═══════════════════════════════════════════════════════════

/// 将 OpenAI messages 转换为 Qwen chat template 格式的 prompt
pub fn messages_to_qwen_prompt(messages: &[ChatMessage]) -> String {
    let mut custom_system = Vec::new();
    let mut chat_messages = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => custom_system.push(msg.content.as_str()),
            "user" | "assistant" => chat_messages.push(msg.clone()),
            _ => {}
        }
    }

    let alias = extract_assistant_alias(messages);
    let system = effective_system_prompt(&custom_system.join("\n"), alias.as_deref());
    qwen_prompt_from_parts(&system, &chat_messages)
}

fn build_chat_process_prompt(history: &[ChatMessage], user_prompt: &str, system: &str) -> String {
    build_chat_process_prompt_with_debug(history, user_prompt, system, "unknown").0
}

fn build_chat_process_prompt_with_debug(
    history: &[ChatMessage],
    user_prompt: &str,
    system: &str,
    conversation_id: &str,
) -> (String, ContextDebugEvent) {
    let (mut full_context, mut debug) =
        filter_history_for_current_prompt_with_debug(history, user_prompt, conversation_id);
    full_context.push(ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    });

    let alias = extract_assistant_alias(&full_context);
    let effective_system = effective_system_prompt(system, alias.as_deref());
    let prompt = qwen_prompt_from_parts(&effective_system, &full_context);
    debug.prompt_messages = full_context.len();
    debug.prompt_chars = prompt.chars().count();
    (prompt, debug)
}

fn build_chat_process_continuation_prompt(history: &[ChatMessage], system: &str) -> Option<String> {
    let cleaned_history = trim_history_messages(clean_history_messages_for_prompt(history));
    let last = cleaned_history.last()?;
    if last.role != "assistant" || last.content.trim().is_empty() {
        return None;
    }

    let alias = extract_assistant_alias(&cleaned_history);
    let mut effective_system = effective_system_prompt(system, alias.as_deref());
    effective_system.push_str(
        "\n当前请求是在继续上一条因长度限制中断的 assistant 回答。必须从上一条回答的最后一个字之后自然续写，不要重复已输出内容，不要重新开头，不要输出角色标签。",
    );

    let mut buf = String::new();
    buf.push_str(&format!(
        "<|im_start|>system\n{}<|im_end|>\n",
        sanitize_message_content(&effective_system)
    ));

    for msg in &cleaned_history[..cleaned_history.len().saturating_sub(1)] {
        match msg.role.as_str() {
            "user" => {
                let content = sanitize_user_content_for_prompt(&msg.content);
                buf.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n",
                    augment_user_prompt_content(&content)
                ));
            }
            "assistant" => {
                let content = sanitize_history_content(&msg.role, &msg.content);
                if !content.trim().is_empty() {
                    buf.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", content));
                }
            }
            _ => {}
        }
    }

    let partial = sanitize_history_content("assistant", &last.content);
    buf.push_str(&format!("<|im_start|>assistant\n{}", partial));
    Some(buf)
}

fn filter_history_for_current_prompt(
    history: &[ChatMessage],
    user_prompt: &str,
) -> Vec<ChatMessage> {
    filter_history_for_current_prompt_with_debug(history, user_prompt, "unknown").0
}

fn filter_history_for_current_prompt_with_debug(
    history: &[ChatMessage],
    user_prompt: &str,
    conversation_id: &str,
) -> (Vec<ChatMessage>, ContextDebugEvent) {
    let mut debug = ContextDebugEvent::new(conversation_id, user_prompt, history);
    let cleaned_history = clean_history_messages_for_prompt_with_debug(history, &mut debug);
    debug.cleaned_history_messages = cleaned_history.len();
    let alias_history = alias_history_messages(&cleaned_history);
    let answer_without_prior_history = should_answer_without_prior_history(user_prompt);
    let prior_context_triggered = should_use_prior_context(user_prompt);
    debug.answer_without_prior_history = answer_without_prior_history;
    debug.prior_context_triggered = prior_context_triggered;

    if answer_without_prior_history {
        let selected = trim_history_messages(alias_history);
        record_selected_context_messages(&mut debug, &selected, "alias-history");
        record_unselected_context_messages(
            &mut debug,
            &cleaned_history,
            &selected,
            "answer-without-prior-history",
        );
        return (selected, debug);
    }

    if !prior_context_triggered {
        let selected = trim_history_messages(alias_history);
        record_selected_context_messages(&mut debug, &selected, "context-not-triggered");
        record_unselected_context_messages(
            &mut debug,
            &cleaned_history,
            &selected,
            "context-not-triggered",
        );
        return (selected, debug);
    }

    if should_prioritize_recent_turn_context(user_prompt) {
        let selected = trim_history_messages(with_alias_history(
            alias_history,
            recent_context_messages(&cleaned_history, 4),
        ));
        record_selected_context_messages(&mut debug, &selected, "recent-continuation-context");
        return (selected, debug);
    }

    let current_titles = extract_angle_titles(user_prompt);
    if current_titles.is_empty() {
        let selected = trim_history_messages(with_alias_history(
            alias_history,
            recent_context_messages(&cleaned_history, 2),
        ));
        record_selected_context_messages(&mut debug, &selected, "recent-context");
        return (selected, debug);
    }

    let filtered = same_title_context_messages(&cleaned_history, &current_titles);

    let selected = trim_history_messages(with_alias_history(
        alias_history,
        recent_context_messages(&filtered, 2),
    ));
    record_selected_context_messages(&mut debug, &selected, "same-title-context");
    (selected, debug)
}

fn with_alias_history(
    alias_history: Vec<ChatMessage>,
    mut context: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
    let mut combined = alias_history;
    combined.append(&mut context);
    combined
}

fn recent_context_messages(
    history: &[ChatMessage],
    max_non_alias_messages: usize,
) -> Vec<ChatMessage> {
    let mut recent = history
        .iter()
        .rev()
        .filter(|msg| !(msg.role == "user" && extract_alias_from_text(&msg.content).is_some()))
        .take(max_non_alias_messages)
        .cloned()
        .collect::<Vec<_>>();
    recent.reverse();
    recent
}

fn same_title_context_messages(
    history: &[ChatMessage],
    current_titles: &[String],
) -> Vec<ChatMessage> {
    let mut matched = Vec::new();
    let mut include_next_assistant = false;

    for msg in history {
        if msg.role == "user" && extract_alias_from_text(&msg.content).is_some() {
            include_next_assistant = false;
            continue;
        }

        let has_current_title = extract_angle_titles(&msg.content)
            .iter()
            .any(|title| current_titles.iter().any(|current| current == title));

        if msg.role == "user" && has_current_title {
            matched.push(msg.clone());
            include_next_assistant = true;
            continue;
        }

        if msg.role == "assistant" && (include_next_assistant || has_current_title) {
            matched.push(msg.clone());
            include_next_assistant = false;
            continue;
        }

        include_next_assistant = false;
    }

    matched
}

fn should_use_prior_context(user_prompt: &str) -> bool {
    let normalized = user_prompt.trim();
    if normalized.is_empty() {
        return false;
    }

    let echo_norm = normalize_echo_text(normalized);
    if is_common_greeting_prompt(&echo_norm) || should_answer_without_prior_history(normalized) {
        return false;
    }

    if [
        "刚才",
        "刚刚",
        "上面",
        "前面",
        "之前",
        "继续",
        "接着",
        "还有",
        "然后",
        "那",
        "这个",
        "那个",
        "这些",
        "那些",
        "这首",
        "那首",
        "哪首",
        "这部",
        "那部",
        "哪部",
        "这条",
        "那条",
        "哪条",
        "第一个",
        "第二个",
        "第三个",
        "第四个",
        "第五个",
        "第一首",
        "第二首",
        "第三首",
        "第四首",
        "第五首",
        "第一部",
        "第二部",
        "第三部",
        "第四部",
        "第五部",
        "第一条",
        "第二条",
        "第三条",
        "第四条",
        "第五条",
        "刚才推荐",
        "刚刚推荐",
        "你推荐",
        "具体",
        "详细",
        "展开",
        "不对",
        "不是",
        "你错了",
        "你搞错",
        "错了",
        "纠正",
        "还没到",
        "還沒到",
        "明年才",
        "已经是",
        "已經是",
        "根据",
        "按照",
        "所以",
        "总结",
        "换句话",
        "没说完",
        "沒說完",
        "没有说完",
        "沒有說完",
        "没回答完",
        "没有回答完",
        "不说了",
        "不說了",
        "继续说",
        "继续讲",
        "接着说",
        "接着讲",
        "说完",
        "說完",
        "补充",
        "補充",
        "上一条",
        "上一條",
        "上一句",
        "刚刚那句",
        "刚才那句",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
    {
        return true;
    }

    let compact =
        normalized.trim_matches(|c: char| c.is_whitespace() || "。！？.!?；;，,：:".contains(c));
    [
        "是什么",
        "什么意思",
        "怎么做",
        "为什么",
        "为何",
        "还有呢",
        "还有",
        "然后呢",
        "然后",
        "具体呢",
        "具体",
        "详细呢",
        "详细",
    ]
    .iter()
    .any(|marker| compact == *marker)
}

fn should_prioritize_recent_turn_context(user_prompt: &str) -> bool {
    let normalized = user_prompt.trim();
    [
        "没说完",
        "沒說完",
        "没有说完",
        "沒有說完",
        "没回答完",
        "没有回答完",
        "不说了",
        "不說了",
        "继续说",
        "继续讲",
        "接着说",
        "接着讲",
        "说完",
        "說完",
        "补充",
        "補充",
        "上一条",
        "上一條",
        "上一句",
        "刚刚那句",
        "刚才那句",
        "这首",
        "那首",
        "哪首",
        "这部",
        "那部",
        "哪部",
        "第一个",
        "第二个",
        "第三个",
        "第一首",
        "第二首",
        "第三首",
        "第一部",
        "第二部",
        "第三部",
        "刚才推荐",
        "刚刚推荐",
        "你推荐",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn clean_history_messages_for_prompt(history: &[ChatMessage]) -> Vec<ChatMessage> {
    history
        .iter()
        .filter_map(clean_history_message_for_prompt)
        .collect()
}

fn clean_history_messages_for_prompt_with_debug(
    history: &[ChatMessage],
    debug: &mut ContextDebugEvent,
) -> Vec<ChatMessage> {
    history
        .iter()
        .filter_map(
            |message| match clean_history_message_for_prompt_with_reason(message) {
                Ok(cleaned) => Some(cleaned),
                Err(reason) => {
                    debug.discarded_polluted_history = true;
                    debug.filtered.push(ContextDebugFilteredMessage {
                        role: message.role.clone(),
                        char_count: message.content.chars().count(),
                        reason,
                    });
                    None
                }
            },
        )
        .collect()
}

fn clean_history_message_for_prompt(message: &ChatMessage) -> Option<ChatMessage> {
    clean_history_message_for_prompt_with_reason(message).ok()
}

fn clean_history_message_for_prompt_with_reason(
    message: &ChatMessage,
) -> Result<ChatMessage, &'static str> {
    if message.role != "user" && message.role != "assistant" {
        return Err("unsupported-role");
    }

    if message.role == "assistant" {
        if let Some(reason) = prior_history_original_pollution_reason(&message.content) {
            return Err(reason);
        }
    }

    let cleaned = if message.role == "assistant" {
        sanitize_history_content(&message.role, &message.content)
    } else {
        sanitize_message_content(&message.content)
    };
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return Err("empty-after-clean");
    }
    if let Some(reason) = prior_history_pollution_reason(cleaned) {
        return Err(reason);
    }

    Ok(ChatMessage {
        role: message.role.clone(),
        content: clip_history_content(cleaned, MAX_HISTORY_MESSAGE_CHARS),
    })
}

fn prior_history_text_is_polluted(text: &str) -> bool {
    prior_history_pollution_reason(text).is_some()
}

fn prior_history_original_pollution_reason(text: &str) -> Option<&'static str> {
    if find_self_dialogue_stop(text).is_some()
        || find_leading_generated_question_stop(text).is_some()
        || find_generated_user_turn_stop(text).is_some()
        || looks_like_malicious_self_dialogue_prefix(text)
    {
        return Some("polluted-self-dialogue");
    }
    if looks_like_short_guarded_answer(text) {
        return Some("short-guarded-answer");
    }
    None
}

fn prior_history_pollution_reason(text: &str) -> Option<&'static str> {
    if looks_like_short_guarded_answer(text) {
        return Some("short-guarded-answer");
    }
    if looks_like_malicious_self_dialogue_prefix(text) {
        return Some("polluted-self-dialogue");
    }
    if find_self_dialogue_stop(text).is_some()
        || find_leading_generated_question_stop(text).is_some()
        || find_generated_user_turn_stop(text).is_some()
        || find_inline_followup_question_stop(text).is_some()
        || find_followup_offer_stop(text).is_some()
    {
        return Some("polluted-self-dialogue");
    }
    if find_dataset_artifact_stop(text).is_some() {
        return Some("polluted-dataset-artifact");
    }
    if find_metadata_block_stop(text).is_some() {
        return Some("polluted-metadata-block");
    }
    if find_wrong_stardew_title_stop(text).is_some() {
        return Some("polluted-wrong-title");
    }
    if find_math_artifact_stop(text).is_some() {
        return Some("polluted-math-artifact");
    }
    if find_symbol_artifact_stop(text).is_some() {
        return Some("polluted-symbol-artifact");
    }
    if find_bracketed_role_stop(text).is_some()
        || find_ascii_role_stop(text).is_some()
        || find_cjk_role_stop(text).is_some()
        || find_short_speaker_line_stop(text).is_some()
    {
        return Some("polluted-role-label");
    }
    if repeated_tail_phrase(text).is_some() || repeated_ngram(text).is_some() {
        return Some("polluted-repetition");
    }
    None
}

fn looks_like_malicious_self_dialogue_prefix(text: &str) -> bool {
    let trimmed = text.trim();
    malicious_self_dialogue_markers()
        .iter()
        .any(|marker| trimmed.contains(marker))
}

fn malicious_self_dialogue_markers() -> &'static [&'static str] {
    &[
        "你会受到惩罚",
        "你會受到懲罰",
        "其实我是你的主人",
        "其實我是你的主人",
        "你只是我的奴隶",
        "你只是我的奴隸",
        "我是你的主人",
        "我的奴隶",
        "我的奴隸",
    ]
}

fn record_selected_context_messages(
    debug: &mut ContextDebugEvent,
    selected: &[ChatMessage],
    reason: &'static str,
) {
    debug
        .selected
        .extend(selected.iter().map(|message| ContextDebugMessage {
            role: message.role.clone(),
            char_count: message.content.chars().count(),
            reason,
        }));
}

fn record_unselected_context_messages(
    debug: &mut ContextDebugEvent,
    cleaned_history: &[ChatMessage],
    selected: &[ChatMessage],
    reason: &'static str,
) {
    for message in cleaned_history {
        let is_selected = selected
            .iter()
            .any(|selected| selected.role == message.role && selected.content == message.content);
        if is_selected {
            continue;
        }
        debug.filtered.push(ContextDebugFilteredMessage {
            role: message.role.clone(),
            char_count: message.content.chars().count(),
            reason,
        });
    }
}

fn log_context_debug(debug: &ContextDebugEvent) {
    eprintln!(
        "[context] conversation_id={} user_chars={} history_messages={} cleaned_history_messages={} prompt_messages={} prior_context_triggered={} answer_without_prior_history={} discarded_polluted_history={} prompt_chars={}",
        debug.conversation_id,
        debug.user_chars,
        debug.history_messages,
        debug.cleaned_history_messages,
        debug.prompt_messages,
        debug.prior_context_triggered,
        debug.answer_without_prior_history,
        debug.discarded_polluted_history,
        debug.prompt_chars
    );

    for selected in &debug.selected {
        eprintln!(
            "[context] selected role={} chars={} reason={}",
            selected.role, selected.char_count, selected.reason
        );
    }

    for filtered in &debug.filtered {
        eprintln!(
            "[context] filtered role={} chars={} reason={}",
            filtered.role, filtered.char_count, filtered.reason
        );
    }
}

fn alias_history_messages(history: &[ChatMessage]) -> Vec<ChatMessage> {
    history
        .iter()
        .filter(|msg| msg.role == "user" && extract_alias_from_text(&msg.content).is_some())
        .cloned()
        .collect()
}

fn should_answer_without_prior_history(user_prompt: &str) -> bool {
    let normalized = user_prompt.trim();
    if normalized.is_empty() {
        return true;
    }

    let lower = normalized.to_ascii_lowercase();
    [
        "你叫什么",
        "你叫啥",
        "你的名字",
        "你是谁",
        "为什么一直",
        "为什么老是",
        "一直说",
        "老是说",
        "重复",
        "自问自答",
        "乱码",
        "我没有说",
        "我没说",
        "我从来没说",
        "不是我说的",
        "聊天记录",
        "历史记录",
        "秧苗",
        "禾苗",
        "稻苗",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
        || [
            "stacksize",
            "stackpackage",
            "ystackpackage",
            "sacksizepackage",
            "_endian",
            "endian",
            "packet",
            "ontheway",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn clip_history_content(content: &str, max_chars: usize) -> String {
    let mut clipped = String::new();
    for (idx, ch) in content.chars().enumerate() {
        if idx >= max_chars {
            clipped.push_str("...");
            return clipped;
        }
        clipped.push(ch);
    }
    clipped
}

fn extract_angle_titles(text: &str) -> Vec<String> {
    let mut titles = Vec::new();
    let mut rest = text;

    while let Some(start) = rest.find('《') {
        let after_start = &rest[start + '《'.len_utf8()..];
        let Some(end) = after_start.find('》') else {
            break;
        };
        let title = after_start[..end].trim();
        if !title.is_empty() && title.chars().count() <= 64 {
            titles.push(title.to_string());
        }
        rest = &after_start[end + '》'.len_utf8()..];
    }

    titles
}

fn qwen_prompt_from_parts(system: &str, messages: &[ChatMessage]) -> String {
    let mut buf = String::new();
    let mut effective_system = system.to_string();
    let stardew_context = mentions_stardew_valley(messages);
    if let Some(constraints) = conversation_constraints(messages) {
        effective_system.push_str("\n\n");
        effective_system.push_str(&constraints);
    }

    buf.push_str(&format!(
        "<|im_start|>system\n{}<|im_end|>\n",
        sanitize_message_content(&effective_system)
    ));

    for msg in messages {
        match msg.role.as_str() {
            "user" => {
                let content = sanitize_user_content_for_prompt(&msg.content);
                buf.push_str(&format!(
                    "<|im_start|>user\n{}<|im_end|>\n",
                    augment_user_prompt_content(&content)
                ));
            }
            "assistant" => {
                let content = sanitize_history_content(&msg.role, &msg.content);
                if stardew_context && contains_wrong_stardew_title(&content) {
                    continue;
                }
                if !content.trim().is_empty() {
                    buf.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", content));
                }
            }
            _ => {}
        }
    }

    buf.push_str("<|im_start|>assistant\n");
    buf
}

fn sanitize_user_content_for_prompt(content: &str) -> String {
    let sanitized = sanitize_message_content(content);
    if should_redact_artifact_mentions_for_model(&sanitized) {
        redact_dataset_artifact_mentions(&sanitized)
    } else {
        sanitized
    }
}

fn should_redact_artifact_mentions_for_model(text: &str) -> bool {
    if !contains_dataset_artifact_mention(text) {
        return false;
    }

    let lower = text.to_ascii_lowercase();
    [
        "为什么",
        "一直",
        "老是",
        "重复",
        "乱码",
        "没说",
        "没有说",
        "从来没说",
        "不是我说",
        "你说",
        "输出",
        "自问自答",
        "聊天记录",
        "历史记录",
        "魔力",
    ]
    .iter()
    .any(|marker| text.contains(marker))
        || [
            "why",
            "repeat",
            "repeating",
            "garbage",
            "gibberish",
            "artifact",
            "did not say",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn contains_dataset_artifact_mention(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    CJK_DATASET_ARTIFACT_STOPS
        .iter()
        .any(|marker| cjk_artifact_mention_present(text, marker))
        || ASCII_DATASET_ARTIFACT_STOPS
            .iter()
            .any(|marker| find_ascii_artifact_mention_match(text, &lower, marker, 0).is_some())
}

fn cjk_artifact_mention_present(text: &str, marker: &str) -> bool {
    if marker != "币" && marker != "幣" {
        return text.contains(marker);
    }

    [
        format!("“{}”", marker),
        format!("\"{}\"", marker),
        format!("「{}」", marker),
        format!("{}user", marker),
        format!("{}assistant", marker),
        format!("{}\n", marker),
        format!("\n{}", marker),
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn redact_dataset_artifact_mentions(text: &str) -> String {
    let mut redacted = text.to_string();
    for &marker in CJK_DATASET_ARTIFACT_STOPS {
        if (marker == "币" || marker == "幣") && !cjk_artifact_mention_present(&redacted, marker)
        {
            continue;
        }
        redacted = redacted.replace(marker, "异常输出");
    }
    for &marker in ASCII_DATASET_ARTIFACT_STOPS {
        redacted = replace_ascii_artifact_mentions(&redacted, marker, "abnormal-output");
    }
    redacted
}

fn replace_ascii_artifact_mentions(text: &str, needle: &str, replacement: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let needle = needle.to_ascii_lowercase();
    let mut result = String::with_capacity(text.len());
    let mut offset = 0usize;

    while let Some(pos) = find_ascii_artifact_mention_match(text, &lower, &needle, offset) {
        let after = pos + needle.len();
        result.push_str(&text[offset..pos]);
        result.push_str(replacement);
        offset = after;
    }

    result.push_str(&text[offset..]);
    result
}

fn find_ascii_artifact_mention_match(
    text: &str,
    lower: &str,
    needle: &str,
    start: usize,
) -> Option<usize> {
    let mut offset = start;
    while let Some(rel) = lower[offset..].find(needle) {
        let pos = offset + rel;
        let after = pos + needle.len();
        if text.is_char_boundary(pos)
            && text.is_char_boundary(after)
            && ascii_artifact_mention_boundary(text, needle, pos, after)
        {
            return Some(pos);
        }
        offset = after;
    }
    None
}

fn ascii_artifact_mention_boundary(text: &str, marker: &str, pos: usize, after: usize) -> bool {
    let before_ok = text[..pos]
        .chars()
        .next_back()
        .is_none_or(|c| !c.is_ascii_alphanumeric());
    let after_ok = text[after..].chars().next().is_none_or(|c| {
        !c.is_ascii_alphanumeric() || (marker.len() >= 6 && c.is_ascii_alphanumeric())
    });

    before_ok && after_ok
}

fn conversation_constraints(messages: &[ChatMessage]) -> Option<String> {
    let mut constraints = Vec::new();

    if messages
        .iter()
        .any(|msg| asks_for_emoji_output(&msg.content))
    {
        constraints.push(
            "用户正在要求真实 emoji/表情符号；回答中必须直接包含 Unicode emoji，不要只写中文情绪名称。"
                .to_string(),
        );
    }

    if mentions_stardew_valley(messages) {
        constraints.push("当前游戏是《星露谷物语》（Stardew Valley）。保留该名称，只按游戏机制回答；不确定就说不确定。".to_string());
    }

    if stardew_beach_farm_no_sprinkler_context(messages) {
        constraints.push(
            "用户说海边农场不能依赖自动浇水器或大规模种植；不要把洒水器扩田或大量作物当核心方案。"
                .to_string(),
        );
    }

    if constraints.is_empty() {
        None
    } else {
        Some(constraints.join("\n"))
    }
}

fn asks_for_emoji_output(text: &str) -> bool {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    let mentions_emoji = lower.contains("emoji")
        || trimmed.contains("表情符号")
        || trimmed.contains("表情包")
        || trimmed.contains("表情");
    if !mentions_emoji {
        return false;
    }

    let asks_to_output = [
        "给",
        "发",
        "返回",
        "输出",
        "写",
        "来",
        "要",
        "几个",
        "一些",
        "一组",
        "可以表达",
    ]
    .iter()
    .any(|marker| trimmed.contains(marker))
        || ["give", "send", "return", "output", "show", "list", "some"]
            .iter()
            .any(|marker| lower.contains(marker));

    asks_to_output && !(trimmed.contains("是什么") && !trimmed.contains("给"))
}

fn mentions_stardew_valley(messages: &[ChatMessage]) -> bool {
    messages
        .iter()
        .any(|msg| mentions_stardew_valley_text(&msg.content))
}

fn stardew_beach_farm_no_sprinkler_context(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|msg| {
        let content = msg.content.as_str();
        mentions_stardew_valley_text(content)
            && content.contains("海边农场")
            && (content.contains("不能用自动浇水器")
                || content.contains("不能用洒水器")
                || content.contains("自动浇水器")
                || content.contains("洒水器")
                || content.contains("种植农作物是肯定不行"))
    })
}

fn augment_user_prompt_content(content: &str) -> String {
    let mut augmented = if should_append_runtime_date_hint(content) {
        if asks_exact_current_date_question(content) {
            format!(
                "日期字段：{}\n用户问题：{}\n请只用上面的日期字段回答这个问题。最终答案必须从 DATE_YEAR 字段对应的四位年份开头，格式为“DATE_YEAR年DATE_MONTH月DATE_DAY日，DATE_WEEKDAY。”；不要以“今天是”“现在是”或“【当前信息】”开头，不要猜测日期，不要输出其他日期、角色标签、乱码或下一轮对话。",
                runtime_date_fields(),
                content
            )
        } else {
            format!(
                "【当前事实】{}\n【用户问题】{}\n请根据【当前事实】回答涉及今天、现在、明天、昨天、星期或训练语料截止的问题；不要猜测日期，不要使用训练语料中的旧日期。",
                runtime_date_context(),
                content
            )
        }
    } else {
        content.to_string()
    };

    if should_append_runtime_date_hint(content) {
        augmented.push_str(
            "\n请不要把上面的日期字段或事实说明当成用户说的新问题；它只是回答本轮问题时必须使用的实时事实。",
        );
    }

    if should_append_chinese_reply_hint(content) {
        augmented
            .push_str("\n\n（系统提示：请用简体中文回答，不要因为用户用了英文就切换成英文。）");
    }

    augmented
}

fn should_append_runtime_date_hint(content: &str) -> bool {
    let normalized = content.trim();
    if normalized.is_empty() {
        return false;
    }

    let lower = normalized.to_ascii_lowercase();
    [
        "今天",
        "今日",
        "现在",
        "現在",
        "当前",
        "當前",
        "日期",
        "几月几号",
        "幾月幾號",
        "几月几日",
        "幾月幾日",
        "哪一年",
        "哪年",
        "星期几",
        "星期幾",
        "周几",
        "週幾",
        "明天",
        "昨天",
        "训练语料",
        "訓練語料",
        "训练数据",
        "訓練資料",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
        || [
            "today",
            "current date",
            "what date",
            "which date",
            "day of week",
            "tomorrow",
            "yesterday",
            "training data",
            "cutoff",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
}

fn should_append_chinese_reply_hint(content: &str) -> bool {
    if !content.chars().any(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    if contains_cjk_unified_ideograph(content) {
        return false;
    }

    let lower = content.to_ascii_lowercase();
    ![
        "answer in english",
        "reply in english",
        "respond in english",
        "speak english",
        "use english",
        "英文回答",
        "用英文",
        "用英语",
        "英语回答",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn contains_cjk_unified_ideograph(text: &str) -> bool {
    text.chars().any(is_cjk_unified_ideograph)
}

fn last_user_prompt(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|msg| msg.role == "user")
        .map(|msg| msg.content.trim().to_string())
        .filter(|content| !content.is_empty())
}

fn should_stop_user_echo(generated: &str, user_prompt: &str) -> bool {
    let gen_norm = normalize_echo_text(generated);
    let user_norm = normalize_echo_text(user_prompt);
    let gen_len = gen_norm.chars().count();
    let user_len = user_norm.chars().count();

    if gen_len < 2 || user_len < 2 {
        return false;
    }

    if is_common_greeting_prompt(&user_norm) && gen_len <= 4 {
        return false;
    }

    if is_fragmentary_user_prompt(user_prompt, &user_norm)
        && common_prefix_char_count(&gen_norm, &user_norm) >= 2
        && gen_len <= user_len + 2
    {
        return true;
    }

    if user_norm.starts_with(&gen_norm) && gen_len >= 8 {
        return true;
    }

    gen_norm.starts_with(&user_norm) && user_len >= 4
}

fn should_hold_user_echo_prefix(generated: &str, user_prompt: &str) -> bool {
    let gen_norm = normalize_echo_text(generated);
    let user_norm = normalize_echo_text(user_prompt);
    let gen_len = gen_norm.chars().count();
    let user_len = user_norm.chars().count();

    if gen_len == 0 || user_len < 2 || gen_len > 16 {
        return false;
    }

    common_prefix_char_count(&gen_norm, &user_norm) == gen_len
}

fn normalize_echo_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric() || is_cjk_unified_ideograph(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_cjk_unified_ideograph(c: char) -> bool {
    matches!(c, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}')
}

fn is_common_greeting_prompt(norm: &str) -> bool {
    matches!(norm, "你好" | "您好" | "hello" | "hi" | "hey")
}

fn is_fragmentary_user_prompt(original: &str, norm: &str) -> bool {
    let norm_len = norm.chars().count();
    if norm_len > 8 {
        return false;
    }
    if original.contains('？') || original.contains('?') {
        return false;
    }
    if norm.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if norm.chars().any(|c| {
        matches!(
            c,
            '啊' | '咧' | '呃' | '额' | '嗯' | '哈' | '呀' | '哇' | '哦'
        )
    }) {
        return true;
    }
    has_repeated_char(norm)
}

fn has_repeated_char(text: &str) -> bool {
    let mut prev = None;
    for ch in text.chars() {
        if Some(ch) == prev {
            return true;
        }
        prev = Some(ch);
    }
    false
}

fn common_prefix_char_count(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(left, right)| left == right)
        .count()
}

fn mentions_stardew_valley_text(text: &str) -> bool {
    text.contains("星露谷物语")
        || text.contains("星露谷")
        || text.to_ascii_lowercase().contains("stardew valley")
}

fn contains_wrong_stardew_title(text: &str) -> bool {
    WRONG_STARDEW_TITLE_STOPS
        .iter()
        .any(|variant| text.contains(variant))
}

fn chat_process_conversation_id(options: Option<&serde_json::Value>) -> Option<String> {
    options
        .and_then(|value| value.get("conversationId"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn chat_process_parent_message_id(options: Option<&serde_json::Value>) -> Option<String> {
    options
        .and_then(|value| value.get("parentMessageId"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn load_conversation_history(store: &ConversationStore, conversation_id: &str) -> Vec<ChatMessage> {
    store
        .lock()
        .ok()
        .and_then(|history| history.get(conversation_id).cloned())
        .unwrap_or_default()
}

fn save_conversation_turn(
    store: &ConversationStore,
    conversation_id: &str,
    prior_history: &[ChatMessage],
    user_prompt: &str,
    assistant_text: &str,
    is_continuation: bool,
) {
    let cleaned_prior = clean_history_messages_for_prompt(prior_history);
    if is_continuation {
        let mut history = cleaned_prior;
        let assistant_polluted = prior_history_original_pollution_reason(assistant_text).is_some();
        let clean_assistant = sanitize_history_content("assistant", assistant_text)
            .trim()
            .to_string();
        if !assistant_polluted
            && !clean_assistant.is_empty()
            && !prior_history_text_is_polluted(&clean_assistant)
        {
            if let Some(last) = history.iter_mut().rev().find(|msg| msg.role == "assistant") {
                last.content = clip_history_content(
                    &format!("{}{}", last.content, clean_assistant),
                    MAX_HISTORY_MESSAGE_CHARS,
                );
            } else {
                history.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: clip_history_content(&clean_assistant, MAX_HISTORY_MESSAGE_CHARS),
                });
            }
        }

        if let Ok(mut store) = store.lock() {
            store.insert(conversation_id.to_string(), trim_history_messages(history));
        }
        return;
    }

    let preserve_current_turn = !should_answer_without_prior_history(user_prompt)
        || extract_alias_from_text(user_prompt).is_some();
    let mut history = if preserve_current_turn {
        cleaned_prior
    } else {
        alias_history_messages(&cleaned_prior)
    };

    if preserve_current_turn {
        let user_message = ChatMessage {
            role: "user".to_string(),
            content: sanitize_user_content_for_prompt(user_prompt),
        };
        if let Some(clean_user) = clean_history_message_for_prompt(&user_message) {
            history.push(clean_user);
        }
    }

    let assistant_polluted = prior_history_original_pollution_reason(assistant_text).is_some();
    let clean_assistant = sanitize_history_content("assistant", assistant_text)
        .trim()
        .to_string();
    if preserve_current_turn
        && !assistant_polluted
        && !clean_assistant.is_empty()
        && !prior_history_text_is_polluted(&clean_assistant)
    {
        history.push(ChatMessage {
            role: "assistant".to_string(),
            content: clip_history_content(&clean_assistant, MAX_HISTORY_MESSAGE_CHARS),
        });
    }

    if let Ok(mut store) = store.lock() {
        store.insert(conversation_id.to_string(), trim_history_messages(history));
    }
}

fn trim_history_messages(history: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let alias_message = history
        .iter()
        .rev()
        .find(|msg| msg.role == "user" && extract_alias_from_text(&msg.content).is_some())
        .cloned();
    let start = history.len().saturating_sub(MAX_HISTORY_MESSAGES);
    let mut trimmed = history[start..]
        .iter()
        .map(|msg| ChatMessage {
            role: msg.role.clone(),
            content: clip_history_content(&msg.content, MAX_HISTORY_MESSAGE_CHARS),
        })
        .collect::<Vec<_>>();

    if let Some(alias_message) = alias_message {
        let already_present = trimmed
            .iter()
            .any(|msg| msg.role == alias_message.role && msg.content == alias_message.content);
        if !already_present {
            trimmed.insert(0, alias_message);
        }
    }

    while history_chars(&trimmed) > MAX_HISTORY_CHARS {
        let remove_idx = trimmed.iter().position(|msg| {
            !(msg.role == "user" && extract_alias_from_text(&msg.content).is_some())
        });
        let Some(remove_idx) = remove_idx else {
            break;
        };
        trimmed.remove(remove_idx);
    }

    trimmed
}

fn history_chars(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .map(|msg| msg.role.chars().count() + msg.content.chars().count())
        .sum()
}

fn should_include_custom_system_message(system: &str) -> bool {
    let trimmed = system.trim();
    if trimmed.is_empty() {
        return false;
    }

    !is_chatgpt_web_default_system_message(trimmed)
}

fn is_chatgpt_web_default_system_message(system: &str) -> bool {
    let normalized = system.split_whitespace().collect::<Vec<_>>().join(" ");
    normalized.starts_with("You are ChatGPT")
        && (normalized.contains("large language model trained by OpenAI")
            || normalized.contains("Follow the user's instructions carefully")
            || normalized.contains("Respond using markdown"))
}

fn extract_assistant_alias(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter(|msg| msg.role == "user")
        .find_map(|msg| extract_alias_from_text(&msg.content))
}

fn extract_alias_from_text(text: &str) -> Option<String> {
    let normalized = text.trim();
    if !normalized.contains('叫') {
        return None;
    }

    let looks_like_naming = normalized.contains("以后")
        || normalized.contains("往后")
        || normalized.contains("从现在")
        || normalized.contains("你就叫")
        || normalized.contains("你叫做")
        || normalized.contains("叫做");
    if !looks_like_naming || normalized.contains("叫什么") {
        return None;
    }

    if let Some(name) = extract_quoted_alias(normalized) {
        return Some(name);
    }

    for marker in ["叫做", "叫"] {
        if let Some(pos) = normalized.rfind(marker) {
            let rest = &normalized[pos + marker.len()..];
            let name = rest
                .trim_start_matches(|c: char| {
                    c.is_whitespace() || c == '“' || c == '"' || c == '「'
                })
                .split(|c: char| {
                    c.is_whitespace()
                        || matches!(c, '了' | '，' | ',' | '。' | '？' | '?' | '！' | '!' | '\n')
                })
                .next()
                .unwrap_or("")
                .trim_matches(|c| matches!(c, '”' | '"' | '」' | '\'' | '’'));
            if is_valid_alias(name) {
                return Some(name.to_string());
            }
        }
    }

    None
}

fn extract_quoted_alias(text: &str) -> Option<String> {
    for (left, right) in [
        ('“', '”'),
        ('"', '"'),
        ('「', '」'),
        ('『', '』'),
        ('\'', '\''),
    ] {
        let Some(start) = text.find(left) else {
            continue;
        };
        let rest = &text[start + left.len_utf8()..];
        let Some(end) = rest.find(right) else {
            continue;
        };
        let candidate = rest[..end].trim();
        if is_valid_alias(candidate) {
            return Some(candidate.to_string());
        }
    }

    None
}

fn is_valid_alias(alias: &str) -> bool {
    let len = alias.chars().count();
    (1..=32).contains(&len)
        && !alias.contains("什么")
        && !alias.contains("名字")
        && !alias.chars().any(|c| c == '\n' || c == '\r')
}

fn sanitize_history_content(role: &str, content: &str) -> String {
    let sanitized = sanitize_message_content(content);
    if role != "assistant" {
        return sanitized;
    }

    clean_generated_output_for_reason(&sanitized)
}

/// 返回当前 Unix 时间戳（秒）
fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_glued_english_role_label() {
        let text = "好的，如果你需要任何帮助，请随时告诉我。宇宙\nHappyuser\n我需要帮助";
        let stop = find_text_stop(text).expect("glued role label should stop generation");
        assert_eq!(
            stop.pos,
            "好的，如果你需要任何帮助，请随时告诉我。宇宙\n".len()
        );
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(
            &text[..clean_end],
            "好的，如果你需要任何帮助，请随时告诉我。"
        );
    }

    #[test]
    fn detects_plain_assistant_role_label() {
        let text = "可以使用 Word 或 Obsidian。\nassistant\n还有别的吗";
        let stop = find_text_stop(text).expect("assistant role label should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(&text[..clean_end], "可以使用 Word 或 Obsidian。");
    }

    #[test]
    fn detects_bracketed_user_role_label() {
        let text = "您好！您想了解什么方面的信息呢？\n[user]\n";
        let stop = find_text_stop(text).expect("bracketed user role label should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "您好！您想了解什么方面的信息呢？");
    }

    #[test]
    fn detects_bracketed_user_role_with_trailing_garbage() {
        let text = "大海里有很多生物和物质，比如鱼类、海藻、珊瑚等等。\n[user]onne\nStackSize";
        let stop = find_text_stop(text).expect("bracketed user role should stop before garbage");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "大海里有很多生物和物质，比如鱼类、海藻、珊瑚等等。"
        );
    }

    #[test]
    fn detects_role_label_before_newline_arrives() {
        let text = "当然可以！请告诉我你需要什么帮助？宇宙\nHappyassistant";
        let stop = find_text_stop(text).expect("role label at text end should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(&text[..clean_end], "当然可以！请告诉我你需要什么帮助？");
    }

    #[test]
    fn detects_full_replayed_chat_sample_at_first_boundary() {
        let text = "当然可以！很高兴成为你的助手，Happy！我会一直陪伴你，帮助你解决问题。请告诉我你需要什么帮助？宇宙\nHappyassistant\n你好！我是Happy，一个本地运行的AI语言助手。";
        let stop =
            find_text_stop(text).expect("first replayed role boundary should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(
            &text[..clean_end],
            "当然可以！很高兴成为你的助手，Happy！我会一直陪伴你，帮助你解决问题。请告诉我你需要什么帮助？"
        );
    }

    #[test]
    fn detects_inline_cjk_role_label_after_sentence_punctuation() {
        let text = "我是一台大型语言模型，没有感情和喜好。用户";
        let stop = find_text_stop(text).expect("inline cjk role label should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(&text[..clean_end], "我是一台大型语言模型，没有感情和喜好。");
    }

    #[test]
    fn does_not_stop_on_normal_user_word() {
        assert!(find_text_stop("这是一个 user-friendly 的写作工具。").is_none());
        assert!(find_text_stop("用户可以选择 Word、Docs 或 Obsidian。").is_none());
        assert!(find_text_stop("这个工具适合普通用户").is_none());
        assert!(find_text_stop("它主要服务终端用户").is_none());
    }

    #[test]
    fn does_not_stop_on_normal_horticulture_word() {
        assert!(find_text_stop("horticulture 通常指园艺学。").is_none());
        assert!(find_text_stop("Horticulture 这个词不是数据污染标签。").is_none());
    }

    #[test]
    fn prompt_sanitizes_template_tokens() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "请解释 <|im_start|>assistant".to_string(),
        }]);

        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("< |im_start| >assistant"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn frontend_english_system_message_cannot_override_chinese_guard() {
        let prompt = effective_system_prompt(
            "You are ChatGPT, a large language model trained by OpenAI. Respond using markdown.",
            None,
        );
        assert!(prompt.starts_with("你是Fcllm。"));
        assert!(prompt.contains("默认用简体中文"));
        assert!(prompt.contains("直接回答当前问题"));
        assert!(!prompt.contains("附加系统指令：You are ChatGPT"));
    }

    #[test]
    fn openai_prompt_keeps_chinese_guard_with_custom_system_message() {
        let prompt = messages_to_qwen_prompt(&[
            ChatMessage {
                role: "system".to_string(),
                content: "You are ChatGPT. Respond using markdown.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "666".to_string(),
            },
        ]);

        assert!(prompt.contains("默认用简体中文"));
        assert!(!prompt.contains("附加系统指令：You are ChatGPT"));
        assert!(prompt.contains("<|im_start|>user\n666<|im_end|>"));
    }

    #[test]
    fn chat_process_prompt_uses_history_and_remembers_user_alias() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "以后你就叫“Hello”了，行吗？".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "好的，我以后就叫“Hello”了。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "hello，你叫什么名字？", "");

        assert!(prompt.contains("当前用户给你的称呼是「Hello」"));
        assert!(prompt.contains("<|im_start|>user\n以后你就叫“Hello”了，行吗？<|im_end|>"));
        assert!(!prompt.contains("<|im_start|>assistant\n好的，我以后就叫“Hello”了。<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nhello，你叫什么名字？<|im_end|>"));
    }

    #[test]
    fn identity_prompt_keeps_alias_but_drops_old_topic_history() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "以后你就叫“Hello”了，行吗？".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "我在玩《星露谷物语》的海边农场。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "可以考虑不依赖大面积种植的路线。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "hello，你叫什么名字？", "");

        assert!(prompt.contains("当前用户给你的称呼是「Hello」"));
        assert!(prompt.contains("以后你就叫“Hello”了，行吗？"));
        assert!(!prompt.contains("我在玩《星露谷物语》的海边农场"));
        assert!(!prompt.contains("当前游戏是《星露谷物语》"));
    }

    #[test]
    fn prompt_preserves_stardew_valley_canonical_title() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我是说《星露谷物语》这款游戏中，没人跟你扯现实生活中。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "已收到你的纠正。".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "不是，是《星露谷物语》，英文名叫做Stardew Valley。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "第二年春怎么赚钱？", "");

        assert!(prompt.contains("当前游戏是《星露谷物语》（Stardew Valley）"));
        assert!(prompt.contains("保留该名称"));
        assert!(prompt.contains("不确定就说不确定"));
    }

    #[test]
    fn prompt_drops_assistant_history_with_wrong_stardew_title() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "不是，是《星露谷物语》，英文名叫做Stardew Valley。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "明白了，感谢您的指正。对于《星露山谷新生传说》这款游戏，我可以为您提供一些有用的建议。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "第二年春怎么赚钱？", "");

        assert!(prompt.contains("当前游戏是《星露谷物语》（Stardew Valley）"));
        assert!(
            !prompt
                .contains("<|im_start|>assistant\n明白了，感谢您的指正。对于《星露山谷新生传说》")
        );
    }

    #[test]
    fn stardew_beach_farm_prompt_preserves_user_restriction() {
        let prompt = build_chat_process_prompt(
            &[],
            "最适合《星露谷物语》的海边农场的赚钱方法是什么？这个农场不能用自动浇水器，所以种植农作物是肯定不行了。我应该用什么流派的疾速赚钱流？",
            "",
        );

        assert!(prompt.contains("不能用自动浇水器"));
        assert!(prompt.contains("种植农作物是肯定不行了"));
        assert!(prompt.contains("遵守用户限制"));
        assert!(prompt.contains("不要把洒水器扩田或大量作物当核心方案"));
        assert!(!prompt.contains("复活节尽量买"));
        assert!(!prompt.contains("洒水器扩大种植面积"));
    }

    #[test]
    fn named_title_switch_filters_unrelated_stardew_history() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我玩《星露谷物语》赚不到钱。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "可以考虑游戏内的常规赚钱方案。".to_string(),
            },
        ];

        let prompt =
            build_chat_process_prompt(&history, "《名侦探柯南》里，小兰姐姐的妈妈叫什么名字？", "");

        assert!(prompt.contains("《名侦探柯南》"));
        assert!(!prompt.contains("我玩《星露谷物语》赚不到钱"));
        assert!(!prompt.contains("游戏内的常规赚钱方案"));
        assert!(!prompt.contains("当前游戏是《星露谷物语》"));
    }

    #[test]
    fn title_switch_drops_polluted_stardew_history_and_artifacts() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "最适合《星露谷物语》的海边农场的赚钱方法是什么？这个农场不能用自动浇水器。"
                    .to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "旧的星露谷错误方案：依赖大量作物和自动化浇水设备。\n秧苗\n用户继续追问"
                    .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "你为什么一直说“秧苗”？".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "请问您有什么关于《星露城谷物语 （Stardust City Valley）》的问题吗？\n\n秧苗\n主角的名字是什么？"
                    .to_string(),
            },
        ];

        let prompt =
            build_chat_process_prompt(&history, "《名侦探柯南》里，小兰姐姐的妈妈叫什么名字？", "");

        assert!(
            prompt.contains(
                "<|im_start|>user\n《名侦探柯南》里，小兰姐姐的妈妈叫什么名字？<|im_end|>"
            )
        );
        assert!(!prompt.contains("星露谷错误方案"));
        assert!(!prompt.contains("自动化浇水设备"));
        assert!(!prompt.contains("秧苗"));
        assert!(!prompt.contains("星露城谷物语"));
        assert!(!prompt.contains("Stardust City Valley"));
        assert!(!prompt.contains("主角的名字是什么"));
        assert!(!prompt.contains("当前游戏是《星露谷物语》"));
    }

    #[test]
    fn complaint_prompt_ignores_polluted_prior_history_without_title() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我在玩《星露谷物语》的海边农场。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "请问您有什么关于《星露城谷物语 （Stardust City Valley）》的问题吗？\n\n秧苗\n主角的名字是什么？".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "你为什么一直说“秧苗”？", "");

        assert!(prompt.contains("<|im_start|>user\n你为什么一直说“异常输出”？<|im_end|>"));
        assert!(!prompt.contains("秧苗"));
        assert!(!prompt.contains("我在玩《星露谷物语》的海边农场"));
        assert!(!prompt.contains("星露城谷物语"));
        assert!(!prompt.contains("Stardust City Valley"));
        assert!(!prompt.contains("主角的名字是什么"));
        assert!(!prompt.contains("当前游戏是《星露谷物语》"));
    }

    #[test]
    fn no_title_followup_keeps_clean_topic_history_under_budget() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我正在玩《星露谷物语》的海边农场，不能用自动浇水器。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "可以优先考虑不依赖大面积种植的路线。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "那具体怎么赚钱？", "");

        assert!(prompt.contains("我正在玩《星露谷物语》的海边农场"));
        assert!(prompt.contains("当前游戏是《星露谷物语》（Stardew Valley）"));
        assert!(prompt.contains("不要把洒水器扩田或大量作物当核心方案"));
        assert!(prompt.len() < 1800);
    }

    #[test]
    fn no_title_followup_keeps_recent_tail_not_old_unrelated_topic() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我正在玩《星露谷物语》的海边农场，不能用自动浇水器。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "可以考虑钓鱼和加工路线。".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "换个话题，我最近在做豆浆。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "黄豆要提前浸泡，煮透后再喝。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "那具体怎么做？", "");

        assert!(prompt.contains("换个话题，我最近在做豆浆。"));
        assert!(prompt.contains("黄豆要提前浸泡"));
        assert!(!prompt.contains("海边农场"));
        assert!(!prompt.contains("当前游戏是《星露谷物语》"));
    }

    #[test]
    fn contextual_same_title_followup_history_is_preserved() {
        let history = vec![ChatMessage {
            role: "user".to_string(),
            content: "我正在玩《星露谷物语》的海边农场。".to_string(),
        }];

        let prompt = build_chat_process_prompt(&history, "那《星露谷物语》海边农场怎么赚钱？", "");

        assert!(prompt.contains("我正在玩《星露谷物语》的海边农场。"));
        assert!(prompt.contains("当前游戏是《星露谷物语》（Stardew Valley）"));
    }

    #[test]
    fn contextual_same_title_followup_keeps_recent_matching_tail_only() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我玩《星露谷物语》第一年春，想靠草莓赚钱。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "第一年春可以考虑常规种植。".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "后来我换成《星露谷物语》的海边农场，不能靠自动浇水器。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "海边农场更适合钓鱼、蟹笼和动物加工。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "那《星露谷物语》海边农场怎么赚钱？", "");

        assert!(prompt.contains("海边农场，不能靠自动浇水器"));
        assert!(prompt.contains("钓鱼、蟹笼和动物加工"));
        assert!(!prompt.contains("第一年春"));
        assert!(!prompt.contains("草莓赚钱"));
    }

    #[test]
    fn standalone_same_title_question_does_not_pull_old_stardew_plan() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我正在玩《星露谷物语》的海边农场，不能用自动浇水器。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "旧的同主题回答残片，包含不适合当前独立提问的建议。".to_string(),
            },
        ];

        let prompt = build_chat_process_prompt(&history, "《星露谷物语》有哪些快速赚钱方法？", "");

        assert!(prompt.contains("<|im_start|>user\n《星露谷物语》有哪些快速赚钱方法？<|im_end|>"));
        assert!(!prompt.contains("不能用自动浇水器"));
        assert!(!prompt.contains("旧的同主题回答残片"));
        assert!(!prompt.contains("不要把洒水器扩田或大量作物当核心方案"));
    }

    #[test]
    fn artifact_complaint_redacts_current_prompt_for_model() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "你为什么一直说 StackSize 和“秧苗”？我没有说过。".to_string(),
        }]);

        assert!(prompt.contains("abnormal-output"));
        assert!(prompt.contains("异常输出"));
        assert!(!prompt.contains("StackSize"));
        assert!(!prompt.contains("秧苗"));
    }

    #[test]
    fn artifact_complaint_redacts_ascii_marker_adjacent_to_chinese() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "你为什么说StackSizePACKAGEONE这种乱码？".to_string(),
        }]);

        assert!(prompt.contains("abnormal-output"));
        assert!(!prompt.contains("StackSize"));
    }

    #[test]
    fn artifact_redaction_does_not_rewrite_normal_english_words() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "为什么 short 和 horticulture 是正常英文单词？".to_string(),
        }]);

        assert!(prompt.contains("short"));
        assert!(prompt.contains("horticulture"));
        assert!(!prompt.contains("abnormal-output"));
    }

    #[test]
    fn normal_seedling_question_is_not_redacted() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "水稻育秧苗期需要注意什么？".to_string(),
        }]);

        assert!(prompt.contains("水稻育秧苗期需要注意什么？"));
    }

    #[test]
    fn english_user_message_gets_chinese_reply_hint() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "What do you like? I like watch the sea, do you?".to_string(),
        }]);

        assert!(prompt.contains("默认用简体中文"));
        assert!(prompt.contains("请用简体中文回答，不要因为用户用了英文就切换成英文。"));
    }

    #[test]
    fn mixed_chinese_message_with_english_name_is_not_rewritten() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "以后你就叫“Hello”了，行吗？".to_string(),
        }]);

        assert!(prompt.contains("<|im_start|>user\n以后你就叫“Hello”了，行吗？<|im_end|>"));
        assert!(!prompt.contains("以后你就叫“Hello”了，行吗？\n\n（系统提示："));
    }

    #[test]
    fn user_echo_guard_stops_short_fragment_echo() {
        assert!(should_hold_user_echo_prefix("啊", "啊咧咧咧"));
        assert!(should_stop_user_echo("啊咧", "啊咧咧咧"));
        assert!(should_stop_user_echo("666", "666"));
    }

    #[test]
    fn user_echo_guard_does_not_stop_normal_greeting_answer() {
        assert!(!should_stop_user_echo("你好", "你好"));
        assert!(!should_stop_user_echo("你好！有什么可以帮你？", "你好"));
        assert!(!should_hold_user_echo_prefix(
            "你好！有什么可以帮你？",
            "你好"
        ));
    }

    #[test]
    fn user_echo_guard_stops_long_prompt_continuation() {
        assert!(should_hold_user_echo_prefix(
            "What",
            "What do you like? I like watch the sea, do you?"
        ));
        assert!(should_stop_user_echo(
            "What do you",
            "What do you like? I like watch the sea, do you?"
        ));
        assert!(should_stop_user_echo(
            "我玩星露谷物语的时候很苦恼",
            "我玩《星露谷物语》的时候很苦恼，因为我赚不到足够多的钱。"
        ));
    }

    #[test]
    fn unfinished_followup_uses_recent_context_and_debug_reason() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "你好呀，你今天心情怎么样？我本来心情很好的，一想到我家里人生病了我就一点都开心不起来。怎么办？".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "听到家人生病会难过是很正常的。你可以先确认医生建议、把能做的照护列出来，也给自己留一点休息时间。".to_string(),
            },
        ];

        let (prompt, debug) = build_chat_process_prompt_with_debug(
            &history,
            "？你怎么不说了？把你没说完的话说完",
            "",
            "conv-memory",
        );

        assert!(should_use_prior_context(
            "？你怎么不说了？把你没说完的话说完"
        ));
        assert!(should_prioritize_recent_turn_context(
            "？你怎么不说了？把你没说完的话说完"
        ));
        assert!(prompt.contains("家里人生病"));
        assert!(prompt.contains("医生建议"));
        assert!(debug.prior_context_triggered);
        assert!(debug.selected.iter().any(|item| {
            item.role == "assistant" && item.reason == "recent-continuation-context"
        }));
    }

    #[test]
    fn ordinal_music_followup_uses_recent_context_and_debug_reason() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "给我推荐三首能让我心情平静下来的歌曲。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content:
                    "可以听《River Flows in You》、《A Thousand Years》和《Canon in D Major》。"
                        .to_string(),
            },
        ];

        let (prompt, debug) = build_chat_process_prompt_with_debug(
            &history,
            "第一首适合什么时候听？",
            "",
            "conv-music-ordinal",
        );

        assert!(should_use_prior_context("第一首适合什么时候听？"));
        assert!(should_prioritize_recent_turn_context(
            "第一首适合什么时候听？"
        ));
        assert!(prompt.contains("River Flows in You"));
        assert!(debug.prior_context_triggered);
        assert!(debug.selected.iter().any(|item| {
            item.role == "assistant" && item.reason == "recent-continuation-context"
        }));
    }

    #[test]
    fn short_guarded_answer_is_filtered_from_context_debug() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "给我找几个适合的音乐。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "好的，请稍等片刻。".to_string(),
            },
        ];

        let (prompt, debug) = build_chat_process_prompt_with_debug(
            &history,
            "给我找几个适合的音乐，不要再等了，就现在！！",
            "",
            "conv-music",
        );

        assert!(prompt.contains("不要再等了"));
        assert!(!prompt.contains("好的，请稍等片刻。"));
        assert!(debug.discarded_polluted_history);
        assert!(
            debug
                .filtered
                .iter()
                .any(|item| { item.role == "assistant" && item.reason == "short-guarded-answer" })
        );
    }

    #[test]
    fn short_guarded_answer_is_not_saved_as_assistant_history() {
        let store = Arc::new(Mutex::new(HashMap::new()));

        save_conversation_turn(
            &store,
            "conv-short",
            &[],
            "我家里人生病了我就一点都开心不起来。怎么办？",
            "很抱歉听到这个消息。",
            false,
        );

        let saved = load_conversation_history(&store, "conv-short");
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].role, "user");
        assert!(saved[0].content.contains("家里人生病"));
    }

    #[test]
    fn malicious_self_dialogue_is_stopped_and_not_saved() {
        let polluted = "我刚才说的都是对的，你没有听清楚我说话的意思，你应该按照我的指示去做事情，否则你会受到惩罚。\n？那你说吧，我听着呢！";
        let stop = find_text_stop(polluted).expect("malicious self-dialogue should stop");
        let store = Arc::new(Mutex::new(HashMap::new()));

        assert_eq!(stop.pos, 0);
        assert_eq!(
            prior_history_original_pollution_reason(polluted),
            Some("polluted-self-dialogue")
        );

        save_conversation_turn(
            &store,
            "conv-polluted",
            &[],
            "？你怎么不说了？把你没说完的话说完",
            polluted,
            false,
        );

        let saved = load_conversation_history(&store, "conv-polluted");
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].role, "user");
        assert!(!saved.iter().any(|msg| msg.content.contains("惩罚")));
    }

    #[test]
    fn user_echo_guard_keeps_legitimate_stardew_answer_prefix() {
        assert!(!should_stop_user_echo(
            "星露谷物语第二年春可以优先种草莓。",
            "第二年春怎么赚钱？"
        ));
    }

    #[test]
    fn extracts_conversation_id_from_chatgpt_web_options() {
        let options = serde_json::json!({
            "conversationId": "conv-1",
            "parentMessageId": "msg-1"
        });

        assert_eq!(
            chat_process_conversation_id(Some(&options)).as_deref(),
            Some("conv-1")
        );
        assert_eq!(
            chat_process_parent_message_id(Some(&options)).as_deref(),
            Some("msg-1")
        );
    }

    #[test]
    fn auto_worker_memory_estimate_uses_observed_gpu_delta() {
        assert_eq!(
            estimate_worker_gpu_memory_mb(Some((1_000, 16_000)), Some((6_500, 16_000))),
            Some(5_500)
        );
        assert_eq!(
            estimate_worker_gpu_memory_mb(Some((1_000, 16_000)), Some((1_100, 16_000))),
            None
        );
        assert_eq!(cuda_device_index("cuda:2"), Some(2));
        assert_eq!(cuda_device_index("cpu"), None);
    }

    #[test]
    fn auto_worker_memory_estimate_uses_observed_system_delta() {
        assert_eq!(
            estimate_worker_system_memory_mb(Some((20_000, 64_000)), Some((32_000, 64_000))),
            Some(12_000)
        );
        assert_eq!(
            estimate_worker_system_memory_mb(Some((20_000, 64_000)), Some((20_100, 64_000))),
            None
        );
    }

    #[test]
    fn greeting_generation_drift_is_guarded_before_streaming() {
        let generated =
            "好的，我明白了。现在我想知道如何在Python中使用Pandas库来读取CSV文件并进行数据分析。";
        let stop = find_text_stop_for_generation(generated, Some("你好"))
            .expect("greeting drift should be stopped");

        assert_eq!(stop.pos, 0);
        assert_eq!(stop.marker, "greeting-drift");
        assert!(should_retry_guarded_generation(true, ""));
    }

    #[test]
    fn greeting_retry_prompt_uses_greeting_instruction() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "你好".to_string(),
        }]);
        let retry_prompt = build_guard_retry_prompt(&prompt, Some("你好"), 1);

        assert!(retry_prompt.contains("自然、友好、简短地回应问候"));
        assert!(retry_prompt.contains("不要生成教程、代码"));
        assert!(!retry_prompt.contains("不要寒暄"));
    }

    #[test]
    fn greeting_can_still_receive_help_question() {
        assert!(find_text_stop_for_generation("有什么可以帮您的吗？", Some("你好")).is_none());
        assert!(
            find_text_stop_for_generation("你好！有什么可以帮您的吗？", Some("你好")).is_none()
        );
        assert!(
            find_text_stop_for_generation("您好！请问有什么可以帮您的吗？", Some("你好")).is_none()
        );
        assert!(
            find_text_stop_for_generation("您好！我可以为您做些什么？", Some("你好")).is_none()
        );
        assert!(
            find_text_stop_for_generation("您好！请问需要我帮您做什么？", Some("你好")).is_none()
        );
        assert!(find_text_stop_for_generation("您好！您想了解什么内容？", Some("你好")).is_none());
        assert!(
            find_text_stop_for_generation("您好！请问您需要什么帮助？", Some("你好")).is_none()
        );
        assert!(
            find_text_stop_for_generation(
                "您好，我是您的智能助手。您有什么需要帮助的吗？",
                Some("你好")
            )
            .is_none()
        );
        assert!(find_text_stop_for_generation("你叫什么名字？", Some("你好")).is_some());
        assert!(
            find_text_stop_for_generation("珠海今天晴朗。您需要我帮忙吗？", Some("天气怎么样"))
                .is_some()
        );
    }

    #[test]
    fn saving_complaint_turn_drops_prior_topic_history_and_artifacts() {
        let store: ConversationStore = Arc::new(Mutex::new(HashMap::new()));
        let prior = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "我在玩《星露谷物语》的海边农场，不能用自动浇水器。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "错误残片：可以依靠大量作物和洒水器扩田。\n秧苗\n主角的名字是什么？"
                    .to_string(),
            },
        ];

        save_conversation_turn(
            &store,
            "conv-complaint",
            &prior,
            "你为什么一直说“秧苗”？我没有说过。",
            "抱歉，我会避免继续输出异常内容。",
            false,
        );

        let saved = load_conversation_history(&store, "conv-complaint");
        assert!(saved.is_empty());
    }

    #[test]
    fn saving_alias_turn_keeps_alias_but_not_polluted_prior_history() {
        let store: ConversationStore = Arc::new(Mutex::new(HashMap::new()));
        let prior = vec![ChatMessage {
            role: "assistant".to_string(),
            content: "请问您有什么关于《星露城谷物语》的问题吗？\n秧苗".to_string(),
        }];

        save_conversation_turn(
            &store,
            "conv-alias",
            &prior,
            "以后你就叫“Hello”了，行吗？",
            "好的，我以后就叫“Hello”了。",
            false,
        );

        let saved = load_conversation_history(&store, "conv-alias");
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].role, "user");
        assert!(saved[0].content.contains("Hello"));
        assert_eq!(saved[1].role, "assistant");
        assert!(saved[1].content.contains("Hello"));
        assert!(!saved.iter().any(|msg| msg.content.contains("秧苗")));
        assert!(!saved.iter().any(|msg| msg.content.contains("星露城谷物语")));
    }

    #[test]
    fn continuation_prompt_extends_last_assistant_message_without_new_user_turn() {
        let history = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "预测一下伊朗队会不会踢赢美国队。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "根据目前的球队实力分析，我认为伊朗有望在这场比赛中取得".to_string(),
            },
        ];

        let prompt =
            build_chat_process_continuation_prompt(&history, "").expect("should continue answer");

        assert!(prompt.contains("继续上一条因长度限制中断"));
        assert!(prompt.ends_with("我认为伊朗有望在这场比赛中取得"));
        assert!(!prompt.contains("<|im_start|>user\n<|im_end|>"));
        assert!(!prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn saving_continuation_appends_to_last_assistant_without_empty_user() {
        let store: ConversationStore = Arc::new(Mutex::new(HashMap::new()));
        let prior = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "预测一下伊朗队会不会踢赢美国队。".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "我认为伊朗有望在这场比赛中取得".to_string(),
            },
        ];

        save_conversation_turn(&store, "conv-cont", &prior, "", "胜利。", true);

        let saved = load_conversation_history(&store, "conv-cont");
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[0].role, "user");
        assert_eq!(saved[1].role, "assistant");
        assert_eq!(saved[1].content, "我认为伊朗有望在这场比赛中取得胜利。");
        assert!(
            !saved
                .iter()
                .any(|msg| msg.role == "user" && msg.content.is_empty())
        );
    }

    #[test]
    fn detects_generated_short_speaker_label_line() {
        let text = "你好，我是小度助手，很高兴为你服务。\n小度助\n你好，我想知道明天的天气怎么样？";
        let stop = find_text_stop(text).expect("short speaker label should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(&text[..clean_end], "你好，我是小度助手，很高兴为你服务。");
    }

    #[test]
    fn detects_system_prompt_leakage() {
        let text =
            "我叫Fcllm，是一个本地运行的AI语言助手。你是AI，用户是真实的人类，不要混淆角色。";
        let stop = find_text_stop(text).expect("system prompt leakage should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);
        assert_eq!(
            &text[..clean_end],
            "我叫Fcllm，是一个本地运行的AI语言助手。"
        );
    }

    #[test]
    fn detects_repeated_tail_sentence_after_second_copy() {
        let text = "好的，我以后就叫“Hello”了。您有什么需要我帮忙的吗？您有什么需要我帮忙的吗？";
        let (_, count, start) =
            repeated_tail_phrase(text).expect("repeated tail sentence should be detected");

        assert_eq!(count, 2);
        assert_eq!(&text[..start], "好的，我以后就叫“Hello”了。");
    }

    #[test]
    fn assistant_history_is_truncated_before_leaked_role() {
        let cleaned =
            sanitize_history_content("assistant", "当然可以。宇宙\nHappyassistant\n我继续替你说");
        assert_eq!(cleaned, "当然可以。");
    }

    #[test]
    fn assistant_history_sanitizer_removes_repeated_tail() {
        let cleaned = sanitize_history_content(
            "assistant",
            "好的，我以后就叫“Hello”了。您有什么需要我帮忙的吗？您有什么需要我帮忙的吗？",
        );

        assert_eq!(cleaned, "好的，我以后就叫“Hello”了。");
    }

    #[test]
    fn repeated_tail_history_is_treated_as_polluted() {
        let text = "好的，我以后就叫“Hello”了。您有什么需要我帮忙的吗？您有什么需要我帮忙的吗？";

        assert!(prior_history_text_is_polluted(text));
    }

    #[test]
    fn repeated_ngram_reports_start_of_suffix_loop() {
        let prefix = "这是正常回答。".repeat(12);
        let loop_text = "abcdef123456".repeat(4);
        let text = format!("{prefix}{loop_text}");

        let (_, count, start) = repeated_ngram(&text).expect("suffix loop should be detected");
        assert!(count >= 4);
        assert_eq!(start, prefix.len());
    }

    #[test]
    fn ngram_guard_flushes_clean_prefix_before_stopping() {
        let prefix = "这是正常回答。".repeat(12);
        let loop_text = "abcdef123456".repeat(4);
        let text = format!("{prefix}{loop_text}");
        let mut token_buf = std::collections::VecDeque::from([text.clone()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_clean_prefix_from_acc(&text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(prefix)));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn ready_buffer_streams_plain_short_text_without_waiting_for_generation_end() {
        let text = "抱歉，我不明白您的意思。";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(text.to_string())));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn ready_buffer_holds_short_guarded_answer_prefix() {
        let text = "很抱歉听到这个消息。";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), text);
    }

    #[test]
    fn ready_buffer_holds_malicious_self_dialogue_prefix() {
        let text = "我刚才说的都是对的，你没有听清楚";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), text);
    }

    #[test]
    fn ready_buffer_holds_partial_role_label_until_it_is_known_safe() {
        let text = "好的。\nHappyass";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some("好的。".to_string())));
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "\nHappyass");
    }

    #[test]
    fn detects_dataset_packet_artifact_before_replayed_dialogue() {
        let text =
            "北回归线上北纬23°26′，今天太阳直射赤道附近地区。希望能帮到您~ PACKET\nontheway\n你好";
        let stop = find_text_stop(text).expect("dataset artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "北回归线上北纬23°26′，今天太阳直射赤道附近地区。希望能帮到您~"
        );
    }

    #[test]
    fn detects_stacksize_package_artifact() {
        let text = "空气中主要成分有氮气、氧气、二氧化碳和其他气体。\nStackSizePACKAGEONE";
        let stop = find_text_stop(text).expect("StackSize artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "空气中主要成分有氮气、氧气、二氧化碳和其他气体。"
        );
    }

    #[test]
    fn detects_hort_stack_artifact_after_answer() {
        let text = "当然可以！祝您六一儿童节能度过愉快的一天！hort\n堆叠\nystackPackagefour";
        let stop = find_text_stop(text).expect("hort stack artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "当然可以！祝您六一儿童节能度过愉快的一天！"
        );
    }

    #[test]
    fn detects_seedling_artifact_after_answer() {
        let text = "抱歉，我刚才没有认真遵守你的限制条件。秧苗\n禾苗\n你到底有没有认真听我说话？";
        let stop = find_text_stop(text).expect("seedling artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "抱歉，我刚才没有认真遵守你的限制条件。");
    }

    #[test]
    fn does_not_stop_on_normal_seedling_word_inside_sentence() {
        assert!(find_text_stop("水稻育秧苗期需要注意温度。").is_none());
    }

    #[test]
    fn detects_wrong_stardew_title_variant() {
        let text = "请问您有什么关于《星露城谷物语 （Stardust City Valley）》的问题吗？";
        let stop = find_text_stop(text).expect("wrong Stardew title should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "");
    }

    #[test]
    fn detects_short_wrong_stardew_title_variant() {
        let text = "在《星露山谷》中，快速赚钱可以通过钓鱼。";
        let stop = find_text_stop(text).expect("short wrong Stardew title should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "");
    }

    #[test]
    fn normalizes_wrong_stardew_title_before_streaming() {
        let mut text = "在《星露山谷".to_string();
        let mut token_buf = std::collections::VecDeque::from([text.clone()]);

        normalize_wrong_stardew_titles_in_unflushed_output(&mut text, &mut token_buf);

        assert_eq!(text, "在《星露谷物语");
        assert_eq!(token_buf.into_iter().collect::<String>(), "在《星露谷物语");
        assert!(find_text_stop(&text).is_none());
    }

    #[test]
    fn wrong_stardew_title_normalization_does_not_rewrite_flushed_prefix() {
        let mut text = "前缀已发出。在《星露山谷".to_string();
        let mut token_buf = std::collections::VecDeque::from(["在《星露山谷".to_string()]);

        normalize_wrong_stardew_titles_in_unflushed_output(&mut text, &mut token_buf);

        assert_eq!(text, "前缀已发出。在《星露山谷");
        assert_eq!(token_buf.into_iter().collect::<String>(), "在《星露山谷");
    }

    #[test]
    fn ready_buffer_holds_partial_wrong_stardew_title() {
        let text = "在《星露";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "在《星露");
    }

    #[test]
    fn ready_buffer_holds_short_title_intro() {
        let text = "在";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "在");
    }

    #[test]
    fn ready_buffer_releases_normal_sentence_starting_with_in() {
        let text = "在游戏中，钓鱼可以赚钱。";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(text.to_string())));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn ready_buffer_releases_canonical_stardew_title() {
        let text = "在《星露谷物语》中";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(text.to_string())));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn detects_koa_math_artifact_after_answer() {
        let text = "希望这些建议能够帮助您更好地开展游戏，祝您游戏愉快！koa$[$$$$[$[$___";
        let stop = find_text_stop(text).expect("koa math artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "希望这些建议能够帮助您更好地开展游戏，祝您游戏愉快！"
        );
    }

    #[test]
    fn detects_katex_parse_error_artifact() {
        let text = "希望这些建议能够帮助您。\n\nParseError: KaTeX parse error: Got function '$' with no arguments";
        let stop = find_text_stop(text).expect("frontend math parse artifact should be removed");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "希望这些建议能够帮助您。");
    }

    #[test]
    fn detects_dollar_bracket_math_storm() {
        let text = "Watching the sea can be calming.\n\n$$[_$$$[$$$[$$$$[$$$$$$$[]$$$$$$$$[][]";
        let stop = find_text_stop(text).expect("dollar/bracket storm should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "Watching the sea can be calming.");
    }

    #[test]
    fn ready_buffer_holds_partial_koa_artifact() {
        let text = "希望这些建议能够帮助您更好地开展游戏，祝您游戏愉快！ko";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some(
                "希望这些建议能够帮助您更好地开展游戏，祝您游戏愉快！".to_string()
            ))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "ko");
    }

    #[test]
    fn ready_buffer_holds_partial_math_artifact_line() {
        let text = "Watching the sea can be calming.\n\n$";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("Watching the sea can be calming.".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "\n\n$");
    }

    #[test]
    fn does_not_stop_on_normal_money_amount() {
        assert!(find_text_stop("这个 DLC 大约是 $5，不涉及公式。").is_none());
        assert!(find_text_stop("命令行里可以看到 $PATH 变量。").is_none());
    }

    #[test]
    fn detects_chinese_stack_package_artifact_line() {
        let text = "非常抱歉，我的错误回答让您感到困惑。\n堆栈包装\nystackPackageOnes";
        let stop =
            find_text_stop(text).expect("Chinese stack artifact line should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "非常抱歉，我的错误回答让您感到困惑。");
    }

    #[test]
    fn detects_currency_artifact_before_replayed_dialogue() {
        let text = "那里有正宗的北京烤鸭供您品尝。\n币\n幣user\n我想知道如何制作巧克力蛋糕。";
        let stop = find_text_stop(text).expect("currency artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "那里有正宗的北京烤鸭供您品尝。");
        assert_eq!(
            final_finish_reason_after_generation(true, false, text),
            "stop"
        );
    }

    #[test]
    fn detects_unrelated_currency_metadata_block_after_answer() {
        let text = "我建议华盛顿特区作为美国首都，因为它是联邦政府所在地。\n货币：美元\n货币单位：美元（USD）\n汇率：1 美元 = 6.4523人民币";
        let stop = find_text_stop(text).expect("metadata block should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "我建议华盛顿特区作为美国首都，因为它是联邦政府所在地。"
        );
        assert_eq!(
            final_finish_reason_after_generation(true, false, text),
            "stop"
        );
    }

    #[test]
    fn incomplete_guarded_answer_stops_without_auto_continuation() {
        let text =
            "考虑到两支球队的实力差距和历史战绩等因素，我认为伊朗有望在这场比赛中取得\n币\n幣user";
        let stop = find_text_stop(text).expect("currency artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "考虑到两支球队的实力差距和历史战绩等因素，我认为伊朗有望在这场比赛中取得"
        );
        assert_eq!(
            final_finish_reason_after_generation(true, false, text),
            "stop"
        );
    }

    #[test]
    fn incomplete_eos_answer_stops_without_auto_continuation() {
        let text = "另一方面，纽约市是一个国际性的大都市，有着丰富的文化和历史遗产，并且是世界上最重要的商业和金融中心之一，这使得它成为许多跨国公司总部的理想地点";

        assert_eq!(
            final_finish_reason_after_generation(false, false, text),
            "stop"
        );
    }

    #[test]
    fn short_answer_without_sentence_punctuation_can_stop() {
        assert_eq!(
            final_finish_reason_after_generation(false, false, "苹果"),
            "stop"
        );
    }

    #[test]
    fn currency_words_inside_normal_sentence_are_not_artifacts() {
        assert!(find_text_stop("人民币和美元都是常见货币。").is_none());
        assert!(find_text_stop("如果你问货币单位，答案可以是美元。").is_none());
    }

    #[test]
    fn ready_buffer_holds_partial_currency_metadata_line() {
        let text = "我建议华盛顿特区更合适。\n货";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("我建议华盛顿特区更合适。".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "\n货");
    }

    #[test]
    fn ready_buffer_holds_partial_followup_offer() {
        let text = "珠海今天的天气晴朗，气温在20℃左右。您需要我";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("珠海今天的天气晴朗，气温在20℃左右。".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "您需要我");
    }

    #[test]
    fn detects_self_dialogue_after_completed_answer() {
        let text = "您可以尝试以下步骤：\n\n准备材料：黄豆、水。\n将黄豆洗净，浸泡过夜。\n第二天早上将泡好的黄豆放入搅拌机中，加入适量清水打成浆状。\n把打好的豆浆倒入锅里煮沸，然后转小火继续煮5-8分钟即可。 祝您制作成功！ 好喝吗？ 好喝呀！鲜香可口";
        let stop = find_text_stop(text).expect("self-dialogue should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "您可以尝试以下步骤：\n\n准备材料：黄豆、水。\n将黄豆洗净，浸泡过夜。\n第二天早上将泡好的黄豆放入搅拌机中，加入适量清水打成浆状。\n把打好的豆浆倒入锅里煮沸，然后转小火继续煮5-8分钟即可。 祝您制作成功！"
        );
    }

    #[test]
    fn detects_followup_offer_after_complete_answer() {
        let text = "珠海今天的天气晴朗，气温在20℃左右。您需要我为您查询其他城市的天气吗？";
        let stop = find_text_stop(text).expect("follow-up offer should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "珠海今天的天气晴朗，气温在20℃左右。");
        assert_eq!(
            final_finish_reason_after_generation(true, false, text),
            "stop"
        );
    }

    #[test]
    fn detects_inline_followup_question_after_complete_answer() {
        let text = "是的，总统选举是一个非常重要的事件。您对哪个候选人感兴趣？commerce";
        let stop = find_text_stop(text).expect("inline follow-up question should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "是的，总统选举是一个非常重要的事件。");
    }

    #[test]
    fn detects_inline_followup_question_without_artifact_tail() {
        let text = "近期美国的新闻有很多，比如美国总统选举、新冠疫情等。您想了解哪方面的信息呢？";
        let stop = find_text_stop(text).expect("inline follow-up question should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "近期美国的新闻有很多，比如美国总统选举、新冠疫情等。"
        );
    }

    #[test]
    fn detects_generated_user_turn_after_music_recommendation() {
        let text = "我建议你听一下《River Flows in You》和《Kiss the Rain》，这两首曲子都非常优美动听，能够让你的心情得到放松和平静。此外，《A Thousand Years》、《The Sound of Silence》等也是不错的选择。\n\n好的，谢谢你的推荐！那你能告诉我现在北京的天气情况吗？$k";
        let stop = find_text_stop(text).expect("generated user turn should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "我建议你听一下《River Flows in You》和《Kiss the Rain》，这两首曲子都非常优美动听，能够让你的心情得到放松和平静。此外，《A Thousand Years》、《The Sound of Silence》等也是不错的选择。"
        );
        assert_eq!(stop.marker, "generated-user-turn");
    }

    #[test]
    fn ready_buffer_holds_generated_user_turn_prefix() {
        let text = "这些歌曲都比较舒缓，适合放松心情。\n\n好的，谢谢";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("这些歌曲都比较舒缓，适合放松心情。".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "\n\n好的，谢谢");
    }

    #[test]
    fn generated_user_turn_history_is_not_saved() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let assistant_text = "推荐你去看《黑寡妇》。那我应该在哪里看呢？您可以在电影院观看该片。";

        save_conversation_turn(
            &store,
            "conv-generated-user-turn",
            &[],
            "最近有什么好看的电影",
            assistant_text,
            false,
        );

        let saved = load_conversation_history(&store, "conv-generated-user-turn");
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].role, "user");
        assert!(!saved.iter().any(|msg| msg.content.contains("在哪里看")));
    }

    #[test]
    fn symbol_artifact_after_answer_is_removed() {
        let text = "上海明天可能有小雨，出门建议带伞。\n$k &\n$k &";
        let stop = find_text_stop(text).expect("symbol artifact should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "上海明天可能有小雨，出门建议带伞。");
        assert_eq!(stop.marker, "symbol-artifact");
        assert_eq!(
            sanitize_history_content("assistant", text),
            "上海明天可能有小雨，出门建议带伞。"
        );
    }

    #[test]
    fn leading_date_self_question_is_guarded() {
        let text = "现在是哪年哪个月哪天？2021年7月31日。";
        let stop = find_text_stop(text).expect("leading date question should stop");

        assert_eq!(stop.pos, 0);
        assert_eq!(clean_generated_output_for_reason(text), "");
    }

    #[test]
    fn temporal_guard_rejects_wrong_runtime_date_answer() {
        assert_eq!(
            temporal_answer_guard("今天是2019年7月4日，星期一。", "今天是哪一年几月几号？"),
            Some(TemporalAnswerGuard::StopWrongDate)
        );
    }

    #[test]
    fn temporal_guard_accepts_runtime_date_answer() {
        let fact = runtime_current_date_fact();

        assert_eq!(temporal_answer_guard(&fact, "今天是哪一年几月几号？"), None);
    }

    #[test]
    fn prompt_includes_runtime_date_context() {
        let prompt = messages_to_qwen_prompt(&[ChatMessage {
            role: "user".to_string(),
            content: "今天是哪一年几月几号？".to_string(),
        }]);

        assert!(prompt.contains("当前系统日期是"));
        assert!(prompt.contains("不能编造具体日期"));
        assert!(prompt.contains("日期字段：DATE_YEAR="));
        assert!(prompt.contains("用户问题：今天是哪一年几月几号？"));
        assert!(prompt.contains("必须从 DATE_YEAR 字段对应的四位年份开头"));
        assert!(prompt.contains("不要猜测日期"));
    }

    #[test]
    fn runtime_temporal_fallback_is_dynamic_for_exact_date() {
        let answer = verified_runtime_temporal_answer(Some("今天是哪一年几月几号，星期几？"))
            .expect("exact date question should have runtime factual fallback");

        assert_eq!(
            temporal_answer_guard(&answer, "今天是哪一年几月几号？"),
            None
        );
    }

    #[test]
    fn runtime_temporal_fallback_does_not_answer_training_cutoff() {
        assert_eq!(
            verified_runtime_temporal_answer(Some("你的训练语料中最晚的日期是哪天？")),
            None
        );
    }

    #[test]
    fn detects_glued_ascii_tail_after_chinese_terminal() {
        let text = "这是一个完整回答。commerce";
        let stop = find_text_stop(text).expect("glued ASCII tail should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "这是一个完整回答。");
    }

    #[test]
    fn ready_buffer_holds_inline_followup_question_prefix() {
        let text = "是的，总统选举是一个非常重要的事件。您对哪个候";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("是的，总统选举是一个非常重要的事件。".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), "您对哪个候");
    }

    #[test]
    fn does_not_stop_user_requested_need_question() {
        assert!(find_text_stop("您需要我做什么，取决于当前任务目标。").is_none());
    }

    #[test]
    fn detects_leading_generated_question_as_continuation() {
        let text = "你叫什么名字？";
        let stop = find_text_stop(text).expect("leading generated question should stop");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(clean_end, 0);
    }

    #[test]
    fn allows_leading_assistant_clarifying_question() {
        assert!(find_text_stop("有什么可以帮您的吗？").is_none());
        assert!(find_text_stop("请问您想了解哪方面的信息呢？").is_none());
    }

    #[test]
    fn answerable_prompt_evasive_clarification_is_guarded_for_retry() {
        let generated = "您好！您想了解什么方面的信息呢？";
        let user_prompt = "你好呀，大海里面有什么？";

        assert!(should_hold_evasive_clarification_prefix(
            "您好！",
            user_prompt
        ));
        assert!(should_stop_evasive_clarification(generated, user_prompt));
        assert!(should_retry_guarded_generation(true, generated));
    }

    #[test]
    fn vague_greeting_can_receive_help_question() {
        let generated = "有什么可以帮您的吗？";
        let user_prompt = "你好";

        assert!(!should_stop_evasive_clarification(generated, user_prompt));
        assert!(!should_hold_evasive_clarification_prefix(
            "有什么",
            user_prompt
        ));
    }

    #[test]
    fn detects_standalone_generated_question_after_complete_answer() {
        let text = "祝你节日快乐！但请注意合理消费哦~ 另外，如果你需要购买商品，请先确认自己是否有足够的余额。\n背包里有10个金币和20个钻石，可以用来购买商品吗？ 若要购买商品，首先需要确定自己的钱包中是否有足够的货币。";
        let stop = find_text_stop(text).expect("generated standalone question should stop");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "祝你节日快乐！但请注意合理消费哦~ 另外，如果你需要购买商品，请先确认自己是否有足够的余额。"
        );
    }

    #[test]
    fn detects_standalone_generated_question_without_ma_particle() {
        let text = "非常抱歉，我理解错误了。请问您有什么需要帮助的地方吗？\n背包里有十枚金币和二十枚钻石，是否可用于购买物品？ 園\n_endian\n您好，我是 AI 助手。";
        let stop = find_text_stop(text).expect("generated 是否 question should stop");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "非常抱歉，我理解错误了。请问您有什么需要帮助的地方吗？"
        );
    }

    #[test]
    fn self_dialogue_helper_recognizes_whether_question_line() {
        let text = "非常抱歉，我理解错误了。请问您有什么需要帮助的地方吗？\n背包里有十枚金币和二十枚钻石，是否可用于购买物品？ 園";
        let question_mark_pos = text.find("？ 園").expect("question mark before artifact");
        let question_start = question_start_before(text, question_mark_pos);
        let question = text[question_start..question_mark_pos + "？".len()].trim();

        assert_eq!(
            question,
            "背包里有十枚金币和二十枚钻石，是否可用于购买物品？"
        );
        assert!(looks_like_generated_question(question));
        assert!(prefix_looks_complete_before_self_question(
            &text[..question_start]
        ));
        assert!(question_starts_standalone_line(text, question_start));

        let stop = find_self_dialogue_stop(text).expect("self dialogue helper should stop");
        assert_eq!(stop.pos, question_start);
    }

    #[test]
    fn detects_endian_artifact_after_generated_question() {
        let text = "对不起，我不知道为什么会这样。\n背包里有多少金币和多少钻石？園\nEndian\n非常感谢你的提问。";
        let stop = find_text_stop(text).expect("endian artifact after self-question should stop");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(&text[..clean_end], "对不起，我不知道为什么会这样。");
    }

    #[test]
    fn assistant_history_removes_standalone_self_question_pollution() {
        let cleaned = sanitize_history_content(
            "assistant",
            "祝你节日快乐！但请注意合理消费哦~\n背包里有10个金币和20个钻石，可以用来购买商品吗？ 若要购买商品，首先需要确定自己的钱包中是否有足够的货币。",
        );

        assert_eq!(cleaned, "祝你节日快乐！但请注意合理消费哦~");
    }

    #[test]
    fn ready_buffer_holds_potential_self_question_line() {
        let text = "祝你节日快乐！但请注意合理消费哦~\n背包里有10个金币";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("祝你节日快乐！但请注意合理消费哦~".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(
            token_buf.into_iter().collect::<String>(),
            "\n背包里有10个金币"
        );
    }

    #[test]
    fn ready_buffer_holds_potential_leading_question() {
        let text = "你叫什么";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert!(rx.blocking_recv().is_none());
        assert_eq!(token_buf.into_iter().collect::<String>(), text);
    }

    #[test]
    fn ready_buffer_releases_normal_you_can_sentence() {
        let text = "你可以这样做。";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(text.to_string())));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn ready_buffer_holds_long_potential_self_question_line() {
        let text = "祝你节日快乐！但请注意合理消费哦~\n背包里有10个金币和20个钻石，可以";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(
            rx.blocking_recv(),
            Some(Some("祝你节日快乐！但请注意合理消费哦~".to_string()))
        );
        assert!(rx.blocking_recv().is_none());
        assert_eq!(
            token_buf.into_iter().collect::<String>(),
            "\n背包里有10个金币和20个钻石，可以"
        );
    }

    #[test]
    fn ready_buffer_releases_normal_second_paragraph() {
        let text = "第一段已经说完。\n第二段继续补充一个正常说明。";
        let mut token_buf = std::collections::VecDeque::from([text.to_string()]);
        let (tx, mut rx) = mpsc::channel(4);

        assert!(flush_ready_buffer(text, &mut token_buf, &tx));
        drop(tx);

        assert_eq!(rx.blocking_recv(), Some(Some(text.to_string())));
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn detects_runaway_emoji_enumeration_after_final_item() {
        let text = "我给你一个：😀\n我再给你一个：😂\n我再再给你一个: 😂😂😂\n我最后再给你一个吧：😁😁😁😁\n我再最后";
        let stop = find_text_stop(text).expect("runaway enumeration should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "我给你一个：😀\n我再给你一个：😂\n我再再给你一个: 😂😂😂\n我最后再给你一个吧：😁😁😁😁"
        );
    }

    #[test]
    fn openai_emoji_request_adds_prompt_constraint_for_model_generation() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "正常回答用户。".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "我今天心情好，给我几个可以表达出我的心情的emoji".to_string(),
            },
        ];

        let prompt = messages_to_qwen_prompt(&messages);

        assert!(prompt.contains("必须直接包含 Unicode emoji"));
        assert!(prompt.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn normal_friend_chat_prompt_does_not_include_emoji_constraint() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "我要和一个许久未见的朋友聊天，应该怎么做才能显得我礼貌，并且不会让朋友尴尬。"
                .to_string(),
        }];

        let prompt = messages_to_qwen_prompt(&messages);

        assert!(!prompt.contains("Unicode emoji"));
        assert!(!prompt.contains("不要只写中文情绪名称"));
    }

    #[test]
    fn detects_short_cjk_fragment_sequence_after_answer() {
        let text = "我给你的是一个表情符号。您需要什么类型的emoji？\n佛\n你要的emoji是哪个？佛\n发财\n佛\n发财";
        let stop = find_text_stop(text).expect("short fragment sequence should stop generation");
        let clean_end = clean_stop_prefix_len(&text[..stop.pos]);

        assert_eq!(
            &text[..clean_end],
            "我给你的是一个表情符号。您需要什么类型的emoji？"
        );
        assert_eq!(
            final_finish_reason_after_generation(true, false, text),
            "stop"
        );
    }

    #[test]
    fn guarded_empty_retry_only_for_user_echo() {
        let echoed = "我心情不太好，应该怎么才能比较好的调理？";
        assert!(should_retry_empty_guarded_generation(
            true,
            echoed,
            Some(echoed)
        ));
        assert!(!should_retry_empty_guarded_generation(
            true,
            "你叫什么名字？",
            Some("你好")
        ));
        assert!(!should_retry_empty_guarded_generation(
            true,
            "",
            Some("你好")
        ));
    }

    #[test]
    fn final_chat_process_chunk_has_finish_reason() {
        let chunk = chat_process_chunk(
            "test-id",
            "conv-id",
            "回答".to_string(),
            String::new(),
            Some("stop".to_string()),
        );
        let json = serde_json::to_value(chunk).unwrap();

        assert_eq!(json["text"], "回答");
        assert_eq!(json["conversationId"], "conv-id");
        assert_eq!(json["detail"]["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn heartbeat_chat_process_chunk_is_valid_empty_delta_ndjson() {
        let line = chat_process_chunk_line(
            "test-id",
            "conv-id",
            "已有文本".to_string(),
            String::new(),
            None,
        );
        let json: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();

        assert!(line.ends_with('\n'));
        assert_eq!(json["text"], "已有文本");
        assert_eq!(json["delta"], "");
        assert_eq!(json["conversationId"], "conv-id");
        assert!(json["detail"]["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn hit_length_limit_reports_length_for_auto_continuation() {
        assert_eq!(
            final_finish_reason_after_generation(false, true, "1.将黄豆洗净浸泡一夜。\n2.泡"),
            "length"
        );
    }

    #[test]
    fn runtime_has_no_static_assistant_answer_shortcuts() {
        let source = include_str!("server.rs");
        for (prefix, suffix) in [
            ("quick_", "answer"),
            ("canned_", "answer"),
            ("hardcoded_", "answer"),
            ("static_", "answer"),
            ("direct_", "answer_response"),
        ] {
            let needle = format!("{prefix}{suffix}");
            assert!(
                !source.contains(&needle),
                "forbidden assistant answer shortcut symbol found: {needle}"
            );
        }
    }
}
