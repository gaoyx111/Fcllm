/// HTTP 服务器模块
/// 提供与 chatgpt-web 及 OpenAI 兼容的接口，支持流式输出
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use uuid::Uuid;

use tokenizers::Tokenizer;
use candle_core::Tensor;

use crate::ForCausalLM::Qwen2MoeForCausalLM;

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
}

// ═══════════════════════════════════════════════════════════
//  模型工作线程通信通道
// ═══════════════════════════════════════════════════════════

/// 每次推理请求的结构体，由 HTTP handler 发往模型工作线程
pub struct InferRequest {
    /// 格式化好的 prompt 字符串（已应用 Qwen chat template）
    pub prompt: String,
    /// 最大生成 token 数
    pub max_tokens: usize,
    /// 每生成一个 token 就往这里发送解码后的文本；None 表示生成结束
    pub token_tx: mpsc::Sender<Option<String>>,
}

/// 请求发送端类型（在 axum State 中共享）
pub type ModelReqSender = mpsc::Sender<InferRequest>;

/// axum 共享状态
#[derive(Clone)]
pub struct AppState {
    pub req_tx: ModelReqSender,
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
        // OpenAI 兼容接口
        .route("/v1/chat/completions", post(openai_handler))
        .route("/v1/models", get(models_handler))
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
    let prompt = messages_to_qwen_prompt(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(512);

    if req.stream {
        sse_response(state, prompt, max_tokens, false).await
    } else {
        blocking_response(state, prompt, max_tokens).await
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
    // 系统指令：明确角色身份 + few-shot 示例教 Base 模型自行生成 <|im_end|> 停止
    let system_default = "\
你是Fcllm，一个本地运行的AI语言助手。你是AI，用户是真实的人类，不要混淆角色。\
请直接、简洁地回答用户的问题，给出完整答案后立即停止输出。";
    let effective_system = if system.is_empty() { system_default } else { system };

    // few-shot 示例：展示一轮完整对话，让 Base 模型学会在回答后生成 <|im_end|>
    // <|im_end|> 是 token 151645，会被 stop_token_ids 检测到并立即终止生成
    let prompt = format!(
        "<|im_start|>system\n{effective_system}<|im_end|>\n\
         <|im_start|>user\n你好，你是谁？<|im_end|>\n\
         <|im_start|>assistant\n你好！我是Fcllm，一个本地运行的AI语言助手。有什么我可以帮你的吗？<|im_end|>\n\
         <|im_start|>user\n{user_prompt}<|im_end|>\n\
         <|im_start|>assistant\n",
        effective_system = effective_system,
        user_prompt = req.prompt,
    );

    ndjson_stream_response(state, prompt, 400).await
}

/// 构建 NDJSON 流式响应（chatgpt-web 格式）
/// 每个 token 写一行 JSON，前端通过 Axios onDownloadProgress 增量读取
async fn ndjson_stream_response(state: AppState, prompt: String, max_tokens: usize) -> Response {
    let (token_tx, token_rx) = mpsc::channel::<Option<String>>(256);

    if state
        .req_tx
        .send(InferRequest { prompt, max_tokens, token_tx })
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
    let accumulated = Arc::new(std::sync::Mutex::new(String::new()));
    let id_c = id.clone();
    let acc_c = accumulated.clone();

    // 把 token 流转换为 NDJSON 行流
    let ndjson_stream = ReceiverStream::new(token_rx).map(move |token_opt| {
        let line = match token_opt {
            Some(ref text) => {
                let mut acc = acc_c.lock().unwrap();
                acc.push_str(text);
                let chunk = ChatProcessChunk {
                    role: "assistant".to_string(),
                    id: id_c.clone(),
                    text: acc.clone(),       // 累积全文
                    delta: text.clone(),     // 本次新增
                    parent_message_id: id_c.clone(),
                    conversation_id: id_c.clone(),
                };
                let mut s = serde_json::to_string(&chunk).unwrap_or_default();
                s.push('\n');  // 每行结束加换行，供前端增量解析
                s
            }
            // 生成结束：发一个空行让前端知道流结束
            None => String::new(),
        };
        Ok::<String, Infallible>(line)
    });

    Response::builder()
        .header("content-type", "application/octet-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no") // 禁用 nginx 缓冲（如有反代）
        .body(Body::from_stream(ndjson_stream))
        .unwrap_or_else(|_| {
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "stream error").into_response()
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
    max_tokens: usize,
    chatgpt_web_fmt: bool,
) -> Response {
    let (token_tx, token_rx) = mpsc::channel::<Option<String>>(256);

    // 把请求发给模型工作线程
    if state
        .req_tx
        .send(InferRequest {
            prompt,
            max_tokens,
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

    let stream = ReceiverStream::new(token_rx).map(move |token_opt| {
        let data = match token_opt {
            Some(ref text) => {
                if chatgpt_web_fmt {
                    // chatgpt-web 格式：每次推送累积文本 + 本次 delta
                    let mut acc = acc_c.lock().unwrap();
                    acc.push_str(text);
                    let chunk = ChatProcessChunk {
                        role: "assistant".to_string(),
                        id: id_c.clone(),
                        text: acc.clone(),
                        delta: text.clone(),
                        parent_message_id: id_c.clone(),
                        conversation_id: id_c.clone(),
                    };
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
            None => "[DONE]".to_string(),
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

async fn blocking_response(state: AppState, prompt: String, max_tokens: usize) -> Response {
    let (token_tx, mut token_rx) = mpsc::channel::<Option<String>>(256);

    if state
        .req_tx
        .send(InferRequest {
            prompt,
            max_tokens,
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
            finish_reason: "stop".to_string(),
        }],
    };

    Json(resp).into_response()
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
    eprintln!("[worker] 模型工作线程启动，等待请求...");

    while let Some(req) = req_rx.blocking_recv() {
        let InferRequest {
            prompt,
            max_tokens,
            token_tx,
        } = req;

        eprintln!("[worker] 收到请求，prompt 长度 = {} chars", prompt.len());

        // 1. Tokenize 输入
        let encoding = match tokenizer.encode(prompt.as_str(), true) {
            Ok(enc) => enc,
            Err(e) => {
                eprintln!("[worker] tokenize 失败: {e}");
                let _ = token_tx.blocking_send(None);
                continue;
            }
        };

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attn_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let device = model.device.clone();

        let input_tensor = match Tensor::new(input_ids.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[worker] 创建 input tensor 失败: {e}");
                let _ = token_tx.blocking_send(None);
                continue;
            }
        };

        let attn_tensor = match Tensor::new(attn_mask.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[worker] 创建 attention_mask tensor 失败: {e}");
                let _ = token_tx.blocking_send(None);
                continue;
            }
        };

        // 2. 临时覆盖 max_length
        let orig_max = model.max_length;
        model.max_length = max_tokens;

        // 3. 流式生成，每个 token 解码后通过 channel 发出
        let tk = tokenizer.clone();

        // ══ 三层保护 ═══════════════════════════════════════════════════════
        // Layer-1  UTF-8 完整性：积累多个 token ID 一起解码，防止字节级
        //          fallback token 被逐个解码时产生 U+FFFD 乱码。
        //          （Qwen BPE 对词表外字符拆成 3 个单字节 token，需合并解码）
        //
        // Layer-2  文本停止词：检测新一轮对话角色标记，立即中止生成。
        //          同时覆盖简体/繁体两种"用户"写法。
        //
        // Layer-3  前瞻缓冲：延迟 BUFFER_AHEAD 个"文本块"发送，确保
        //          停止词本身不出现在前端输出中。
        // ═══════════════════════════════════════════════════════════════════
        const MAX_ID_ACC:   usize = 8;  // 最多积累多少个 token ID 再强制解码
        const BUFFER_AHEAD: usize = 10; // 前瞻发送缓冲块数（增大以拦截回答后的多余词汇）
        const TEXT_STOPS: &[&str] = &[
            "用户\n",  "用户：",   // 简体中文：用户角色
            "用戶\n",  "用戶：",   // 繁体中文：用戶角色
            "\n用户",  "\n用戶",   // 换行后出现角色名
            "\nUser:", "\nHuman:", // 英文角色标记
            "\n问：",              // 中文 Q&A 格式
        ];

        let mut id_acc:     Vec<u32>                            = Vec::new();
        let mut text_acc:   String                              = String::with_capacity(512);
        let mut token_buf:  std::collections::VecDeque<String> = std::collections::VecDeque::new();
        let mut stopped_by_text = false;

        let result = model.generate_streaming(
            &input_tensor,
            Some(attn_tensor),
            |tok_id| {
                // ── Layer-1：累积 token ID，直到解码结果不含替换字符 ──────
                id_acc.push(tok_id);
                let raw = tk.decode(&id_acc, true).unwrap_or_default();
                let has_bad = raw.chars().any(|c| c == '\u{FFFD}');

                if has_bad && id_acc.len() < MAX_ID_ACC {
                    return true; // 字节序列还不完整，继续积累
                }

                // 字节序列完整（或已达上限）→ 确定文本
                id_acc.clear();
                let text = if has_bad {
                    // 超出上限仍有乱码：丢弃无法解码的字节残留
                    raw.replace('\u{FFFD}', "")
                } else {
                    raw
                };
                if text.is_empty() { return true; }

                // ── Layer-2：重复输出检测 ────────────────────────────────
                // Base 模型容易陷入重复循环（菜单、emoji 等），
                // 在字节级滑动窗口检测到同一模式连续出现 3 次时立即停止。
                // ────────────────────────────────────────────────────────
                text_acc.push_str(&text);
                token_buf.push_back(text);

                // ── 全窗口 n-gram 重复检测 ─────────────────────────────
                // 旧的"尾部匹配"方法无法检测语义重复（如"您有什么X吗？"
                // 每句末尾不同但前缀相同）。
                // 新方法：在近期 400 字节窗口中，以字节级步长扫描所有
                // 候选 n-gram，若任意 n-gram 出现 4+ 次则判定为重复循环。
                {
                    let bytes = text_acc.as_bytes();
                    let n = bytes.len();
                    if n >= 160 {
                        // 窗口缩至 200 字节：让检测更局部、对高密度重复更敏感
                        // 避免"，比如"（6字节）在正常长列表中跨 200+ 字节出现 4 次的误判
                        let window_start = n.saturating_sub(200);
                        let window = &bytes[window_start..n];
                        let wn = window.len();
                        let mut rep_found = false;
                        // 最小 gram 12字节 = 4个汉字（如"您有什么"），
                        // 过滤掉"，比如"（6字节）、"等等"（6字节）等合法高频短词
                        'gram_scan: for &gram_size in &[12usize, 15, 18, 21, 27] {
                            if wn < gram_size * 4 { continue; }
                            // 只在窗口末尾 gram_size×5 字节内扫描候选 gram
                            let cand_start = wn.saturating_sub(gram_size * 5);
                            let mut gi = cand_start;
                            while gi + gram_size <= wn {
                                let gram = &window[gi..gi + gram_size];
                                let mut count = 0usize;
                                let mut pi = 0;
                                while pi + gram_size <= wn {
                                    if &window[pi..pi + gram_size] == gram {
                                        count += 1;
                                        if count >= 4 {
                                            eprintln!(
                                                "[worker] n-gram 重复: {}字节模式×{}次，停止生成",
                                                gram_size, count
                                            );
                                            rep_found = true;
                                            break 'gram_scan;
                                        }
                                    }
                                    pi += 1;
                                }
                                gi += 1; // 字节级步长，覆盖所有对齐方式
                            }
                        }
                        if rep_found {
                            stopped_by_text = true;
                            return false;
                        }
                    }
                }

                // ── Layer-3：文本停止词 + 前瞻缓冲 ─────────────────────
                for &stop in TEXT_STOPS {
                    if let Some(stop_pos) = text_acc.find(stop) {
                        let buf_len:  usize = token_buf.iter().map(|s| s.len()).sum();
                        let buf_start = text_acc.len().saturating_sub(buf_len);
                        let safe_end  = stop_pos.saturating_sub(buf_start);

                        // 发送停止词之前的缓冲内容
                        let mut consumed = 0usize;
                        while consumed < safe_end && !token_buf.is_empty() {
                            let tok_len = token_buf[0].len();
                            if consumed + tok_len <= safe_end {
                                let tok = token_buf.pop_front().unwrap();
                                consumed += tok_len;
                                if token_tx.blocking_send(Some(tok)).is_err() {
                                    stopped_by_text = true;
                                    return false;
                                }
                            } else {
                                break;
                            }
                        }

                        eprintln!(
                            "[worker] 文本停止词命中: {:?}，已生成 {} 字符",
                            stop, text_acc.len()
                        );
                        stopped_by_text = true;
                        return false;
                    }
                }

                // 无停止词：将超出前瞻窗口的旧块发送出去
                while token_buf.len() > BUFFER_AHEAD {
                    let oldest = token_buf.pop_front().unwrap();
                    if token_tx.blocking_send(Some(oldest)).is_err() {
                        return false;
                    }
                }
                true
            },
        );

        // 生成结束后：先处理 id_acc 里可能残留的字节 token
        if !id_acc.is_empty() {
            let tail = tk.decode(&id_acc, true).unwrap_or_default().replace('\u{FFFD}', "");
            if !tail.is_empty() {
                text_acc.push_str(&tail);
                token_buf.push_back(tail);
            }
        }

        // 若是正常结束（EOS / stop_token_ids / max_length），把前瞻缓冲全部发出
        if !stopped_by_text {
            for tok in token_buf.drain(..) {
                let _ = token_tx.blocking_send(Some(tok));
            }
        }

        model.max_length = orig_max;

        if let Err(e) = result {
            eprintln!("[worker] 生成过程出错: {e}");
        }

        // 4. 发送结束信号
        let _ = token_tx.blocking_send(None);
        eprintln!("[worker] 本次生成完成");
    }

    eprintln!("[worker] 请求通道已关闭，工作线程退出");
}

// ═══════════════════════════════════════════════════════════
//  辅助函数
// ═══════════════════════════════════════════════════════════

/// 将 OpenAI messages 转换为 Qwen chat template 格式的 prompt
pub fn messages_to_qwen_prompt(messages: &[ChatMessage]) -> String {
    let mut buf = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => buf.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n",
                msg.content
            )),
            "user" => buf.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n",
                msg.content
            )),
            "assistant" => buf.push_str(&format!(
                "<|im_start|>assistant\n{}<|im_end|>\n",
                msg.content
            )),
            _ => buf.push_str(&msg.content),
        }
    }
    // 末尾追加 assistant 开头，引导模型生成回复
    buf.push_str("<|im_start|>assistant\n");
    buf
}

/// 返回当前 Unix 时间戳（秒）
fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
