use candle_core::{Device, Error};
use std::collections::HashMap;


#[derive(Debug, Clone)]
pub struct Qwen2MoeConfig {
    pub model_type: String,
    
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub rope_theta: f32,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub attention_dropout: f64,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts_per_tok: usize,
    pub num_experts: usize,
    pub norm_topk_prob: bool,
    pub output_router_logits: bool,
    pub router_aux_loss_coef: f64,
    pub mlp_only_layers: Vec<usize>,
    pub eos_token_id: usize,
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    /// 生成时遇到这些 token 就停止（包含 <|im_end|> 等对话结束标记）
    pub stop_token_ids: Vec<u32>,

    // 扩展字段
    pub offload: bool,
    pub device: Device,
    pub _attn_implementation: String,
    pub offload_map: HashMap<usize, usize>,
    pub quan_map: HashMap<usize, usize>,
    pub memory_budget: usize,
}

impl Qwen2MoeConfig {
    pub fn new() -> Self {
        Qwen2MoeConfig {
            model_type: "qwen2_moe".to_string(),
            vocab_size: 151_936,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 8192,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            rope_theta: 1_000_000.0,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: 21,
            attention_dropout: 0.0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 1408,
            shared_expert_intermediate_size: 5632,
            num_experts_per_tok: 4,
            num_experts: 60,
            norm_topk_prob: false,
            output_router_logits: false,
            router_aux_loss_coef: 0.001,
            mlp_only_layers: vec![],
            eos_token_id: 151_643,
            bos_token_id: 151_643,
            pad_token_id: 151_643,
            // Qwen1.5 对话模式停止符：
            //   151643 = <|endoftext|>（文档结束）
            //   151644 = <|im_start|>（新一轮对话开始，此时应当立即停止）
            //   151645 = <|im_end|>   （助手这轮回答结束，最关键！）
            stop_token_ids: vec![151_643, 151_644, 151_645],
            offload: true,
            device: /* Device::cuda_if_available(0).unwrap_or(Device::Cpu) */ Device::Cpu,
            _attn_implementation: "sdpa".to_string(),
            offload_map: HashMap::new(),
            quan_map: HashMap::new(),
            memory_budget: 0, // GB
        }
    }

    pub fn d_ff(&self) -> usize {
        self.intermediate_size
    }
}

pub fn get_qwen_config(name: &str) -> Result<Qwen2MoeConfig, Error> {
    let name = name
        .split('/')
        .last()
        .unwrap_or(name)
        .to_ascii_lowercase();

    match name.as_str() {
        "qwen1.5-moe-a2.7b" => Ok(Qwen2MoeConfig::new()),
        _ => Err(Error::msg(format!("Invalid model name: {}", name))),
    }
}