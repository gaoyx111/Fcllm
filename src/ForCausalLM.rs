use crate::Cache::Cache;
use crate::Model::Qwen2MoeModel;
use crate::args::Args;
use crate::configuration_qwen::{Qwen2MoeConfig, get_qwen_config};
use crate::linear::new_uninitialized_linear;
use crate::load::load_linear_from_files;
use crate::utils::memory_cost_qwen;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

const GENERATION_REPETITION_PENALTY: f32 = 1.15;
const GENERATION_NO_REPEAT_NGRAM_SIZE: usize = 4;
const STATIC_KV_CACHE_MIN_CAPACITY: usize = 128;

fn tensor_debug_enabled() -> bool {
    std::env::var_os("FCLLM_DEBUG_TENSORS").is_some()
}

fn static_kv_cache_capacity(
    seq_len: usize,
    max_new_tokens: usize,
    max_position_embeddings: usize,
) -> usize {
    let capacity = seq_len
        .saturating_add(max_new_tokens)
        .saturating_add(1)
        .min(max_position_embeddings);
    if capacity >= STATIC_KV_CACHE_MIN_CAPACITY {
        capacity
    } else {
        0
    }
}

fn select_next_token_with_repetition_controls(
    logits: &[f32],
    context_ids: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
) -> u32 {
    let mut scores = logits.to_vec();
    apply_repetition_penalty(&mut scores, context_ids, repetition_penalty);

    for token_id in banned_no_repeat_tokens(context_ids, no_repeat_ngram_size) {
        if let Some(score) = scores.get_mut(token_id as usize) {
            *score = f32::NEG_INFINITY;
        }
    }

    argmax_score(&scores).unwrap_or(0)
}

fn select_next_token_from_logits(
    logits: &Tensor,
    generated_ids: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
) -> Result<u32> {
    let raw_argmax = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
    if generated_ids.is_empty() || !generated_ids.contains(&raw_argmax) {
        return Ok(raw_argmax);
    }

    let scores = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(select_next_token_with_repetition_controls(
        &scores,
        generated_ids,
        repetition_penalty,
        no_repeat_ngram_size,
    ))
}

fn apply_repetition_penalty(scores: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }

    let mut seen = HashSet::new();
    for &token_id in token_ids {
        if !seen.insert(token_id) {
            continue;
        }

        let Some(score) = scores.get_mut(token_id as usize) else {
            continue;
        };

        if *score > 0.0 {
            *score /= penalty;
        } else {
            *score *= penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_penalty_can_move_selection_off_repeated_token() {
        let logits = vec![1.0, 2.0, 1.9];

        let selected = select_next_token_with_repetition_controls(&logits, &[1, 1, 1], 1.15, 0);

        assert_eq!(selected, 2);
    }

    #[test]
    fn no_repeat_ngram_bans_token_that_would_repeat_context() {
        let mut logits = vec![0.0; 20];
        logits[12] = 10.0;
        logits[13] = 9.0;

        let selected =
            select_next_token_with_repetition_controls(&logits, &[10, 11, 12, 10, 11], 1.0, 3);

        assert_eq!(banned_no_repeat_tokens(&[10, 11, 12, 10, 11], 3), vec![12]);
        assert_eq!(selected, 13);
    }

    #[test]
    fn tensor_argmax_fast_path_handles_unseen_argmax() -> Result<()> {
        let logits = Tensor::from_vec(vec![1.0f32, 3.0, 2.9], (3,), &Device::Cpu)?;
        let selected =
            select_next_token_from_logits(&logits, &[2, 2], GENERATION_REPETITION_PENALTY, 0)?;

        assert_eq!(selected, 1);
        Ok(())
    }

    #[test]
    fn tensor_argmax_fast_path_falls_back_for_repeated_argmax() -> Result<()> {
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 1.9], (3,), &Device::Cpu)?;
        let selected =
            select_next_token_from_logits(&logits, &[1, 1], GENERATION_REPETITION_PENALTY, 0)?;

        assert_eq!(selected, 2);
        Ok(())
    }

    #[test]
    fn static_kv_cache_skips_tiny_contexts_but_handles_normal_chat() {
        assert_eq!(static_kv_cache_capacity(16, 16, 8192), 0);
        assert_eq!(static_kv_cache_capacity(64, 64, 8192), 129);
        assert_eq!(static_kv_cache_capacity(900, 200, 8192), 1101);
        assert_eq!(static_kv_cache_capacity(8000, 512, 8192), 8192);
    }
}

fn banned_no_repeat_tokens(token_ids: &[u32], ngram_size: usize) -> Vec<u32> {
    if ngram_size < 2 || token_ids.len() + 1 < ngram_size {
        return Vec::new();
    }

    let prefix_len = ngram_size - 1;
    let current_prefix = &token_ids[token_ids.len() - prefix_len..];
    let mut banned = Vec::new();

    for window in token_ids.windows(ngram_size) {
        if &window[..prefix_len] == current_prefix {
            let token_id = window[prefix_len];
            if !banned.contains(&token_id) {
                banned.push(token_id);
            }
        }
    }

    banned
}

fn argmax_score(scores: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;

    for (idx, &score) in scores.iter().enumerate() {
        if score.is_nan() {
            continue;
        }

        match best {
            Some((_, best_score)) if score <= best_score => {}
            _ => best = Some((idx, score)),
        }
    }

    best.map(|(idx, _)| idx as u32)
}

#[derive(Debug, Clone)]
pub struct Qwen2MoeForCausalLM {
    pub config: Qwen2MoeConfig,
    pub device: Device,
    pub min_length: usize,
    pub max_length: usize,
    pub early_stopping: bool,
    pub path: String,
    pub model: Qwen2MoeModel,
    pub vocab_size: usize,
    pub lm_head: Linear,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl Qwen2MoeForCausalLM {
    pub fn new(args: Args) -> Result<Self> {
        let mut config = get_qwen_config(&args.model)?;
        config.device = if args.device.starts_with("cuda") {
            let index: usize = args
                .device
                .split(':')
                .nth(1)
                .unwrap_or("0")
                .parse()
                .unwrap_or(0);
            Device::cuda_if_available(index)?
        } else {
            Device::Cpu
        };
        let device = config.device.clone();
        config._attn_implementation = "sdpa".to_string();

        let (offload_map, quan_map) = memory_cost_qwen(&config, args.memory_budget);
        config.offload_map = offload_map;
        config.quan_map = quan_map;

        println!("offload: {:?}", config.offload_map);
        println!("quan_map: {:?}", config.quan_map);

        let model = Qwen2MoeModel::new(&config)?;
        let lm_head =
            new_uninitialized_linear(config.hidden_size, config.vocab_size, false, &device)?;

        let path_str = args.path.to_str().expect("invalid UTF-8").to_owned();
        let n1 = config.num_experts;
        let n2 = config.num_experts_per_tok;
        let n3 = config.vocab_size;

        Ok(Self {
            config: config,
            device: device,
            min_length: args.min_length,
            max_length: args.max_length,
            early_stopping: args.early_stopping,
            path: path_str,
            model: model,
            vocab_size: n3,
            lm_head: lm_head,
            num_experts: n1,
            num_experts_per_tok: n2,
        })
    } // 注意new函数里没有init_weights!!!记得在main函数里new模型之后加上init_weights!!!

    pub fn init_weights(&mut self) -> Result<()> {
        let base = PathBuf::from(self.path.clone());
        // 拼接出 expanded_path: path/Qwen/Qwen1.5-MoE-A2.7B
        let expanded_path = base.join("Qwen").join("Qwen1.5-MoE-A2.7B");

        // 拼接 check_path: path/Qwen/Qwen1.5-MoE-A2.7B/original/lm_head.weight
        let check_path = expanded_path.join("original").join("lm_head.weight");

        // 若文件不存在，则下载
        if !check_path.exists() {
            println!("check_path: {:?}", check_path);
            panic!("lm_head.weight not found, please check the path or download");
            // download_qwen_weights("Qwen/Qwen1.5-MoE-A2.7B", &self.args.path)?;
        }

        // 加载主模型权重
        self.model.init_weights(expanded_path.to_str().unwrap())?;
        // 加载 lm_head 权重
        self.lm_head =
            load_linear_from_files(check_path.to_str().unwrap(), None, &self.config.device)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mut past_key_values: Option<Cache>,
        inputs_embeds: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        num_logits_to_keep: usize,
    ) -> Result<(Tensor, Option<Cache>)> {
        let (hidden_states, new_past_key_values) = self.model.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            cache_position,
        )?;

        let logits_input = if num_logits_to_keep > 0 {
            let seq_len = hidden_states.dim(1)?;
            hidden_states.narrow(1, seq_len - num_logits_to_keep, num_logits_to_keep)?
        } else {
            hidden_states
        };

        // One-shot diagnostics on the first (prefill) call
        if tensor_debug_enabled() && logits_input.dim(1)? > 1 {
            let li_f32 = logits_input.to_dtype(DType::F32)?;
            let li_abs_max = li_f32.flatten_all()?.abs()?.max(0)?;
            eprintln!(
                "[DEBUG] hidden_states → lm_head input: shape={:?}, dtype={:?}, abs_max={:?}",
                logits_input.shape(),
                logits_input.dtype(),
                li_abs_max.to_scalar::<f32>()
            );
            let w = self.lm_head.weight().to_dtype(DType::F32)?;
            let w_abs_max = w.flatten_all()?.abs()?.max(0)?;
            eprintln!(
                "[DEBUG] lm_head weight: shape={:?}, dtype={:?}, abs_max={:?}",
                self.lm_head.weight().shape(),
                self.lm_head.weight().dtype(),
                w_abs_max.to_scalar::<f32>()
            );
        }

        let logits = self.lm_head.forward(&logits_input)?;

        Ok((logits, new_past_key_values))
    }

    /// 流式生成：每产出一个 token 就调用 callback
    ///
    /// callback 参数是 token id (u32)，返回 true 继续生成，返回 false 立即停止
    /// （客户端断开连接时 channel send 失败，callback 返回 false，自动停止推理）
    pub fn generate_streaming<F>(
        &mut self,
        input_ids: &Tensor,
        _attention_mask: Option<Tensor>,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(u32) -> bool,
    {
        let seq_len = input_ids.dim(1)?;
        let device = self.device.clone();
        let cache_capacity = static_kv_cache_capacity(
            seq_len,
            self.max_length,
            self.config.max_position_embeddings,
        );
        eprintln!(
            "[kv-cache] mode={} seq_len={} max_new_tokens={} capacity={}",
            if cache_capacity > 0 {
                "static-scatter"
            } else {
                "dynamic-cat"
            },
            seq_len,
            self.max_length,
            cache_capacity
        );
        let mut past_key_values = Some(Cache::new_with_capacity(
            true,
            input_ids.dtype(),
            &self.config,
            cache_capacity,
        )?);

        let mut position_ids = if seq_len > 1 {
            Some(Tensor::arange(0u32, seq_len as u32, &device)?.reshape((1, seq_len))?)
        } else {
            None
        };
        let mut cache_position = if seq_len > 1 {
            Some(Tensor::arange(0u32, seq_len as u32, &device)?.reshape((seq_len,))?)
        } else {
            None
        };

        let mut input_ids = input_ids.clone();
        let mut generated_ids = Vec::with_capacity(self.max_length);

        let generation_start = Instant::now();

        for _i in 0..self.max_length {
            let iter_start = Instant::now();
            // The model currently builds causal masking inside attention and does not
            // consume the 2-D tokenizer attention mask, so avoid growing it every token.
            let (logits, new_cache) = self.forward(
                Some(&input_ids),
                None,
                position_ids.as_ref(),
                past_key_values,
                None,
                cache_position.as_ref(),
                1,
            )?;
            let forward_elapsed = iter_start.elapsed();

            past_key_values = new_cache;

            let last_logits = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?;
            let tok_id = select_next_token_from_logits(
                &last_logits,
                &generated_ids,
                GENERATION_REPETITION_PENALTY,
                GENERATION_NO_REPEAT_NGRAM_SIZE,
            )?;
            let last_token = Tensor::new(&[tok_id as i64], &device)?.reshape((1, 1))?;

            // 遇到任意停止 token 就结束（含 <|im_end|>=151645、<|im_start|>=151644、<|endoftext|>=151643）
            if self.config.stop_token_ids.contains(&tok_id) {
                eprintln!(
                    "[gen] stop_token_ids 命中: tok_id={} 第 {} 个 token",
                    tok_id, _i
                );
                break;
            }
            // 调试：打印前 5 个 token id，方便核实模型输出
            if _i < 5 {
                eprintln!("[gen debug] iter={} tok_id={}", _i, tok_id);
            }
            if _i < 8 || _i % 16 == 0 {
                eprintln!(
                    "[gen timing] iter={} forward={:.2?} total={:.2?}",
                    _i,
                    forward_elapsed,
                    generation_start.elapsed()
                );
            }

            // 调用回调；若返回 false（通道关闭）则停止生成
            if !callback(tok_id) {
                break;
            }
            generated_ids.push(tok_id);

            // 准备下一轮输入
            input_ids = last_token;
            position_ids = None;
            cache_position = None;
        }

        Ok(())
    }

    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        _attention_mask: Option<Tensor>,
        experiment_mode: Option<&str>,
    ) -> Result<(Tensor, f64)> {
        let mut prefill_time = 0f64;
        // if let Some("decoding") = experiment_mode {
        //     prefill_time = std::time::Instant::now().elapsed().as_secs_f64(); // 起始计时
        // }

        let seq_len = input_ids.dim(1)?; // (1, seq_len)

        let device = self.device.clone();
        let cache_capacity =
            static_kv_cache_capacity(seq_len, 1024, self.config.max_position_embeddings);
        eprintln!(
            "[kv-cache] mode={} seq_len={} max_new_tokens={} capacity={}",
            if cache_capacity > 0 {
                "static-scatter"
            } else {
                "dynamic-cat"
            },
            seq_len,
            1024,
            cache_capacity
        );
        let mut past_key_values = Some(Cache::new_with_capacity(
            true,
            input_ids.dtype(),
            &self.config,
            cache_capacity,
        )?);

        // 初始位置 id 和 cache_position
        let mut position_ids = if seq_len > 1 {
            Some(Tensor::arange(0u32, seq_len as u32, &device)?.reshape((1, seq_len))?)
        } else {
            None
        };
        let mut cache_position = if seq_len > 1 {
            Some(Tensor::arange(0u32, seq_len as u32, &device)?.reshape((seq_len,))?)
        } else {
            None
        };

        let mut input_ids = input_ids.clone();
        let mut generated_ids = Vec::with_capacity(1024);
        let mut output_ids: Vec<i64> = Vec::with_capacity(1024);

        for i in 0..1024 {
            // The model currently builds causal masking inside attention and does not
            // consume the 2-D tokenizer attention mask, so avoid growing it every token.
            let (logits, new_cache) = self.forward(
                Some(&input_ids),
                None,
                position_ids.as_ref(),
                past_key_values,
                None,
                cache_position.as_ref(),
                1,
            )?;

            past_key_values = new_cache;

            // 检查第一次迭代的 logits 是否有 NaN
            if tensor_debug_enabled() && i == 0 {
                let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
                let max_val = last_logits.max(candle_core::D::Minus1)?;
                let min_val = last_logits.min(candle_core::D::Minus1)?;
                let raw_argmax = last_logits.argmax(candle_core::D::Minus1)?;
                eprintln!(
                    "[DEBUG] first iter logits: max={:?}, min={:?}, argmax={:?}, shape={:?}",
                    max_val.to_vec1::<f32>().unwrap_or_default(),
                    min_val.to_vec1::<f32>().unwrap_or_default(),
                    raw_argmax.to_vec1::<u32>().unwrap_or_default(),
                    logits.shape()
                );
            }

            let last_logits = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?;
            let tok_id = select_next_token_from_logits(
                &last_logits,
                &generated_ids,
                GENERATION_REPETITION_PENALTY,
                GENERATION_NO_REPEAT_NGRAM_SIZE,
            )?;
            let next_token = Tensor::new(&[tok_id as i64], &device)?.reshape((1, 1))?;
            output_ids.push(tok_id as i64);

            input_ids = next_token;

            // 早停
            // if self.early_stopping && i > self.min_length && input_ids.to_vec1::<i64>()?[0] == self.config.eos_token_id as i64 {
            //     return Ok((output.unwrap(), prefill_time));
            // }

            if i < 5 || i % 50 == 0 {
                eprintln!("[token {}] id={}", i, tok_id);
            }
            generated_ids.push(tok_id);
            // 同样检查所有停止 token（<|im_end|> / <|im_start|> / <|endoftext|>）
            if self.config.stop_token_ids.contains(&tok_id) {
                let output =
                    Tensor::new(output_ids.as_slice(), &device)?.reshape((1, output_ids.len()))?;
                return Ok((output, prefill_time));
            }

            // 准备下一轮的位置 id / cache pos
            position_ids = None;
            cache_position = None;
        }

        let output = Tensor::new(output_ids.as_slice(), &device)?.reshape((1, output_ids.len()))?;
        Ok((output, prefill_time))
    }
}
