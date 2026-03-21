use crate::configuration_qwen::{Qwen2MoeConfig, get_qwen_config};
use crate::Model::Qwen2MoeModel;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D, Error, CudaDevice};
use candle_nn::{Module, Linear};
use crate::linear::new_uninitialized_linear;
use crate::load::load_linear_from_files;
use crate::Cache::Cache;
use crate::utils::memory_cost_qwen;
use std::path::PathBuf;
use crate::args::Args;



#[derive(Debug)]
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
            let index: usize = args.device.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
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
        let lm_head = new_uninitialized_linear(config.hidden_size, config.vocab_size, false, &device)?;

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
    }// 注意new函数里没有init_weights!!!记得在main函数里new模型之后加上init_weights!!!

    pub fn init_weights(&mut self) -> Result<()> {
        let base = PathBuf::from(self.path.clone());
        // 拼接出 expanded_path: path/Qwen/Qwen1.5-MoE-A2.7B
        let expanded_path = base
            .join("Qwen")
            .join("Qwen1.5-MoE-A2.7B");

        // 拼接 check_path: path/Qwen/Qwen1.5-MoE-A2.7B/original/lm_head.weight
        let check_path = expanded_path
            .join("original")
            .join("lm_head.weight");

        // 若文件不存在，则下载
        if !check_path.exists() {
            println!("check_path: {:?}", check_path);
            panic!("lm_head.weight not found, please check the path or download");
            // download_qwen_weights("Qwen/Qwen1.5-MoE-A2.7B", &self.args.path)?;
        }

        // 加载主模型权重
        self.model.init_weights(expanded_path.to_str().unwrap())?;
        // 加载 lm_head 权重
        self.lm_head = load_linear_from_files(check_path.to_str().unwrap(),  None, &self.config.device)?;

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
        if logits_input.dim(1)? > 1 {
            let li_f32 = logits_input.to_dtype(DType::F32)?;
            let li_abs_max = li_f32.flatten_all()?.abs()?.max(0)?;
            eprintln!("[DEBUG] hidden_states → lm_head input: shape={:?}, dtype={:?}, abs_max={:?}",
                logits_input.shape(), logits_input.dtype(), li_abs_max.to_scalar::<f32>());
            let w = self.lm_head.weight().to_dtype(DType::F32)?;
            let w_abs_max = w.flatten_all()?.abs()?.max(0)?;
            eprintln!("[DEBUG] lm_head weight: shape={:?}, dtype={:?}, abs_max={:?}",
                self.lm_head.weight().shape(), self.lm_head.weight().dtype(), w_abs_max.to_scalar::<f32>());
        }

        let logits = self.lm_head.forward(&logits_input)?;

        Ok((logits, new_past_key_values))
    }


    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        mut attention_mask: Option<Tensor>,
        experiment_mode: Option<&str>,
    ) -> Result<(Tensor, f64)> {
        let mut prefill_time = 0f64;
        // if let Some("decoding") = experiment_mode {
        //     prefill_time = std::time::Instant::now().elapsed().as_secs_f64(); // 起始计时
        // }

        let mut past_key_values = Some(Cache::new(true, input_ids.dtype(), &self.config)?);
        let seq_len = input_ids.dim(1)?; // (1, seq_len)

        let device = self.device.clone();

        // 如果 attention_mask 为空，则填充为全 1
        let mut attention_mask = match attention_mask {
            Some(mask) => mask,
            None => Tensor::ones((1, seq_len), DType::U32, &device)?.to_dtype(DType::I64)?,
        };

        // 初始位置 id 和 cache_position
        let mut position_ids =
            Tensor::arange(0u32, seq_len as u32, &device)?.reshape((1, seq_len))?;
        let mut cache_position =
            Tensor::arange(0u32, seq_len as u32, &device)?.reshape((seq_len,))?;

        let mut input_ids = input_ids.clone();
        let mut output = None;

        for i in 0..1024 {
            let (logits, new_cache) = self.forward(
                Some(&input_ids),
                Some(&attention_mask),
                Some(&position_ids),
                past_key_values,
                None,
                Some(&cache_position),
                0,
            )?;

            past_key_values = new_cache;

            let dim = logits.dims().len() - 1;

            // 检查第一次迭代的 logits 是否有 NaN
            if i == 0 {
                let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
                let max_val = last_logits.max(candle_core::D::Minus1)?;
                let min_val = last_logits.min(candle_core::D::Minus1)?;
                let raw_argmax = last_logits.argmax(candle_core::D::Minus1)?;
                eprintln!("[DEBUG] first iter logits: max={:?}, min={:?}, argmax={:?}, shape={:?}",
                    max_val.to_vec1::<f32>().unwrap_or_default(),
                    min_val.to_vec1::<f32>().unwrap_or_default(),
                    raw_argmax.to_vec1::<u32>().unwrap_or_default(),
                    logits.shape());
            }

            let logits = candle_nn::ops::softmax(&logits, dim)?;

            // 贪婪搜索：取最大概率的 token
            let dim = logits.dims().len() - 1;
            let next_token = logits.argmax(dim)?;

            if i == 0 {
                output = Some(next_token.narrow(1, next_token.dim(1)? - 1, 1)?);
            } else {
                output = Some(Tensor::cat(&[&output.unwrap(), &next_token], 1)?);
            }

            input_ids = next_token.narrow(1, next_token.dim(1)? - 1, 1)?.to_dtype(DType::I64)?;

            // 早停
            // if self.early_stopping && i > self.min_length && input_ids.to_vec1::<i64>()?[0] == self.config.eos_token_id as i64 {
            //     return Ok((output.unwrap(), prefill_time));
            // }

            let tok_id = input_ids.i((0, 0))?.to_scalar::<i64>()?;
            if i < 5 || i % 50 == 0 {
                eprintln!("[token {}] id={}", i, tok_id);
            }
            if tok_id == self.config.eos_token_id as i64 {
                return Ok((output.unwrap(), prefill_time));
            }

            // 准备下一轮的位置 id / cache pos / mask
            // position_ids[:, -1] + 1 → [B, 1]
            position_ids = position_ids
                .narrow(1, position_ids.dim(1)? - 1, 1)?
                .to_dtype(DType::I64)?
                .add(&Tensor::new(1_i64, &position_ids.device())?.reshape((1, 1))?)?;

            // cache_position[-1] + 1 → [1]，并 reshape 为 [1] or [1, 1]
            let last_cache_pos = cache_position
                .narrow(0, cache_position.dim(0)? - 1, 1)?
                .to_dtype(DType::I64)?;
            cache_position = last_cache_pos
                .add(&Tensor::new(1_i64, &cache_position.device())?.reshape((1,))?)?; // shape: [1]

            // attention_mask: [1, S]，拼接上 [1, 1] 的 ones
            let ones = Tensor::ones((1, 1), DType::I64, &attention_mask.device())?;
            attention_mask = Tensor::cat(&[&attention_mask, &ones], 1)?; // dim=1
        }

        Ok((output.unwrap(), prefill_time))
    }
}