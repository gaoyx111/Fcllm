use crate::Cache::Cache;
use crate::DecoderLayer::Qwen2MoeDecoderLayer;
use crate::RmsNorm::Qwen2MoeRMSNorm;
use crate::configuration_qwen::Qwen2MoeConfig;
use crate::load::ExpertTensorLoader;
use crate::nn_embedding::Embedding;
use candle_core::{Device, Error, Result, Tensor};
use std::path::PathBuf;
use tqdm::tqdm;

#[derive(Debug, Clone)]
pub struct Qwen2MoeModel {
    pub config: Qwen2MoeConfig,
    pub padding_idx: usize,
    pub vocab_size: usize,

    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen2MoeDecoderLayer>,
    pub norm: Qwen2MoeRMSNorm,
}

impl Qwen2MoeModel {
    pub fn new(cfg: &Qwen2MoeConfig) -> Result<Self> {
        let embed_tokens = Embedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            Some(cfg.pad_token_id),
            &cfg.device,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = Qwen2MoeDecoderLayer::new(cfg, layer_idx)?;
            layers.push(layer);
        }

        let norm = Qwen2MoeRMSNorm::new(cfg.hidden_size, cfg.device.clone(), cfg.rms_norm_eps);

        Ok(Self {
            config: cfg.clone(),
            padding_idx: cfg.pad_token_id,
            vocab_size: cfg.vocab_size,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        let base = PathBuf::from(path);
        let original_dir = base.join("original");
        // 初始化每一层
        // for (i, layer) in self.layers.iter_mut().enumerate() {
        //     layer.init_weights(path)?;
        // }
        for i in tqdm(0..self.config.num_hidden_layers) {
            self.layers[i].init_weights(path)?;
        }
        // 加载最终 LayerNorm 权重
        let ln_path = original_dir.join("model.norm.weight");
        self.norm.init_weights(ln_path.to_str().unwrap())?;
        // 加载 embedding 权重
        let embed_path = original_dir.join("model.embed_tokens.weight");
        let loader = ExpertTensorLoader::new(embed_path.to_str().unwrap());
        let embed_weight = loader.load_tensor("tensor", &self.config.device)?;
        self.embed_tokens = self
            .embed_tokens
            .from_weight(embed_weight, self.embed_tokens.padding_idx)?;

        Ok(())
    }

    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mut past_key_values: Option<Cache>,
        inputs_embeds: Option<&Tensor>,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Cache>)> {
        // 1. input_ids 与 inputs_embeds 只能选其一
        if input_ids.is_some() && inputs_embeds.is_some() {
            return Err(Error::msg(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one",
            ));
        }
        if input_ids.is_none() && inputs_embeds.is_none() {
            return Err(Error::msg(
                "You must specify either input_ids or inputs_embeds",
            ));
        }

        // 2. (判断是否 legacy cache，需要转换为 DynamicCache) -> 解包传入的past_key_values
        // do nothing

        // 3. inputs_embeds 计算
        let inputs_embeds = match inputs_embeds {
            Some(embeds) => embeds.clone(),
            None => self.embed_tokens.forward(input_ids.unwrap())?,
        };

        // 4. 计算 cache_position
        let cache_position = match cache_position {
            Some(pos) => pos.clone(),
            None => {
                let past_seen_tokens = if let Some(cache) = &past_key_values {
                    cache.get_seq_length()
                } else {
                    0
                };

                let start = past_seen_tokens as i64;
                let seq_len = inputs_embeds.dim(1)?; // [batch, seq, hidden]
                Tensor::arange(start, start + seq_len as i64, inputs_embeds.device())?
            }
        };

        // 5. position_ids 默认是 cache_position.unsqueeze(0)
        let position_ids = match position_ids {
            Some(pos_ids) => pos_ids.clone(),
            None => cache_position.unsqueeze(0)?,
        };

        // 6. 更新 causal_mask
        let causal_mask = self._update_causal_mask(
            attention_mask,
            &inputs_embeds,
            &cache_position,
            &past_key_values,
        )?;

        // 这里不支持非 None causal_mask
        if causal_mask.is_some() {
            return Err(Error::msg("causal_mask is not None, unsupported"));
        }

        // 7. forward，初始化 hidden_states
        let mut hidden_states = inputs_embeds;

        let mut next_prefetch_expert_list = None;

        // 8. 逐层调用 decoder layer forward
        for i in 0..self.config.num_hidden_layers {
            let (left, right) = self.layers.split_at_mut(i + 1);
            let layer = &mut left[i];
            let next_layer = if i + 1 < self.config.num_hidden_layers {
                Some(&mut right[0])
            } else {
                None
            };

            let (new_hidden_states, _new_cache, new_prefetch) = layer.forward(
                &hidden_states,
                causal_mask.as_ref(),
                Some(&position_ids),
                past_key_values.as_mut(),
                Some(&cache_position),
                next_prefetch_expert_list,
                next_layer,
            )?;

            hidden_states = new_hidden_states;
            next_prefetch_expert_list = new_prefetch;
        }

        // 9. norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // 10. 根据 use_legacy_cache 决定 next_cache 返回形式 -> do nothing
        Ok((hidden_states, past_key_values))
    }

    pub fn _update_causal_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_tensor: &Tensor,
        cache_position: &Tensor,
        past_key_values: &Option<Cache>,
    ) -> Result<Option<Tensor>> {
        /*        let past_seen_tokens = if past_key_values.is_some(){
            past_key_values.get_seq_length()
        }else{
            0
        };
        let using_static_cache = false;

        if self.config._attn_implementation == "sdpa" && !using_static_cache {
            if ignore_causal_mask_sdpa(attention_mask, input_tensor, past_seen_tokens, is_training)? {
                return Ok(None);
            }
        }

        let dtype = input_tensor.dtype();
        let device = input_tensor.device();
        let min_dtype = dtype.min_value()?;

        let shape = input_tensor.dims();
        let (batch_size, sequence_length) = (shape[0], shape[1]);

        let target_length = if let Some(mask) = attention_mask {
            mask.dims()[mask.dims().len() - 1]
        } else {
            past_seen_tokens + sequence_length + 1
        };

        let causal_mask = prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length,
            target_length,
            dtype,
            device,
            min_dtype,
            cache_position,
            batch_size,
        )?;

        if config.attn_implementation == "sdpa"
            && attention_mask.is_some()
            && device == &config.device
        {
            return Ok(Some(unmask_unattended(causal_mask, min_dtype)?));
        }

        Ok(Some(causal_mask)) */
        Ok(None)
    }
}

pub fn prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: Option<&Tensor>,
    sequence_length: usize,
    target_length: usize,
    dtype: candle_core::DType,
    device: &Device,
    min_dtype_val: f32,
    cache_position: &Tensor,
    batch_size: usize,
) -> Result<Tensor> {
    if let Some(attn_mask) = attention_mask {
        if attn_mask.dims().len() == 4 {
            return Ok(attn_mask.clone());
        }
    }

    // Tensor::full 参数顺序：填充值，形状，设备
    let mut causal_mask = Tensor::full(min_dtype_val, &[sequence_length, target_length], device)?;

    // candle_core 里没有 triu 这个接口，手动实现上三角 mask
    if sequence_length != 1 {
        causal_mask = upper_triangle_mask(causal_mask, 1)?;
    }

    // arange
    let arange = Tensor::arange(0f32, target_length as f32, device)?;

    // 这里广播 arange
    // 先 reshape cache_position: [batch_size, 1]
    let cache_pos_reshaped = cache_position.reshape((batch_size, 1))?;
    // 需要广播比较 arange: [target_length] > cache_position: [batch_size, 1]
    // 先扩展 arange: [1, target_length]
    let arange_broadcast = arange.reshape((1, target_length))?;
    // 广播比较，得到 bool mask
    let mask_bool = arange_broadcast.gt(&cache_pos_reshaped)?;

    // 转 bool mask 为 float mask (0/1 转换为 0 / min_dtype_val)
    // candle_core 目前没有直接的 bool tensor，通常用 f32 0.0 / 1.0 实现
    let scalar_tensor = Tensor::full(min_dtype_val, &[], device)?; // 0-dim Tensor
    let mask_float = (mask_bool.to_dtype(dtype)? * scalar_tensor)?;

    // causal_mask shape (sequence_length, target_length)
    // 按广播规则 expand causal_mask 为 (batch_size, 1, sequence_length, target_length)
    let causal_mask_expanded = causal_mask.unsqueeze(0)?.unsqueeze(1)?.expand(&[
        batch_size,
        1,
        sequence_length,
        target_length,
    ])?;

    // 把 mask_float reshape 到 (batch_size, 1, 1, target_length)
    let mask_float_expanded = mask_float.unsqueeze(1)?.unsqueeze(2)?;

    causal_mask = (causal_mask_expanded * mask_float_expanded)?;

    if let Some(attn_mask) = attention_mask {
        // 复制以便原地操作
        causal_mask = causal_mask.clone();

        let mask_len = attn_mask.dims()[attn_mask.dims().len() - 1];

        // 取出 causal_mask 的前 mask_len 切片：dim=3, start=0, len=mask_len
        let causal_slice = causal_mask.narrow(3, 0usize, mask_len)?;

        // attn_mask 形状可能是 [batch_size, seq_len] 或其他，根据你的具体需要调整
        // 需要扩展 attn_mask 以对齐 causal_slice 的形状
        // 这里简单扩展为 [batch_size, 1, 1, mask_len]
        let attn_mask_expanded = attn_mask.unsqueeze(1)?.unsqueeze(2)?;

        // 加法
        let sum = (&causal_slice + &attn_mask_expanded)?;

        // PyTorch 里 padding_mask = (sum == 0)
        // candle_core 不支持直接 eq_scalar，使用 sub -> abs -> < epsilon 判断近似等于0
        let zero_tensor = Tensor::full(0.0f32, sum.dims(), device)?;
        let diff = (&sum - &zero_tensor)?.abs()?;

        // 定义阈值，判断是否等于0，这里用1e-6作为阈值
        let scalar = Tensor::full(1e-6f32, &[], device)?;
        let padding_mask = diff.lt(&scalar)?;

        // candle_core 也没有 masked_fill，需要手动用 padding_mask 替换对应位置
        // 这里用 where 操作，类似 PyTorch 的 masked_fill
        causal_mask = tensor_masked_fill(&causal_mask, &padding_mask, min_dtype_val)?;
    }

    Ok(causal_mask)
}

// 手动实现 upper triangular mask
pub fn upper_triangle_mask(tensor: Tensor, diagonal: i64) -> Result<Tensor> {
    let dims = tensor.dims();
    let seq_len = dims[0];
    let tgt_len = dims[1];
    let device = tensor.device();

    // 创建一个矩阵，每个元素是列索引 - 行索引
    let row_idx = Tensor::arange(0f32, seq_len as f32, &device)?.unsqueeze(1)?;
    let col_idx = Tensor::arange(0f32, tgt_len as f32, &device)?.unsqueeze(0)?;

    let diff = (&col_idx - &row_idx)?; // broadcast to [seq_len, tgt_len]

    // mask = diff >= diagonal
    let scalar_diag = Tensor::full(diagonal as f32, &[], device)?;
    let mask = diff.ge(&scalar_diag)?;

    // 用 mask 选择原 tensor 或 min_dtype_val 替代
    tensor_where(&mask, &tensor, &Tensor::full(0.0, tensor.dims(), &device)?)
}

// 用 mask 替换 tensor 中对应元素为 value
pub fn tensor_masked_fill(tensor: &Tensor, mask: &Tensor, value: f32) -> Result<Tensor> {
    let device = tensor.device();

    let val_tensor = Tensor::full(value, tensor.dims(), &device)?;

    // torch.where(mask, val, tensor)
    tensor_where(mask, &val_tensor, tensor)
}

// 类似 torch.where(cond, x, y)
pub fn tensor_where(cond: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    // cond 是 bool tensor，这里用 0/1 float tensor 代替
    // 计算 cond * x + (1 - cond) * y
    let one = Tensor::full(1.0, cond.dims(), &cond.device())?;

    let cond_f = cond.to_dtype(candle_core::DType::F32)?;

    let inv_cond = (&one - &cond_f)?;

    let part1 = (&cond_f * x)?;
    let part2 = (&inv_cond * y)?;

    &part1 + &part2
}
