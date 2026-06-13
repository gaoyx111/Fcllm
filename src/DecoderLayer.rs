use crate::Attention::Qwen2MoeAttention;
use crate::Cache::Cache;
use crate::RmsNorm::Qwen2MoeRMSNorm;
use crate::RotaryEmbedding::Qwen2MoeRotaryEmbedding;
use crate::SparseMoeBlock::Qwen2MoeSparseMoeBlock;
use crate::configuration_qwen::Qwen2MoeConfig;
use candle_core::{Result, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Qwen2MoeDecoderLayer {
    pub config: Qwen2MoeConfig,
    pub hidden_size: usize,
    pub layer_idx: usize,
    pub self_attn: Qwen2MoeAttention,
    pub mlp: Qwen2MoeSparseMoeBlock,
    pub input_layernorm: Qwen2MoeRMSNorm,
    pub post_attention_layernorm: Qwen2MoeRMSNorm,
}

impl Qwen2MoeDecoderLayer {
    pub fn new(cfg: &Qwen2MoeConfig, layer_idx: usize) -> Result<Self> {
        let self_attn = Qwen2MoeAttention::new(cfg, Some(layer_idx))?;
        Self::from_attention(cfg, layer_idx, self_attn)
    }

    pub fn new_with_rotary(
        cfg: &Qwen2MoeConfig,
        layer_idx: usize,
        rotary_emb: Arc<Qwen2MoeRotaryEmbedding>,
    ) -> Result<Self> {
        let self_attn = Qwen2MoeAttention::new_with_rotary(cfg, Some(layer_idx), rotary_emb)?;
        Self::from_attention(cfg, layer_idx, self_attn)
    }

    fn from_attention(
        cfg: &Qwen2MoeConfig,
        layer_idx: usize,
        self_attn: Qwen2MoeAttention,
    ) -> Result<Self> {
        let mlp = Qwen2MoeSparseMoeBlock::new(cfg, layer_idx)?;

        let input_layernorm =
            Qwen2MoeRMSNorm::new(cfg.hidden_size, cfg.device.clone(), cfg.rms_norm_eps);
        let post_attention_layernorm =
            Qwen2MoeRMSNorm::new(cfg.hidden_size, cfg.device.clone(), cfg.rms_norm_eps);

        Ok(Self {
            config: cfg.clone(),
            hidden_size: cfg.hidden_size,
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        self.self_attn.init_weights(path)?;
        self.mlp.init_weights(path)?;

        let base = PathBuf::from(path);
        let original_dir = base.join("original");

        // input_layernorm 权重路径
        let input_ln_path = original_dir.join(format!(
            "model.layers.{}.input_layernorm.weight",
            self.layer_idx
        ));
        self.input_layernorm
            .init_weights(input_ln_path.to_str().unwrap())?;

        // post_attention_layernorm 权重路径
        let post_ln_path = original_dir.join(format!(
            "model.layers.{}.post_attention_layernorm.weight",
            self.layer_idx
        ));
        self.post_attention_layernorm
            .init_weights(post_ln_path.to_str().unwrap())?;

        Ok(())
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_value: Option<&mut Cache>,
        cache_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(&hidden_states)?;

        let (hidden_states, _present_key_value) = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            cache_position,
        )?;
        let hidden_states = (hidden_states + residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // 当前层 MoE 执行
        let hidden_states = self.mlp.forward(&hidden_states, None)?;

        let hidden_states = (hidden_states + residual)?;

        Ok(hidden_states)
    }
}
