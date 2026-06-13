use crate::configuration_qwen::Qwen2MoeConfig;
use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
struct ScatterPositionCache {
    current_len: usize,
    new_len: usize,
    batch_size: usize,
    num_heads: usize,
    token_count: usize,
    head_dim: usize,
    positions: Tensor,
}

#[derive(Debug, Clone)]
pub struct Cache {
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    seq_lengths: Vec<usize>,
    static_kv_capacity: usize,
    max_seq_len: usize,
    scatter_position_cache: Option<ScatterPositionCache>,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Qwen2MoeConfig) -> Result<Self> {
        Self::new_with_capacity(use_kv_cache, dtype, config, 0)
    }

    pub fn new_with_capacity(
        use_kv_cache: bool,
        _dtype: DType,
        config: &Qwen2MoeConfig,
        static_kv_capacity: usize,
    ) -> Result<Self> {
        let max_seq_len = config.max_position_embeddings;
        let static_kv_capacity = if use_kv_cache {
            static_kv_capacity.min(max_seq_len)
        } else {
            0
        };

        Ok(Self {
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            seq_lengths: vec![0; config.num_hidden_layers],
            static_kv_capacity,
            max_seq_len,
            scatter_position_cache: None,
        })
    }

    pub fn get_usable_length(&self, layer_idx: usize, _current_len: usize) -> Result<usize> {
        let seq_len = self.seq_lengths.get(layer_idx).copied().unwrap_or_default();
        if seq_len > 0 {
            Ok(seq_len)
        } else if let Some((k, _)) = &self.kvs[layer_idx] {
            Ok(k.dim(2)?)
        } else {
            Ok(0)
        }
    }

    pub fn update_kv(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        if !self.use_kv_cache {
            return Ok((key.clone(), value.clone()));
        }

        let token_count = key.dim(2)?;
        let current_len = self.get_usable_length(layer_idx, token_count)?;
        let new_len = current_len + token_count;

        if self.static_kv_capacity > 0 && new_len <= self.static_kv_capacity {
            return self.update_static_kv(
                key,
                value,
                layer_idx,
                current_len,
                new_len,
                cache_position,
            );
        }

        self.scatter_position_cache = None;

        if self.kvs[layer_idx].is_none() {
            let mut k = key.clone();
            let mut v = value.clone();
            if token_count > self.max_seq_len {
                k = k.narrow(2, token_count - self.max_seq_len, self.max_seq_len)?;
                v = v.narrow(2, token_count - self.max_seq_len, self.max_seq_len)?;
            }
            self.seq_lengths[layer_idx] = k.dim(2)?;
            self.kvs[layer_idx] = Some((k.clone(), v.clone()));
            return Ok((k, v));
        }

        let (k_cache, v_cache) = self.kvs[layer_idx].take().unwrap();
        let active_len = current_len.min(k_cache.dim(2)?);
        let k_cache = if active_len < k_cache.dim(2)? {
            k_cache.narrow(2, 0, active_len)?
        } else {
            k_cache
        };
        let active_len = current_len.min(v_cache.dim(2)?);
        let v_cache = if active_len < v_cache.dim(2)? {
            v_cache.narrow(2, 0, active_len)?
        } else {
            v_cache
        };

        let mut k = Tensor::cat(&[&k_cache, key], 2)?;
        let mut v = Tensor::cat(&[&v_cache, value], 2)?;
        // k shape: [B, H, S, D] — dims()[2] is the seq_len dimension
        let k_seq_len = k.dims()[2];
        if k_seq_len > self.max_seq_len {
            k = k
                .narrow(2, k_seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?
        }
        let v_seq_len = v.dims()[2];
        if v_seq_len > self.max_seq_len {
            v = v
                .narrow(2, v_seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?
        }

        self.seq_lengths[layer_idx] = k.dim(2)?;
        self.kvs[layer_idx] = Some((k.clone(), v.clone()));
        Ok((k, v))
    }

    fn update_static_kv(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
        current_len: usize,
        new_len: usize,
        cache_position: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, num_heads, token_count, head_dim) = key.dims4()?;
        let allocate = match &self.kvs[layer_idx] {
            Some((k_cache, v_cache)) => {
                let k_dims = k_cache.dims4()?;
                let v_dims = v_cache.dims4()?;
                k_dims != (batch_size, num_heads, self.static_kv_capacity, head_dim)
                    || v_dims != (batch_size, num_heads, self.static_kv_capacity, head_dim)
            }
            None => true,
        };

        if allocate {
            let k_cache = Tensor::zeros(
                (batch_size, num_heads, self.static_kv_capacity, head_dim),
                key.dtype(),
                key.device(),
            )?;
            let v_cache = Tensor::zeros(
                (batch_size, num_heads, self.static_kv_capacity, head_dim),
                value.dtype(),
                value.device(),
            )?;
            self.kvs[layer_idx] = Some((k_cache, v_cache));
            self.seq_lengths[layer_idx] = 0;
        }

        let positions = self.static_scatter_positions(
            cache_position,
            current_len,
            new_len,
            batch_size,
            num_heads,
            token_count,
            head_dim,
            key.device(),
        )?;
        let (k_cache, v_cache) = self.kvs[layer_idx].as_ref().unwrap();
        let key = key.contiguous()?;
        let value = value.contiguous()?;

        k_cache.scatter_set(&positions, &key, 2)?;
        v_cache.scatter_set(&positions, &value, 2)?;

        self.seq_lengths[layer_idx] = new_len;
        Ok((
            k_cache.narrow(2, 0, new_len)?,
            v_cache.narrow(2, 0, new_len)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn static_scatter_positions(
        &mut self,
        cache_position: Option<&Tensor>,
        current_len: usize,
        new_len: usize,
        batch_size: usize,
        num_heads: usize,
        token_count: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if let Some(cache) = &self.scatter_position_cache {
            if cache.current_len == current_len
                && cache.new_len == new_len
                && cache.batch_size == batch_size
                && cache.num_heads == num_heads
                && cache.token_count == token_count
                && cache.head_dim == head_dim
            {
                return Ok(cache.positions.clone());
            }
        }

        let base = match cache_position {
            Some(pos) if pos.elem_count() == token_count => pos.clone(),
            _ => Tensor::arange(current_len as u32, new_len as u32, device)?,
        };
        let positions = base
            .reshape((1, 1, token_count, 1))?
            .expand((batch_size, num_heads, token_count, head_dim))?
            .contiguous()?;

        self.scatter_position_cache = Some(ScatterPositionCache {
            current_len,
            new_len,
            batch_size,
            num_heads,
            token_count,
            head_dim,
            positions: positions.clone(),
        });

        Ok(positions)
    }

    /// 获取当前已缓存的 token 数（即 sequence length）
    pub fn get_seq_length(&self) -> usize {
        if let Some(seq_len) = self.seq_lengths.iter().copied().find(|len| *len > 0) {
            return seq_len;
        }
        for kv in &self.kvs {
            if let Some((k, _)) = kv {
                if let Ok(dim) = k.dim(2) {
                    return dim;
                }
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Qwen2MoeConfig {
        let mut config = Qwen2MoeConfig::new();
        config.device = Device::Cpu;
        config.hidden_size = 4;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 1;
        config.num_hidden_layers = 1;
        config.max_position_embeddings = 8;
        config
    }

    #[test]
    fn static_kv_cache_appends_without_growing_storage() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let mut cache = Cache::new_with_capacity(true, DType::F32, &config, 4)?;

        let key = Tensor::from_vec(vec![1f32, 2.], (1, 1, 2, 1), &device)?;
        let value = Tensor::from_vec(vec![10f32, 20.], (1, 1, 2, 1), &device)?;
        let (k, v) = cache.update_kv(&key, &value, 0, None)?;

        assert_eq!(cache.get_usable_length(0, 0)?, 2);
        assert_eq!(cache.get_seq_length(), 2);
        assert_eq!(cache.kvs[0].as_ref().unwrap().0.dim(2)?, 4);
        assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, vec![1., 2.]);
        assert_eq!(v.flatten_all()?.to_vec1::<f32>()?, vec![10., 20.]);

        let key = Tensor::from_vec(vec![3f32], (1, 1, 1, 1), &device)?;
        let value = Tensor::from_vec(vec![30f32], (1, 1, 1, 1), &device)?;
        let (k, v) = cache.update_kv(&key, &value, 0, None)?;

        assert_eq!(cache.get_usable_length(0, 0)?, 3);
        assert_eq!(cache.get_seq_length(), 3);
        assert_eq!(cache.kvs[0].as_ref().unwrap().0.dim(2)?, 4);
        assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, vec![1., 2., 3.]);
        assert_eq!(v.flatten_all()?.to_vec1::<f32>()?, vec![10., 20., 30.]);

        Ok(())
    }
}
