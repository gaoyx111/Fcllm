use candle_core::{DType, Device, Result, Tensor};
use std::sync::Mutex;

#[derive(Debug)]
pub struct Qwen2MoeRotaryEmbedding {
    pub dim: usize,
    pub base: f32,
    pub device: Device,
    pub dtype: DType,
    pub inv_freq: Tensor,
    pub sin_cached: Mutex<Option<Tensor>>,
    pub cos_cached: Mutex<Option<Tensor>>,
    pub max_seq_len_cached: Mutex<usize>,
}

impl Qwen2MoeRotaryEmbedding {
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Compute inv_freq: shape [dim / 2]
        let inv_freq_data: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (base.powf(i as f32 / dim as f32)) as f32)
            .collect();

        let shape = (inv_freq_data.len(),);
        let inv_freq = Tensor::from_vec(inv_freq_data, shape, device)?.to_dtype(dtype)?;

        let embedding = Self {
            dim,
            base,
            device: device.clone(),
            dtype,
            inv_freq,
            sin_cached: Mutex::new(None),
            cos_cached: Mutex::new(None),
            max_seq_len_cached: Mutex::new(0),
        };

        embedding.set_cos_sin_cache(max_position_embeddings)?; // Initial cache
        Ok(embedding)
    }

    pub fn set_cos_sin_cache(&self, seq_len: usize) -> Result<()> {
        let t = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(self.dtype)?
            .reshape((seq_len, 1))?; // Shape: [seq_len, 1]

        let inv_freq = self.inv_freq.broadcast_as((1, self.inv_freq.dims1()?))?; // Shape: [1, dim/2]

        let freqs = t.matmul(&inv_freq)?; // Shape: [seq_len, dim/2]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?; // Shape: [seq_len, dim]

        let cos = freqs.cos()?.to_dtype(self.dtype)?;
        let sin = freqs.sin()?.to_dtype(self.dtype)?;

        *self.cos_cached.lock().unwrap() = Some(cos);
        *self.sin_cached.lock().unwrap() = Some(sin);
        *self.max_seq_len_cached.lock().unwrap() = seq_len;

        Ok(())
    }

    /// Returns (cos, sin) of shape [seq_len, dim]
    pub fn get_cos_sin(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let mut cached_len = self.max_seq_len_cached.lock().unwrap();
        if seq_len > *cached_len {
            self.set_cos_sin_cache(seq_len)?; // 默认以 self.dtype, self.device 构造缓存
            *cached_len = seq_len;
        }

        let cos = self
            .cos_cached
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .narrow(0, 0, seq_len)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        let sin = self
            .sin_cached
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .narrow(0, 0, seq_len)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        Ok((cos, sin))
    }

    pub fn forward(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let mut cached_len = self.max_seq_len_cached.lock().unwrap();
        if seq_len > *cached_len {
            self.set_cos_sin_cache(seq_len)?; // 默认以 self.dtype, self.device 构造缓存
            *cached_len = seq_len;
        }

        let cos = self
            .cos_cached
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .narrow(0, 0, seq_len)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let sin = self
            .sin_cached
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .narrow(0, 0, seq_len)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok((cos, sin))
    }
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let dim = x.dims().last().copied().unwrap();
    let x1 = x.narrow(x.rank() - 1, 0, dim / 2)?;
    let x2 = x.narrow(x.rank() - 1, dim / 2, dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], x.rank() - 1)
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,            // [bsz, n_heads, seq_len, head_dim]
    k: &Tensor,            // [bsz, n_kv_heads, seq_len, head_dim]
    cos: &Tensor,          // [max_seq_len, head_dim]
    sin: &Tensor,          // [max_seq_len, head_dim]
    position_ids: &Tensor, // [bsz, seq_len]
    unsqueeze_dim: usize,  // 1 => insert dim at position 1 for head broadcast
) -> Result<(Tensor, Tensor)> {
    // cos[position_ids] => [bsz, seq_len, head_dim], then unsqueeze(1) => [bsz, 1, seq_len, head_dim]
    let cos_pos = index_positioned_tensor(cos, position_ids, unsqueeze_dim)?;
    let sin_pos = index_positioned_tensor(sin, position_ids, unsqueeze_dim)?;

    let q_rot = rotate_half(q)?;
    let k_rot = rotate_half(k)?;

    let q_embed = q
        .mul(&cos_pos.broadcast_as(q.shape())?)?
        .add(&q_rot.mul(&sin_pos.broadcast_as(q.shape())?)?)?;
    let k_embed = k
        .mul(&cos_pos.broadcast_as(k.shape())?)?
        .add(&k_rot.mul(&sin_pos.broadcast_as(k.shape())?)?)?;

    Ok((q_embed, k_embed))
}

pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b, num_kv_heads, slen, head_dim) = hidden_states.dims4()?;

    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }

    // Expand shape: [b, num_kv_heads, 1, slen, head_dim] -> [b, num_kv_heads, n_rep, slen, head_dim]
    let expanded = hidden_states
        .unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, slen, head_dim))?;

    // Reshape to [b, num_kv_heads * n_rep, slen, head_dim]
    expanded.reshape((b, num_kv_heads * n_rep, slen, head_dim))
}

// pub fn index_positioned_tensor(
//     base: &Tensor,             // [seq_len, dim]
//     position_ids: &Tensor,     // [bsz, seq_len]
//     unsqueeze_dim: usize,
// ) -> Result<Tensor> {
//     let (bsz, seq_len) = position_ids.dims2()?;  // [bsz, seq_len]
//     let dim = base.dim(1)?;                      // [seq_len, dim]

//     // position_ids: [bsz, seq_len] -> [bsz, seq_len, 1] -> broadcast -> [bsz, seq_len, dim]
//     let position_ids = position_ids
//         .reshape((bsz, seq_len, 1))?
//         .broadcast_as((bsz, seq_len, dim))?
//         .to_dtype(DType::I64)?;

//     // gather from base: [seq_len, dim] -> [bsz, seq_len, dim]
//     let gathered = base.gather(&position_ids, 0)?;  // gather along seq_len dim (dim 0 of base)

//     // unsqueeze to desired shape
//     gathered.unsqueeze(unsqueeze_dim)
// }

pub fn index_positioned_tensor(
    base: &Tensor,         // [max_seq_len, dim]
    position_ids: &Tensor, // [bsz, seq_len]
    unsqueeze_dim: usize,
) -> Result<Tensor> {
    let (bsz, seq_len) = position_ids.dims2()?;
    let dim = base.dim(1)?;

    // Python: cos[position_ids] => fancy indexing, selects entire rows
    // Flatten position_ids -> index_select -> reshape back
    let flat_ids = position_ids.flatten_all()?.to_dtype(DType::U32)?;
    let gathered = base.index_select(&flat_ids, 0)?; // [bsz*seq_len, dim]
    let gathered = gathered.reshape((bsz, seq_len, dim))?; // [bsz, seq_len, dim]
    gathered.unsqueeze(unsqueeze_dim) // e.g. unsqueeze(1) => [bsz, 1, seq_len, dim]
}
