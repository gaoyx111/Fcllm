use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Qwen2MoeRotaryEmbedding {
    pub dim: usize,
    pub base: f32,
    pub device: Device,
    pub dtype: DType,
    pub inv_freq: Tensor,
    pub sin_cached: Tensor,
    pub cos_cached: Tensor,
    pub max_seq_len_cached: usize,
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

        let (cos_cached, sin_cached) =
            Self::build_cos_sin_cache(&inv_freq, max_position_embeddings, dtype, device)?;

        Ok(Self {
            dim,
            base,
            device: device.clone(),
            dtype,
            inv_freq,
            sin_cached,
            cos_cached,
            max_seq_len_cached: max_position_embeddings,
        })
    }

    fn build_cos_sin_cache(
        inv_freq: &Tensor,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((seq_len, 1))?; // Shape: [seq_len, 1]

        let inv_freq = inv_freq.broadcast_as((1, inv_freq.dims1()?))?; // Shape: [1, dim/2]

        let freqs = t.matmul(&inv_freq)?; // Shape: [seq_len, dim/2]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?; // Shape: [seq_len, dim]

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok((cos, sin))
    }

    fn ensure_cos_sin_cache(&self, seq_len: usize) -> Result<()> {
        if seq_len > self.max_seq_len_cached {
            return Err(candle_core::Error::Msg(format!(
                "rotary seq_len {seq_len} exceeds cached maximum {}",
                self.max_seq_len_cached
            )));
        }
        Ok(())
    }

    /// Returns (cos, sin) of shape [seq_len, dim]
    pub fn get_cos_sin(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        self.ensure_cos_sin_cache(seq_len)?;

        let cos = self
            .cos_cached
            .narrow(0, 0, seq_len)?
            .to_dtype(self.dtype)?
            .to_device(&self.device)?;
        let sin = self
            .sin_cached
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
        self.ensure_cos_sin_cache(seq_len)?;

        let cos = self
            .cos_cached
            .narrow(0, 0, seq_len)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let sin = self
            .sin_cached
            .narrow(0, 0, seq_len)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok((cos, sin))
    }

    pub fn forward_positioned(
        &self,
        seq_len: usize,
        position_ids: &Tensor,
        dtype: DType,
        device: &Device,
        unsqueeze_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.ensure_cos_sin_cache(seq_len)?;

        let cos = index_positioned_tensor(&self.cos_cached, position_ids, unsqueeze_dim)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let sin = index_positioned_tensor(&self.sin_cached, position_ids, unsqueeze_dim)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok((cos, sin))
    }

    pub fn forward_single_position(
        &self,
        seq_len: usize,
        position: usize,
        dtype: DType,
        device: &Device,
        unsqueeze_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        if position >= seq_len {
            return Err(candle_core::Error::Msg(format!(
                "rotary position {position} out of range for seq_len {seq_len}"
            )));
        }
        self.ensure_cos_sin_cache(seq_len)?;

        let cos = single_positioned_tensor(&self.cos_cached, position, unsqueeze_dim)?
            .to_dtype(dtype)?
            .to_device(device)?;
        let sin = single_positioned_tensor(&self.sin_cached, position, unsqueeze_dim)?
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

pub fn apply_rotary_pos_emb_positioned(
    q: &Tensor,
    k: &Tensor,
    cos_pos: &Tensor,
    sin_pos: &Tensor,
) -> Result<(Tensor, Tensor)> {
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

pub fn single_positioned_tensor(
    base: &Tensor,
    position: usize,
    unsqueeze_dim: usize,
) -> Result<Tensor> {
    let gathered = base.narrow(0, position, 1)?.unsqueeze(0)?;
    gathered.unsqueeze(unsqueeze_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "value {idx} differs: actual={a} expected={e}"
            );
        }
        Ok(())
    }

    #[test]
    fn positioned_rotary_matches_full_cache_gather() -> Result<()> {
        let device = Device::Cpu;
        let rotary = Qwen2MoeRotaryEmbedding::new(4, 16, 10_000.0, DType::F32, &device)?;
        let position_ids = Tensor::from_vec(vec![3u32, 5], (1, 2), &device)?;
        let q = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            ],
            (1, 2, 2, 4),
            &device,
        )?;
        let k = Tensor::from_vec(
            vec![1.6f32, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9],
            (1, 1, 2, 4),
            &device,
        )?;

        let (full_cos, full_sin) = rotary.forward(6, DType::F32, &device)?;
        let (old_q, old_k) = apply_rotary_pos_emb(&q, &k, &full_cos, &full_sin, &position_ids, 1)?;
        let (cos_pos, sin_pos) =
            rotary.forward_positioned(6, &position_ids, DType::F32, &device, 1)?;
        let (new_q, new_k) = apply_rotary_pos_emb_positioned(&q, &k, &cos_pos, &sin_pos)?;

        assert_close(&new_q, &old_q)?;
        assert_close(&new_k, &old_k)?;
        Ok(())
    }

    #[test]
    fn single_position_rotary_matches_positioned_gather() -> Result<()> {
        let device = Device::Cpu;
        let rotary = Qwen2MoeRotaryEmbedding::new(4, 16, 10_000.0, DType::F32, &device)?;
        let position_ids = Tensor::from_vec(vec![5u32], (1, 1), &device)?;
        let q = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            (1, 2, 1, 4),
            &device,
        )?;
        let k = Tensor::from_vec(vec![1.6f32, 1.5, 1.4, 1.3], (1, 1, 1, 4), &device)?;

        let (cos_pos, sin_pos) =
            rotary.forward_positioned(6, &position_ids, DType::F32, &device, 1)?;
        let (old_q, old_k) = apply_rotary_pos_emb_positioned(&q, &k, &cos_pos, &sin_pos)?;
        let (single_cos, single_sin) =
            rotary.forward_single_position(6, 5, DType::F32, &device, 1)?;
        let (new_q, new_k) = apply_rotary_pos_emb_positioned(&q, &k, &single_cos, &single_sin)?;

        assert_close(&new_q, &old_q)?;
        assert_close(&new_k, &old_k)?;
        Ok(())
    }
}
