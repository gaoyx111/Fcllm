use candle_core::{DType, Device, Error, Result, Tensor};

#[derive(Debug)]
pub struct Embedding {
    pub weight: Tensor,
    pub padding_idx: Option<usize>,
}

impl Embedding {
    /// 初始化 embedding，默认用 0 填充
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        padding_idx: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::zeros((vocab_size, hidden_size), DType::F32, device)?;
        Ok(Self {
            weight,
            padding_idx,
        })
    }

    pub fn from_weight(&self, mut weight: Tensor, padding_idx: Option<usize>) -> Result<Self> {
        if weight.dims().len() != 2 {
            return Err(Error::Msg(format!(
                "Expected 2D tensor for embedding weight, got shape {:?}",
                weight.dims()
            )));
        }

        if let Some(pidx) = padding_idx {
            let hidden_size = weight.dims()[1];
            let mut mask_data = vec![1f32; weight.dims()[0]];
            mask_data[pidx] = 0.0;

            let mask = Tensor::from_vec(mask_data, (weight.dims()[0], 1), weight.device())?;
            let mask = mask
                .to_dtype(weight.dtype())?
                .broadcast_as(weight.shape())?;
            weight = (&weight * &mask)?;
        }

        Ok(Self {
            weight,
            padding_idx,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let original_dims = input_ids.dims();
        let input_ids = input_ids.flatten(0, input_ids.rank() - 1)?;
        let embeddings = self.weight.index_select(&input_ids, 0)?;

        let embeddings =
            embeddings.reshape(&[original_dims[0], original_dims[1], self.weight.dims()[1]])?;

        if let Some(pidx) = self.padding_idx {
            let input_ids_u32 = input_ids.to_dtype(DType::U32)?;
            let mask = input_ids_u32.eq(pidx as u32)?;
            let zeros = Tensor::zeros_like(&embeddings)?;
            let result = mask
                .reshape(&[original_dims[0], original_dims[1], 1])?
                .broadcast_as(embeddings.shape())?
                .where_cond(&zeros, &embeddings)?;
            Ok(result)
        } else {
            Ok(embeddings)
        }
    }
}
