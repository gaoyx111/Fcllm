use candle_core::{DType, Device, Result, Tensor}; 
use candle_nn::Linear;


// #[derive(Debug, Clone)]
// pub struct Linear {
//     inner: candle_nn::Linear,
// }

// impl Linear {
//     pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
//         let inner = candle_nn::Linear::new(weights, bias);
//         Self { inner }
//     }
// }

// pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
//     let inner = candle_nn::linear_b(d1, d2, b, vb)?;
//     Ok(Linear { inner })
// }

// pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
//     let inner = candle_nn::linear(d1, d2, vb)?;
//     Ok(Linear { inner })
// }

// pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
//     let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
//     Ok(Linear { inner })
// }

// impl Module for Linear {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
//         self.inner.forward(xs)
//     }
// }


// #[cfg(feature = "my_own_version")]
pub fn new_uninitialized_linear(
    in_features: usize,
    out_features: usize,
    with_bias: bool,
    device: &Device,
) -> Result<Linear> {
    let w = Tensor::zeros((out_features, in_features), DType::F32, device)?;
    let b = if with_bias {
        Some(Tensor::zeros(out_features, DType::F32, device)?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

