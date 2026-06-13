use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
// use candle_core::safetensors::{Load, load, load_buffer};
use crate::load::ExpertTensorLoader;

//#[cfg(feature = "my_own_version")]
#[derive(Debug, Clone)]
pub struct Qwen2MoeRMSNorm {
    pub weight: Tensor,
    weight_f32: Tensor,
    eps_f32: Tensor,
    pub variance_epsilon: f64,
    pub device: Device,
}
//#[cfg(feature = "my_own_version")]
impl Qwen2MoeRMSNorm {
    pub fn new(hidden_size: usize, device: Device, eps: f64) -> Self {
        Self {
            weight: Tensor::ones(hidden_size, DType::U8, &device).expect("REASON"),
            weight_f32: Tensor::ones(hidden_size, DType::F32, &device).expect("REASON"),
            eps_f32: Tensor::new(eps as f32, &device).expect("REASON"),
            variance_epsilon: eps,
            device,
        }
    }

    // pub fn init_weights(&mut self, path: &str) -> Result<()> {
    //     // 读取 .safetensors 文件内容
    //     let data = fs::read(path)?;
    //     let safetensors = SafeTensors::deserialize(&data)?;
    //     // 获取文件名作为 key
    //     let key = Path::new(path)
    //         .file_name()
    //         .and_then(|s| s.to_str())
    //         .ok_or_else(|| candle_core::Error::Msg("Invalid filename for safetensors key".into()))?;
    //     // 从 safetensors 中加载张量
    //     let tensor = safetensors.load(key, &self.device)?;
    //     self.weight = tensor;
    //     Ok(())
    // }

    pub fn init_weights(&mut self, path: &str) -> Result<()> {
        let loader = ExpertTensorLoader::new(path);
        let tensor = loader.load_tensor("tensor", &self.device)?;
        self.weight_f32 = tensor.to_dtype(DType::F32)?;
        self.weight = tensor;
        Ok(())
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states_f32 = if input_dtype == DType::F32 {
            hidden_states.clone()
        } else {
            hidden_states.to_dtype(DType::F32)?
        };
        // 计算均方差
        let variance = hidden_states_f32
            .sqr()? // pow(2)
            .mean_keepdim([hidden_states_f32.rank() - 1])?; // mean(-1, keepdim=True)
        // variance + eps => sqrt => rsqrt
        let denom = variance.broadcast_add(&self.eps_f32)?.sqrt()?.recip()?;
        // 归一化
        let normalized = hidden_states_f32.broadcast_mul(&denom)?;
        let output = normalized.broadcast_mul(&self.weight_f32)?;
        if input_dtype == DType::F32 {
            Ok(output)
        } else {
            output.to_dtype(input_dtype)
        }
    }

    pub fn debug_string(&self) -> String {
        format!(
            "Qwen2MoeRMSNorm(weight.shape={:?}, eps={})",
            self.weight.dims(),
            self.variance_epsilon
        )
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::rms_norm(hidden_size, eps, vb)?;
        Ok(Self { inner })
    }

    pub fn forward_diff(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward_diff(x)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_rmsnorm_matches_reference_formula() -> Result<()> {
        let device = Device::Cpu;
        let eps = 1e-6;
        let mut norm = Qwen2MoeRMSNorm::new(3, device.clone(), eps);
        let weight = Tensor::from_vec(vec![1f32, 2.0, 3.0], (3,), &device)?;
        norm.weight = weight.clone();
        norm.weight_f32 = weight.to_dtype(DType::F32)?;

        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, -2.0, 0.5, 4.0], (1, 2, 3), &device)?;
        let actual = norm.forward(&x)?;

        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim([x_f32.rank() - 1])?;
        let eps_tensor = Tensor::full(eps, variance.dims(), &device)?.to_dtype(DType::F32)?;
        let denom = (variance + eps_tensor)?.sqrt()?.recip()?;
        let denom_expanded = denom.broadcast_as(x_f32.shape())?.contiguous()?;
        let normalized = (x_f32 * denom_expanded)?;
        let weight_expanded = norm
            .weight
            .to_dtype(DType::F32)?
            .broadcast_as(normalized.shape())?
            .contiguous()?;
        let expected = (&normalized * &weight_expanded)?;

        let actual_values = actual.to_vec3::<f32>()?;
        let expected_values = expected.to_vec3::<f32>()?;
        for (actual_plane, expected_plane) in actual_values.iter().zip(expected_values.iter()) {
            for (actual_row, expected_row) in actual_plane.iter().zip(expected_plane.iter()) {
                for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                    assert!((actual - expected).abs() < 1e-5);
                }
            }
        }

        Ok(())
    }
}
