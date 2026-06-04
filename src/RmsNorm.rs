use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};
// use candle_core::safetensors::{Load, load, load_buffer};
use crate::load::ExpertTensorLoader;

//#[cfg(feature = "my_own_version")]
#[derive(Debug, Clone)]
pub struct Qwen2MoeRMSNorm {
    pub weight: Tensor,
    pub variance_epsilon: f64,
    pub device: Device,
}
//#[cfg(feature = "my_own_version")]
impl Qwen2MoeRMSNorm {
    pub fn new(hidden_size: usize, device: Device, eps: f64) -> Self {
        Self {
            weight: Tensor::ones(hidden_size, DType::U8, &device).expect("REASON"),
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
        self.weight = tensor;
        Ok(())
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states_f32 = hidden_states.to_dtype(DType::F32)?;
        // 计算均方差
        let variance = hidden_states_f32
            .sqr()? // pow(2)
            .mean_keepdim([hidden_states_f32.rank() - 1])?; // mean(-1, keepdim=True)
        let eps_tensor = Tensor::full(self.variance_epsilon, variance.dims(), &self.device)?
            .to_dtype(DType::F32)?;
        // variance + eps => sqrt => rsqrt
        let denom = (variance + eps_tensor)?.sqrt()?.recip()?;
        // 归一化
        let denom_expanded = denom
            .broadcast_as(hidden_states_f32.shape())?
            .contiguous()?;
        let normalized = (hidden_states_f32 * denom_expanded)?;
        let weight_expanded = self
            .weight
            .to_dtype(DType::F32)?
            .broadcast_as(normalized.shape())?
            .contiguous()?;
        let output = (&normalized * &weight_expanded)?;
        output.to_dtype(input_dtype)
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
