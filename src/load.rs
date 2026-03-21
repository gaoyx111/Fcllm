use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;
use candle_core::safetensors::load_buffer;
use std::path::{PathBuf, Path};

pub fn load_tensor_from_file(path: &str, key: &str, device: &Device) -> candle_core::Result<Tensor> {
    let data = std::fs::read(path)?; // 读取 .safetensors 文件为字节流
    let tensors: HashMap<String, Tensor> = load_buffer(&data, device)?;
    tensors
        .into_iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| candle_core::Error::Msg(format!("Key {} not found", key)))
}

pub fn load_linear_from_files(
    weight_path: &str,
    bias_path: Option<&str>,
    device: &Device,
) -> Result<Linear> {
    let weight = load_tensor_from_file(weight_path, "tensor", device)?;
    let bias = match bias_path {
        Some(path) => Some(load_tensor_from_file(path, "tensor", device)?),
        None => None,
    };
    //println!("weight: {:?}, bias: {:?}", weight, bias);
    Ok(Linear::new(weight, bias))
}


pub struct ExpertTensorLoader {
    pub root: PathBuf,
    pub parts: Vec<String>, // 用于存文件名段
}

impl ExpertTensorLoader {
    pub fn new<P: AsRef<Path>>(full_path: P) -> Self {
        let full_path = full_path.as_ref();

        // 父目录：即文件所在目录
        let root = full_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();

        // 文件名（如 "model.layers.0.mlp.shared_expert.gate_proj.weight"）
        let filename = full_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new(""))
            .to_string_lossy()
            .to_string();

        // 拆成 vec!["model", "layers", "0", ..., "weight"]
        let parts = filename
            .split('.')
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        Self { root, parts }
    }

    pub fn pp<S: AsRef<str>>(&self, part: S) -> Self {
        let mut new_parts = self.parts.clone();
        new_parts.push(part.as_ref().to_string());
        Self {
            root: self.root.clone(),
            parts: new_parts,
        }
    }

    // 取唯一键，否则报错：
    //Error: Expected exactly one tensor in file "E:\\Rust\\model_weights\\Qwen\\Qwen1.5-MoE-A2.7B\\original\\model.layers.0.mlp.experts.0.weight",
    //but found 3 keys: ["gate", "down", "up"]
    pub fn load_only_tensor(&self, device: &Device) -> Result<Tensor> {
        let filename = self.parts.join(".");
        let path = self.root.join(&filename);

        //println!("Loading tensor from: {:?}", path); // 调试输出

        let data = std::fs::read(&path)?;
        let tensors: HashMap<String, Tensor> = load_buffer(&data, device)?;

        // 取出唯一的 tensor
        if tensors.len() != 1 {
            return Err(candle_core::Error::Msg(format!(
                "Expected exactly one tensor in file {:?}, but found {} keys: {:?}",
                path,
                tensors.len(),
                tensors.keys().collect::<Vec<_>>()
            )));
        }

        Ok(tensors.into_values().next().unwrap())
    }

    pub fn load_all(&self, device: &Device) -> Result<HashMap<String, Tensor>> {
        let filename = self.parts.join(".");
        let path = self.root.join(&filename);

        //println!("Loading tensors from: {:?}", path); // 调试输出

        let data = std::fs::read(&path)?;
        let tensors: HashMap<String, Tensor> = load_buffer(&data, device)?;
        Ok(tensors)
    }

    pub fn load_tensor(&self, key: &str, device: &Device) -> Result<Tensor> {
        let filename = self.parts.join(".");
        let path = self.root.join(&filename);

        //println!("Loading tensor from: {:?}", path); // 调试输出

        let data = std::fs::read(&path)?;
        let tensors: HashMap<String, Tensor> = load_buffer(&data, device)?;
        //默认键 "tensor"
        //let actual_key = if key.is_empty() { "tensor" } else { key };

        // tensors
        //     .get(actual_key)
        //     .cloned()
        //     .ok_or_else(|| candle_core::Error::Msg(format!("Key '{}' not found in {:?}", actual_key, path)))
        tensors
            .get(key)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("Key '{}' not found in {:?}", key, path)))
    }
}