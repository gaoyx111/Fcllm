use crate::configuration_qwen::Qwen2MoeConfig;
use crate::load::ExpertTensorLoader;
use crate::quantizer::dequantize;
use candle_core::{Device, Error, Result, Tensor};
use candle_nn::Activation;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug)]
pub struct Qwen2MoeMLP {
    pub device: Device,
    pub layer_idx: usize,
    pub quan_bit: usize,
    pub is_shared: bool,

    pub gate: Option<Tensor>,
    pub up: Option<Tensor>,
    pub down: Option<Tensor>,
    pub act_fn: Activation,

    pub idx: Option<usize>,

    pub gate_cpu: HashMap<usize, HashMap<String, Tensor>>,
    pub up_cpu: HashMap<usize, HashMap<String, Tensor>>,
    pub down_cpu: HashMap<usize, HashMap<String, Tensor>>,

    pub gate_device: HashMap<usize, HashMap<String, Tensor>>,
    pub up_device: HashMap<usize, HashMap<String, Tensor>>,
    pub down_device: HashMap<usize, HashMap<String, Tensor>>,
}

impl Qwen2MoeMLP {
    pub fn new(cfg: &Qwen2MoeConfig, layer_idx: usize, is_shared: bool) -> Self {
        let quan_bit = if layer_idx == 0 {
            0
        } else {
            cfg.quan_map[&layer_idx]
        };
        let actfn = map_activation(&cfg.hidden_act).unwrap();
        Self {
            device: cfg.device.clone(),
            layer_idx,
            quan_bit,
            is_shared,
            gate: None,
            up: None,
            down: None,
            act_fn: actfn,
            idx: None,
            gate_cpu: HashMap::new(),
            up_cpu: HashMap::new(),
            down_cpu: HashMap::new(),
            gate_device: HashMap::new(),
            up_device: HashMap::new(),
            down_device: HashMap::new(),
        }
    }

    pub fn init_weights(
        &mut self,
        path: &str,
        idx: Option<usize>,
        num_in_mem: Option<usize>,
    ) -> Result<()> {
        if idx.is_none() {
            let base = PathBuf::from(path);
            let original_dir = base.join("original");

            let filename_prefix = format!("model.layers.{}.mlp.shared_expert.", self.layer_idx);
            let gate_proj_file = format!("{}gate_proj.weight", filename_prefix);
            let up_proj_file = format!("{}up_proj.weight", filename_prefix);
            let down_proj_file = format!("{}down_proj.weight", filename_prefix);

            let gate_proj_path = original_dir.join(gate_proj_file);
            let up_proj_path = original_dir.join(up_proj_file);
            let down_proj_path = original_dir.join(down_proj_file);

            let loader1 = ExpertTensorLoader::new(gate_proj_path);
            let loader2 = ExpertTensorLoader::new(up_proj_path);
            let loader3 = ExpertTensorLoader::new(down_proj_path);

            self.gate = Some(loader1.load_tensor("tensor", &self.device)?);
            self.up = Some(loader2.load_tensor("tensor", &self.device)?);
            self.down = Some(loader3.load_tensor("tensor", &self.device)?);
        } else {
            self.idx = idx;
            let mut weight_path: HashMap<usize, PathBuf> = HashMap::new();
            let base_path = PathBuf::from(path); // path 是根目录字符串

            // locate fp16 weights
            weight_path.insert(
                0,
                base_path.join("original").join(format!(
                    "model.layers.{}.mlp.experts.{}.weight",
                    self.layer_idx,
                    idx.unwrap()
                )),
            );
            // locate int4/2 weights
            weight_path.insert(
                4,
                base_path.join("quantized").join("int4").join(format!(
                    "model.layers.{}.mlp.experts.{}.weight",
                    self.layer_idx,
                    idx.unwrap()
                )),
            );
            weight_path.insert(
                2,
                base_path.join("quantized").join("int2").join(format!(
                    "model.layers.{}.mlp.experts.{}.weight",
                    self.layer_idx,
                    idx.unwrap()
                )),
            );

            let init_device = Device::Cpu;
            // === 加载 FP16 权重 ===
            if self.quan_bit == 0 {
                let loader = ExpertTensorLoader::new(weight_path[&0].clone());
                let tensors = loader.load_all(&init_device)?;

                // 构造 HashMap<String, Tensor> 存入 CPU map
                let mut gate_map = HashMap::new();
                gate_map.insert("weight".to_string(), tensors["gate"].clone());
                self.gate_cpu.insert(0, gate_map);

                let mut up_map = HashMap::new();
                up_map.insert("weight".to_string(), tensors["up"].clone());
                self.up_cpu.insert(0, up_map);

                let mut down_map = HashMap::new();
                down_map.insert("weight".to_string(), tensors["down"].clone());
                self.down_cpu.insert(0, down_map);

                // 设置为当前激活参数
                self.gate = Some(tensors["gate"].to_device(&self.device)?);
                self.up = Some(tensors["up"].to_device(&self.device)?);
                self.down = Some(tensors["down"].to_device(&self.device)?);

                // 移除缓存
                self.gate_cpu.remove(&0);
                self.up_cpu.remove(&0);
                self.down_cpu.remove(&0);
            }
            // 加载 int4 和 int2 权重，并 extract_keys 三次（mutably）
            for &bit in &[4, 2] {
                let loader = ExpertTensorLoader::new(weight_path[&bit].clone());
                let mut tensors = loader.load_all(&init_device)?; // 可变 HashMap<String, Tensor>

                let gate_map = self.extract_keys("gate", &mut tensors);
                let up_map = self.extract_keys("up", &mut tensors);
                let down_map = self.extract_keys("down", &mut tensors);

                self.gate_cpu.insert(bit, gate_map);
                self.up_cpu.insert(bit, up_map);
                self.down_cpu.insert(bit, down_map);
            }

            // if let Some(limit) = num_in_mem {
            //     if idx.unwrap() < limit {
            //         if self.quan_bit != 0 {
            //             // 将 self.gate_cpu[quan_bit] 中的每个k,v移动到目标设备
            //             let gate_map = self.gate_cpu.get(&self.quan_bit).unwrap();
            //             let up_map = self.up_cpu.get(&self.quan_bit).unwrap();
            //             let down_map = self.down_cpu.get(&self.quan_bit).unwrap();

            //             let gate_device_map: HashMap<String, Tensor> = gate_map
            //                 .iter()
            //                 .map(|(k, v)| Ok((k.clone(), v.to_device(&self.device)?)))
            //                 .collect::<Result<_>>()?;
            //             self.gate_device.insert(self.quan_bit, gate_device_map);

            //             let up_device_map: HashMap<String, Tensor> = up_map
            //                 .iter()
            //                 .map(|(k, v)| Ok((k.clone(), v.to_device(&self.device)?)))
            //                 .collect::<Result<_>>()?;
            //             self.up_device.insert(self.quan_bit, up_device_map);

            //             let down_device_map: HashMap<String, Tensor> = down_map
            //                 .iter()
            //                 .map(|(k, v)| Ok((k.clone(), v.to_device(&self.device)?)))
            //                 .collect::<Result<_>>()?;
            //             self.down_device.insert(self.quan_bit, down_device_map);
            //         }
            //     }
            // }
        }
        Ok(())
    }

    pub fn extract_keys(
        &self,
        prefix: &str,
        tensors: &mut HashMap<String, Tensor>,
    ) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();

        if let Some(nbits) = tensors.remove(&format!("{}_nbits", prefix)) {
            map.insert("nbits".to_string(), nbits);
        }
        if let Some(shape) = tensors.remove(&format!("{}_shape", prefix)) {
            map.insert("shape".to_string(), shape);
        }
        if let Some(wq) = tensors.remove(prefix) {
            map.insert("W_q".to_string(), wq);
        }
        if let Some(scale) = tensors.remove(&format!("{}_scale", prefix)) {
            map.insert("scale".to_string(), scale);
        }
        if let Some(zero) = tensors.remove(&format!("{}_zero", prefix)) {
            map.insert("zero".to_string(), zero);
        }

        map
    }

    pub fn load_from_cpu(
        &self,
        weight: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut result = HashMap::new();
        // 不移动 nbits / shape，它们通常是元信息
        if let Some(nbits) = weight.get("nbits") {
            result.insert("nbits".to_string(), nbits.clone());
        }
        if let Some(shape) = weight.get("shape") {
            result.insert("shape".to_string(), shape.clone());
        }
        // 移动权重到目标 device
        if let Some(wq) = weight.get("W_q") {
            result.insert("W_q".to_string(), wq.to_device(&self.device)?);
        }
        if let Some(scale) = weight.get("scale") {
            result.insert("scale".to_string(), scale.to_device(&self.device)?);
        }
        if let Some(zero) = weight.get("zero") {
            result.insert("zero".to_string(), zero.to_device(&self.device)?);
        }

        Ok(result)
    }

    pub fn load_weights(&mut self, is_now: bool, nbit: Option<usize>) -> Result<()> {
        let quan_bit = nbit.unwrap_or(self.quan_bit);

        // 从 CPU 缓存中获取指定位宽的权重 map
        let gate_cpu = self
            .gate_cpu
            .get(&quan_bit)
            .ok_or_else(|| Error::Msg(format!("Missing gate_cpu for quan_bit {}", quan_bit)))?;
        let up_cpu = self
            .up_cpu
            .get(&quan_bit)
            .ok_or_else(|| Error::Msg(format!("Missing up_cpu for quan_bit {}", quan_bit)))?;
        let down_cpu = self
            .down_cpu
            .get(&quan_bit)
            .ok_or_else(|| Error::Msg(format!("Missing down_cpu for quan_bit {}", quan_bit)))?;

        // 只取出 W_q 并移动到目标 device
        self.gate = Some(
            gate_cpu
                .get("W_q")
                .ok_or_else(|| Error::Msg("Missing W_q in gate".into()))?
                .to_device(&self.device)?,
        );

        self.up = Some(
            up_cpu
                .get("W_q")
                .ok_or_else(|| Error::Msg("Missing W_q in up".into()))?
                .to_device(&self.device)?,
        );

        self.down = Some(
            down_cpu
                .get("W_q")
                .ok_or_else(|| Error::Msg("Missing W_q in down".into()))?
                .to_device(&self.device)?,
        );

        Ok(())
    }

    pub fn dequan_experts(&mut self) -> Result<()> {
        if !self.is_shared && self.quan_bit != 0 {
            let quan_bit = self.quan_bit;

            // 从 CPU 缓存中获取对应 bit 的权重 map
            let gate_cpu = self
                .gate_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing gate_cpu for quan_bit".into()))?;
            let up_cpu = self
                .up_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing up_cpu for quan_bit".into()))?;
            let down_cpu = self
                .down_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing down_cpu for quan_bit".into()))?;

            // 调用 dequantize 解码
            let gate_dequan = dequantize(&gate_cpu)?;
            let up_dequan = dequantize(&up_cpu)?;
            let down_dequan = dequantize(&down_cpu)?;

            // 反量化后立刻搬到目标设备（GPU），避免 forward 时每次都做 CPU→GPU 拷贝
            self.gate = Some(gate_dequan.to_device(&self.device)?);
            self.up = Some(up_dequan.to_device(&self.device)?);
            self.down = Some(down_dequan.to_device(&self.device)?);
        }
        Ok(())
    }

    pub fn quan_experts(&mut self) -> Result<()> {
        if self.quan_bit != 0 {
            let quan_bit = self.quan_bit;

            // 从 CPU 缓存中获取对应 bit 的权重 map
            let gate_cpu = self
                .gate_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing gate_cpu for quan_bit".into()))?;
            let up_cpu = self
                .up_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing up_cpu for quan_bit".into()))?;
            let down_cpu = self
                .down_cpu
                .get(&quan_bit)
                .ok_or_else(|| candle_core::Error::Msg("Missing down_cpu for quan_bit".into()))?;

            // 调用 load_from_cpu 移动到 device（返回的是 HashMap<String, Tensor>）
            let gate_loaded = self.load_from_cpu(gate_cpu)?;
            let up_loaded = self.load_from_cpu(up_cpu)?;
            let down_loaded = self.load_from_cpu(down_cpu)?;

            // 提取 W_q 张量赋值
            self.gate = Some(
                gate_loaded
                    .get("W_q")
                    .ok_or_else(|| candle_core::Error::Msg("Missing W_q in gate_loaded".into()))?
                    .clone(),
            );
            self.up = Some(
                up_loaded
                    .get("W_q")
                    .ok_or_else(|| candle_core::Error::Msg("Missing W_q in up_loaded".into()))?
                    .clone(),
            );
            self.down = Some(
                down_loaded
                    .get("W_q")
                    .ok_or_else(|| candle_core::Error::Msg("Missing W_q in down_loaded".into()))?
                    .clone(),
            );
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.gate = None;
        self.up = None;
        self.down = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing gate tensor".to_string()))?
            .to_dtype(x.dtype())?;
        let up = self
            .up
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing up tensor".to_string()))?
            .to_dtype(x.dtype())?;
        let down = self
            .down
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing down tensor".to_string()))?
            .to_dtype(x.dtype())?;

        let gate_out = x.matmul(&gate.transpose(0, 1)?)?;
        let gate_act = candle_nn::ops::silu(&gate_out)?;
        let up_out = x.matmul(&up.transpose(0, 1)?)?;
        let fused = gate_act.mul(&up_out)?;
        let out = fused.matmul(&down.transpose(0, 1)?)?;
        Ok(out)
    }
}

pub fn map_activation(name: &str) -> Result<Activation> {
    match name.to_lowercase().as_str() {
        "gelu" => Ok(Activation::Gelu),
        "relu" => Ok(Activation::Relu),
        "silu" | "swish" => Ok(Activation::Silu),
        _ => Err(Error::Msg(format!("Unknown activation function: {}", name))),
    }
}
