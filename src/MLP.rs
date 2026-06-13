use crate::configuration_qwen::Qwen2MoeConfig;
use crate::load::ExpertTensorLoader;
use crate::quantizer::dequantize;
use crate::utils::cpu_dequant_cache_has_memory_headroom;
use candle_core::{DType, Device, Error, Result, Tensor};
use candle_nn::Activation;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static CPU_DEQUANT_CACHE_PAUSED_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
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
    gate_t: Option<Tensor>,
    up_t: Option<Tensor>,
    down_t: Option<Tensor>,
    gate_dequant_cpu_t: Option<Tensor>,
    up_dequant_cpu_t: Option<Tensor>,
    down_dequant_cpu_t: Option<Tensor>,
}

impl Qwen2MoeMLP {
    pub fn new(cfg: &Qwen2MoeConfig, layer_idx: usize, is_shared: bool) -> Self {
        let quan_bit = cfg.quan_map[&layer_idx];
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
            gate_t: None,
            up_t: None,
            down_t: None,
            gate_dequant_cpu_t: None,
            up_dequant_cpu_t: None,
            down_dequant_cpu_t: None,
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

            self.set_weight_tensors(
                loader1.load_tensor("tensor", &self.device)?,
                loader2.load_tensor("tensor", &self.device)?,
                loader3.load_tensor("tensor", &self.device)?,
            )?;
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
                self.set_weight_tensors(
                    tensors["gate"].to_device(&self.device)?,
                    tensors["up"].to_device(&self.device)?,
                    tensors["down"].to_device(&self.device)?,
                )?;

                // 移除缓存
                self.gate_cpu.remove(&0);
                self.up_cpu.remove(&0);
                self.down_cpu.remove(&0);

                // Full-precision resident experts never need the quantized
                // side copies. Skipping them saves startup time and RAM for
                // the early hot layer.
                return Ok(());
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
        let gate = gate_cpu
            .get("W_q")
            .ok_or_else(|| Error::Msg("Missing W_q in gate".into()))?
            .to_device(&self.device)?;

        let up = up_cpu
            .get("W_q")
            .ok_or_else(|| Error::Msg("Missing W_q in up".into()))?
            .to_device(&self.device)?;

        let down = down_cpu
            .get("W_q")
            .ok_or_else(|| Error::Msg("Missing W_q in down".into()))?
            .to_device(&self.device)?;

        self.set_weight_tensors(gate, up, down)?;

        Ok(())
    }

    fn set_weight_tensors(&mut self, gate: Tensor, up: Tensor, down: Tensor) -> Result<()> {
        let gate_t = gate.transpose(0, 1)?;
        let up_t = up.transpose(0, 1)?;
        let down_t = down.transpose(0, 1)?;

        self.set_transposed_weight_tensors(gate_t, up_t, down_t)
    }

    fn set_transposed_weight_tensors(
        &mut self,
        gate_t: Tensor,
        up_t: Tensor,
        down_t: Tensor,
    ) -> Result<()> {
        // Forward only uses transposed weights. Dropping the original handles
        // avoids keeping extra tensor views alive for short-lived expert loads.
        self.gate = None;
        self.up = None;
        self.down = None;
        self.gate_t = Some(gate_t);
        self.up_t = Some(up_t);
        self.down_t = Some(down_t);

        Ok(())
    }

    pub fn dequan_experts(&mut self) -> Result<()> {
        if !self.is_shared && self.quan_bit != 0 {
            if self.has_dequantized_weights_on_device() {
                return Ok(());
            }

            self.warm_dequant_cpu_cache()?;

            // 反量化后立刻搬到目标设备（GPU），避免 forward 时每次都做 CPU→GPU 拷贝
            if let (Some(gate), Some(up), Some(down)) = (
                self.gate_dequant_cpu_t.as_ref(),
                self.up_dequant_cpu_t.as_ref(),
                self.down_dequant_cpu_t.as_ref(),
            ) {
                self.set_transposed_weight_tensors(
                    gate.to_device(&self.device)?,
                    up.to_device(&self.device)?,
                    down.to_device(&self.device)?,
                )?;
            } else {
                let (gate_dequan, up_dequan, down_dequan) = self.dequantize_cpu_weights()?;
                self.set_transposed_weight_tensors(
                    gate_dequan.transpose(0, 1)?.to_device(&self.device)?,
                    up_dequan.transpose(0, 1)?.to_device(&self.device)?,
                    down_dequan.transpose(0, 1)?.to_device(&self.device)?,
                )?;
            }
        }
        Ok(())
    }

    pub fn warm_dequant_cpu_cache(&mut self) -> Result<bool> {
        if self.is_shared || self.quan_bit == 0 || self.has_dequant_cpu_cache() {
            return Ok(false);
        }

        if !cpu_dequant_cache_has_memory_headroom() {
            if !CPU_DEQUANT_CACHE_PAUSED_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[moe-cache] CPU dequant cache paused: system free memory is below safety margin"
                );
            }
            return Ok(false);
        }

        let (gate_dequan, up_dequan, down_dequan) = self.dequantize_cpu_weights()?;
        let gate_t = gate_dequan.transpose(0, 1)?;
        let up_t = up_dequan.transpose(0, 1)?;
        let down_t = down_dequan.transpose(0, 1)?;

        self.gate_dequant_cpu_t = Some(gate_t);
        self.up_dequant_cpu_t = Some(up_t);
        self.down_dequant_cpu_t = Some(down_t);
        Ok(true)
    }

    fn has_dequant_cpu_cache(&self) -> bool {
        self.gate_dequant_cpu_t.is_some()
            && self.up_dequant_cpu_t.is_some()
            && self.down_dequant_cpu_t.is_some()
    }

    fn dequantize_cpu_weights(&self) -> Result<(Tensor, Tensor, Tensor)> {
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

        Ok((
            dequantize(&gate_cpu)?,
            dequantize(&up_cpu)?,
            dequantize(&down_cpu)?,
        ))
    }

    pub fn has_dequantized_weights_on_device(&self) -> bool {
        let Some(gate) = self.gate_t.as_ref() else {
            return false;
        };
        let Some(up) = self.up_t.as_ref() else {
            return false;
        };
        let Some(down) = self.down_t.as_ref() else {
            return false;
        };

        gate.dtype() != DType::U8 && up.dtype() != DType::U8 && down.dtype() != DType::U8
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
            let gate = gate_loaded
                .get("W_q")
                .ok_or_else(|| candle_core::Error::Msg("Missing W_q in gate_loaded".into()))?
                .clone();
            let up = up_loaded
                .get("W_q")
                .ok_or_else(|| candle_core::Error::Msg("Missing W_q in up_loaded".into()))?
                .clone();
            let down = down_loaded
                .get("W_q")
                .ok_or_else(|| candle_core::Error::Msg("Missing W_q in down_loaded".into()))?
                .clone();
            self.set_weight_tensors(gate, up, down)?;
        }

        Ok(())
    }

    pub fn clear(&mut self) {
        self.gate = None;
        self.up = None;
        self.down = None;
        self.gate_t = None;
        self.up_t = None;
        self.down_t = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_ref = self
            .gate_t
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing gate transpose tensor".to_string()))?;
        let up_ref = self
            .up_t
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing up transpose tensor".to_string()))?;
        let down_ref = self
            .down_t
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("Missing down transpose tensor".to_string()))?;

        forward_with_transposed_weights(x, gate_ref, up_ref, down_ref)
    }
}

fn forward_with_transposed_weights(
    x: &Tensor,
    gate_ref: &Tensor,
    up_ref: &Tensor,
    down_ref: &Tensor,
) -> Result<Tensor> {
    let gate_owned;
    let gate = if gate_ref.dtype() == x.dtype() {
        gate_ref
    } else {
        gate_owned = gate_ref.to_dtype(x.dtype())?;
        &gate_owned
    };

    let up_owned;
    let up = if up_ref.dtype() == x.dtype() {
        up_ref
    } else {
        up_owned = up_ref.to_dtype(x.dtype())?;
        &up_owned
    };

    let down_owned;
    let down = if down_ref.dtype() == x.dtype() {
        down_ref
    } else {
        down_owned = down_ref.to_dtype(x.dtype())?;
        &down_owned
    };

    let gate_out = x.matmul(gate)?;
    let gate_act = candle_nn::ops::silu(&gate_out)?;
    let up_out = x.matmul(up)?;
    let fused = gate_act.mul(&up_out)?;
    fused.matmul(down)
}

pub fn map_activation(name: &str) -> Result<Activation> {
    match name.to_lowercase().as_str() {
        "gelu" => Ok(Activation::Gelu),
        "relu" => Ok(Activation::Relu),
        "silu" | "swish" => Ok(Activation::Silu),
        _ => Err(Error::Msg(format!("Unknown activation function: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_transposes_follow_weight_lifecycle() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = Qwen2MoeConfig::new();
        cfg.device = device.clone();
        cfg.quan_map.insert(0, 4);

        let mut mlp = Qwen2MoeMLP::new(&cfg, 0, true);
        mlp.set_weight_tensors(
            Tensor::zeros((3, 2), DType::F32, &device)?,
            Tensor::zeros((3, 2), DType::F32, &device)?,
            Tensor::zeros((2, 3), DType::F32, &device)?,
        )?;

        assert_eq!(mlp.gate_t.as_ref().unwrap().dims(), &[2, 3]);
        assert_eq!(mlp.up_t.as_ref().unwrap().dims(), &[2, 3]);
        assert_eq!(mlp.down_t.as_ref().unwrap().dims(), &[3, 2]);

        mlp.clear();
        assert!(mlp.gate.is_none());
        assert!(mlp.up.is_none());
        assert!(mlp.down.is_none());
        assert!(mlp.gate_t.is_none());
        assert!(mlp.up_t.is_none());
        assert!(mlp.down_t.is_none());

        Ok(())
    }
}
