use candle_core::{Tensor, DType, Result};
use std::collections::HashMap;
use crate::utils::*;
use crate::models::Qwen::modeling_qwen_moe::TensorOrMap;

// Supported bit widths
const SUPPORTED_BITS: &[usize] = &[1, 2, 3, 4, 8];

// Mapping from bit width to function tag
fn bit_to_packing(bits: usize) -> &'static str {
    match bits {
        8 => "8bit",
        4 => "4bit",
        3 => "3bit",
        2 => "2bit",
        1 => "1bit",
        _ => panic!("Unsupported bit width: {}", bits),
    }
}

fn is_divisible(val1: usize, val2: usize) -> bool {
    (val2 * ((val1 + val2 - 1) / val2)) == val1
}

pub fn quantize(tensor: &Tensor, nbits: usize, group_size: Option<usize>) -> Result<HashMap<&'static str, Tensor>> {
    if !SUPPORTED_BITS.contains(&nbits) {
        return Err(candle_core::Error::Msg(format!("nbits={} not supported.", nbits)));
    }

    let group_size = match nbits {
        4 => Some(64),
        2 => Some(32),
        _ => group_size,
    }.ok_or_else(|| candle_core::Error::Msg("Group size must be specified for nbits".into()))?;

    let shape = tensor.shape().clone();
    let w = tensor.to_dtype(DType::F32)?;
    let numel = w.elem_count();

    if numel % group_size != 0 {
        return Err(candle_core::Error::Msg("group_size must divide total tensor elements".into()));
    }

    let w = w.reshape((group_size, numel / group_size))?;
    let (_min, _max) = (w.min(0)?, w.max(0)?);

    let max_v = (1 << nbits) - 1;
    let min_v = 0;
    let scale = ((max_v as f64) / (&_max - &_min)).clamp_max(2e4)?;
    let zero = -(&_min * &scale)?;

    let w_q = w.mul(&scale)?.add(&zero)?.round()?.clamp(min_v as f64, max_v as f64)?;

    let packed = match nbits {
        8 => pack_8bit_u8(&w_q)?,
        4 => pack_4bit_u8(&w_q)?,
        3 => pack_3bit_32(&w_q)?,
        2 => pack_2bit_u8(&w_q)?,
        1 => pack_1bit_u8(&w_q)?,
        _ => unreachable!(),
    };

    Ok([
        ("nbits", Tensor::new(nbits as u32, w.device())?),
        ("shape", Tensor::new(shape.dims().to_vec(), w.device())?),
        ("W_q", packed),
        ("scale", scale.recip()? /* 1 / scale */),
        ("zero", zero),
    ].into_iter().collect())
}

pub fn dequantize(tensor_map: &TensorOrMap) -> Result<Tensor> {
    match tensor_map {
        TensorOrMap::Map(map) => {
            let nbits = usize::try_from(map["nbits"].to_scalar::<u32>()?)?;
            let shape_dims: Vec<usize> = map["shape"].to_vec1::<usize>()?;
            let compute_dtype = DType::BF16;

            let packing = bit_to_packing(nbits);
            let w_q = &map["W_q"];
            let scale = &map["scale"];
            let zero = &map["zero"];

            let unpacked = match packing {
                "8bit" => unpack_8bit_u8(w_q)?,
                "4bit" => unpack_4bit_u8(w_q)?,
                "3bit" => unpack_3bit_32(w_q)?,
                "2bit" => unpack_2bit_u8(w_q)?,
                "1bit" => unpack_1bit_u8(w_q)?,
                _ => return Err("Unsupported bit width".into()),
            };

            let restored = (unpacked.sub(zero)?.mul(scale)?)
                .reshape(shape_dims)?
                .to_dtype(compute_dtype)?;

            Ok(restored)
        }
        TensorOrMap::Single(_) => Err("dequantize expects TensorOrMap::Map, but got Single".into()),
    }
}

