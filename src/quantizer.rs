use candle_core::{Result, Tensor, DType};
use std::collections::HashMap;
use crate::utils::unpack;

/* pub fn dequantize(quantized_data: &HashMap<String, Tensor>) -> Result<Tensor> {
    let w_q = quantized_data.get("W_q").ok_or_else(|| candle_core::Error::Msg("Missing W_q".into()))?;
    let scale = quantized_data.get("scale").ok_or_else(|| candle_core::Error::Msg("Missing scale".into()))?;
    let zero = quantized_data.get("zero").ok_or_else(|| candle_core::Error::Msg("Missing zero".into()))?;
    let shape = quantized_data.get("shape").ok_or_else(|| candle_core::Error::Msg("Missing shape".into()))?;
    let nbits = quantized_data.get("nbits").ok_or_else(|| candle_core::Error::Msg("Missing nbits".into()))?;

    let nbits = nbits.to_scalar::<u8>()? as usize;

    let w_r = unpack(w_q, nbits, DType::BF16)?;
    let w_r = w_r.sub(zero)?;
    let w_r = w_r.mul(scale)?;
    let shape_dims_i64 = shape.to_vec1::<i64>()?;
    let shape_dims: Vec<usize> = shape_dims_i64.iter().map(|&x| x as usize).collect();
    let w_r = w_r.reshape(shape_dims.as_slice())?.to_dtype(DType::BF16)?;

    Ok(w_r)
} */

pub fn dequantize(quantized_data: &HashMap<String, Tensor>) -> Result<Tensor> {
    let w_q = quantized_data.get("W_q").ok_or(candle_core::Error::Msg("Missing key: W_q".into()))?;

    let mut scale = quantized_data.get("scale").ok_or(candle_core::Error::Msg("Missing key: scale".into()))?.clone();
    if scale.dtype() != DType::BF16 {
        scale = scale.to_dtype(DType::BF16)?;
    }

    let mut zero = quantized_data.get("zero").ok_or(candle_core::Error::Msg("Missing key: zero".into()))?.clone();
    if zero.dtype() != DType::BF16 {
        zero = zero.to_dtype(DType::BF16)?;
    }

    let mut shape = quantized_data.get("shape").ok_or(candle_core::Error::Msg("Missing key: shape".into()))?.clone();
    if shape.dtype() != DType::I64 {
        shape = shape.to_dtype(DType::I64)?;
    }
    let shape_dims_i64 = shape.to_vec1::<i64>()?;
    let shape_dims: Vec<usize> = shape_dims_i64.iter().map(|&x| x as usize).collect();

    let mut nbits = quantized_data.get("nbits").ok_or(candle_core::Error::Msg("Missing key: nbits".into()))?.clone();
    if nbits.dtype() != DType::U8 {
        nbits = nbits.to_dtype(DType::U8)?;
    }
    let nbits = nbits.to_scalar::<u8>()? as usize;

    let mut w_r = unpack(w_q, nbits, DType::BF16)?;
    let zero = zero.broadcast_as(w_r.shape())?;
    let scale = scale.broadcast_as(w_r.shape())?;
    w_r = w_r.sub(&zero)?;
    w_r = w_r.mul(&scale)?;
    w_r = w_r.reshape(shape_dims.as_slice())?.to_dtype(DType::BF16)?;

    Ok(w_r)
}