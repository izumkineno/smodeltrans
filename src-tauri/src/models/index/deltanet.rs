//! Gated DeltaNet 单 token recurrent 前向（decode 路径）。
//!
//! 直译 transformers `Qwen3NextGatedDeltaNet.forward` 的 cached-decode 分支
//! + llama.cpp `build_delta_net_autoregressive`：
//!   S_t = S_{t-1} * exp(g) + beta * (v - S_{t-1}ᵀ k) ⊗ k  （per value head）
//!   o = S_t q ；qk 先 L2Norm；输出经 RMSNormGated(z) + out_proj。
//!
//! P0 取证 dims（Index-2B）：hidden 2048，qkv 6144（= 2*key_dim + 2*value_dim 的 fused 形态），
//! num_v_heads = 16（`ssm_a/dt [16]`），key head dim 由 `ssm_norm [128]` 得 value head dim 128，
//! conv kernel 4。prefill chunk 路径后置（P2b），此处只走 recurrent。

use super::model::LinearLayerWeights;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops::{sigmoid as tensor_sigmoid, silu};

pub(crate) struct DeltaNetState {
    pub conv_hist: Tensor,
    pub conv_w: Tensor,
    pub ssm: Tensor,
    pub ssm_a: Tensor,
    pub dt_bias: Tensor,
    pub norm_weight: Tensor,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
    pub qkv_dim: usize,
}

impl DeltaNetState {
    pub(crate) fn zeros(
        num_v_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
        qkv_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv_hist: Tensor::zeros((3, qkv_dim), DType::F32, device)?,
            conv_w: Tensor::zeros((4, qkv_dim), DType::F32, device)?,
            ssm: Tensor::zeros(
                (num_v_heads, head_v_dim, head_k_dim),
                DType::F32,
                device,
            )?,
            ssm_a: Tensor::zeros(num_v_heads, DType::F32, device)?,
            dt_bias: Tensor::zeros(num_v_heads, DType::F32, device)?,
            norm_weight: Tensor::zeros(head_v_dim, DType::F32, device)?,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            qkv_dim,
        })
    }
}

#[allow(dead_code)]
fn l2norm_row(v: &[f32]) -> Vec<f32> {
    let inv_norm = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
    v.iter().map(|x| x * inv_norm).collect()
}

fn l2norm_rows(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(1)?.affine(1.0, 1e-6)?.sqrt()?;
    Ok(x.broadcast_div(&n)?)
}

fn softplus_t(x: &Tensor) -> Result<Tensor> {
    let stable = x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(x.relu()?.broadcast_add(&stable)?)
}

#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(dead_code)]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

#[allow(dead_code)]
fn decay_factor(ssm_a: f32, alpha_plus_bias: f32) -> f32 {
    (ssm_a * softplus(alpha_plus_bias)).exp()
}

/// batch DeltaNet recurrent 前向；hidden (S, 2048) F16 GPU in / out。
/// 投影一次 batch 算完，CPU 逐 step 递推（conv 状态天然串行，S=1 即单步解码）。
pub(crate) fn delta_step(
    layer: &LinearLayerWeights,
    hidden: &Tensor,
    state: &mut DeltaNetState,
    device: &Device,
    prof: &mut super::session::StepProfile,
) -> Result<Tensor> {
    let t0 = super::session::prof_snap(prof, device);
    let qkv = layer
        .attn_qkv
        .forward(hidden)?
        .to_dtype(DType::F32)?;
    let z = layer
        .attn_gate
        .forward(hidden)?
        .to_dtype(DType::F32)?;
    let ba = layer
        .ssm_alpha
        .forward(hidden)?
        .to_dtype(DType::F32)?;
    let beta_logits = layer
        .ssm_beta
        .forward(hidden)?
        .to_dtype(DType::F32)?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    let s = hidden.dim(0)?;
    let (nh, vd, kd) = (state.num_v_heads, state.head_v_dim, state.head_k_dim);
    let key_dim = nh * kd;
    let value_dim = nh * vd;
    anyhow::ensure!(
        qkv.dim(1)? == state.qkv_dim,
        "conv state/projection width mismatch"
    );
    let inv_sqrt_k = 1.0 / (kd as f32).sqrt();
    let mut outs: Vec<Tensor> = Vec::with_capacity(s);
    let t0 = super::session::prof_snap(prof, device);
    for i in 0..s {

        let qkv_i = qkv.narrow(0, i, 1)?;
        let window = Tensor::cat(&[&state.conv_hist, &qkv_i], 0)?;
        let convolved = window.broadcast_mul(&state.conv_w)?.sum_keepdim(0)?;
        let conv_out = silu(&convolved)?;
        state.conv_hist = window.narrow(0, 1, 3)?.contiguous()?;
        let q = conv_out
            .narrow(1, 0, key_dim)?
            .contiguous()?
            .reshape((nh, kd))?;
        let k = conv_out
            .narrow(1, key_dim, key_dim)?
            .contiguous()?
            .reshape((nh, kd))?;
        let v = conv_out
            .narrow(1, 2 * key_dim, value_dim)?
            .contiguous()?
            .reshape((nh, vd))?;
        let z_h = z
            .narrow(0, i, 1)?
            .contiguous()?
            .reshape((nh, vd))?;
        let ba_i = ba.narrow(0, i, 1)?.contiguous()?;
        let beta_i = beta_logits.narrow(0, i, 1)?.contiguous()?;
        let qn = l2norm_rows(&q)?.affine(inv_sqrt_k as f64, 0.0)?;
        let kn = l2norm_rows(&k)?;
        let beta = tensor_sigmoid(&beta_i)?.reshape((nh, 1, 1))?;
        let sp = softplus_t(&ba_i.broadcast_add(&state.dt_bias)?)?;
        let decay = state
            .ssm_a
            .broadcast_mul(&sp)?
            .exp()?
            .reshape((nh, 1, 1))?;
        let k3 = kn.unsqueeze(1)?;
        let q3 = qn.unsqueeze(1)?;
        let pred = state.ssm.broadcast_mul(&k3)?.sum_keepdim(2)?;
        let vcol = v.reshape((nh, vd, 1))?;
        let delta = beta.broadcast_mul(&vcol.sub(&pred)?)?;
        state.ssm = state
            .ssm
            .broadcast_mul(&decay)?
            .broadcast_add(&delta.broadcast_mul(&k3)?)?;
        let o = state
            .ssm
            .broadcast_mul(&q3)?
            .sum_keepdim(2)?
            .reshape((nh, vd))?;
        let ms = o.sqr()?.sum_keepdim(1)?.affine(1.0 / vd as f64, 1e-6)?;
        let scale = ms.powf(-0.5)?;
        let gate = tensor_sigmoid(&z_h)?;
        let out = o
            .broadcast_mul(&scale)?
            .broadcast_mul(&state.norm_weight)?
            .broadcast_mul(&z_h)?
            .broadcast_mul(&gate)?;
        outs.push(out.reshape((1, value_dim))?.contiguous()?);
    }
    super::session::prof_acc(&mut prof.delta_ms, t0);
    let t0 = super::session::prof_snap(prof, device);
    let cores = Tensor::cat(&outs, 0)?;
    let gated = cores.to_dtype(DType::F16)?;
    let out = layer.ssm_out.forward(&gated)?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_sanity() {
        let v = l2norm_row(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-5 && (v[1] - 0.8).abs() < 1e-5);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(softplus(0.0) > 0.69 && softplus(0.0) < 0.70);
    }

    #[test]
    fn decay_factor_matches_gguf_ssm_a_semantics() {
        // GGUF blk.0.ssm_a 首值 -0.783377（已是 -exp(A_log)，负数）。
        let decay = decay_factor(-0.783377, 0.0);
        let expected = (-0.783377 * std::f32::consts::LN_2).exp();
        assert!((decay - expected).abs() < 1e-6, "{decay} vs {expected}");
        assert!(decay > 0.0 && decay < 1.0);
        // 与误用 `-exp(ssm_a)` 的旧公式拉开差距（旧值约 0.728，新值约 0.581）。
        assert!((decay - 0.581).abs() < 0.01, "{decay}");
    }
}
