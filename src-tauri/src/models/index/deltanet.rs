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

/// 单层 recurrent state：conv ring（(k-1) × qkv_dim）+ SSM 矩阵（num_v × head_vd × head_kd）
/// + 静态小权重（ssm_a/dt_bias/norm，逐 translate 取一次）。
pub(crate) struct DeltaNetState {
    pub conv_hist: Vec<Vec<f32>>,
    pub ssm: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub conv_weight: Vec<Vec<f32>>,
    pub conv_transposed: bool,
    pub scratch_conv: Vec<f32>,
    pub scratch_q: Vec<f32>,
    pub scratch_k: Vec<f32>,
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
    ) -> Self {
        Self {
            conv_hist: vec![vec![0.0; qkv_dim]; 3],
            ssm: vec![0.0; num_v_heads * head_v_dim * head_k_dim],
            ssm_a: Vec::new(),
            dt_bias: Vec::new(),
            norm_weight: Vec::new(),
            conv_weight: Vec::new(),
            conv_transposed: false,
            scratch_conv: vec![0.0; qkv_dim],
            scratch_q: vec![0.0; head_k_dim],
            scratch_k: vec![0.0; head_k_dim],
            num_v_heads,
            head_v_dim,
            head_k_dim,
            qkv_dim,
        }
    }
}

#[allow(dead_code)]
fn l2norm_row(v: &[f32]) -> Vec<f32> {
    let inv_norm = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
    v.iter().map(|x| x * inv_norm).collect()
}

fn l2norm_into(dst: &mut [f32], src: &[f32]) {
    let mut sum = 0.0f32;
    for v in src {
        sum += v * v;
    }
    let inv = 1.0 / (sum + 1e-6).sqrt();
    for (d, v) in dst.iter_mut().zip(src.iter()) {
        *d = v * inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

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
    let qkv = layer.attn_qkv.forward(hidden)?;
    let z = layer.attn_gate.forward(hidden)?;
    let ba = layer.ssm_alpha.forward(hidden)?;
    let beta_logits = layer.ssm_beta.forward(hidden)?;
    let key_dim = state.num_v_heads * state.head_k_dim;
    let value_dim = state.num_v_heads * state.head_v_dim;
    let rows = Tensor::cat(&[qkv, z, ba, beta_logits], 1)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    let n_steps = rows.len();
    let row_width = 2 * key_dim + 2 * value_dim + 2 * state.num_v_heads;
    let mut cores = Vec::with_capacity(n_steps * value_dim);
    let t0 = super::session::prof_snap(prof, device);
    for row in &rows {
        anyhow::ensure!(
            row.len() == row_width,
            "unexpected fused QKV/Z/A/B width {}",
            row.len()
        );
        let (qkv_raw, rest) = row.split_at(2 * key_dim + value_dim);
        let (z_raw, rest) = rest.split_at(value_dim);
        let (ba_raw, beta_raw) = rest.split_at(state.num_v_heads);
        anyhow::ensure!(
            state.qkv_dim == qkv_raw.len(),
            "conv state/projection width mismatch"
        );

        // Causal depthwise Conv1d: each channel consumes its oldest-to-newest 4-tap window.
        // conv 权重已在 new_states 预取（逐 translate 一次），此处零同步。
        let weights = &state.conv_weight;
        let transposed = state.conv_transposed;
        anyhow::ensure!(
            if transposed {
                weights.len() == qkv_raw.len() && weights.iter().all(|row| row.len() == 4)
            } else {
                weights.iter().all(|row| row.len() == qkv_raw.len())
            },
            "unexpected conv1d weight shape"
        );
        let convolved = &mut state.scratch_conv[..qkv_raw.len()];
        for (channel, slot) in convolved.iter_mut().enumerate() {
            let taps = [
                state.conv_hist[0][channel],
                state.conv_hist[1][channel],
                state.conv_hist[2][channel],
                qkv_raw[channel],
            ];
            let mut acc = 0.0;
            for k in 0..4 {
                let weight = if transposed {
                    weights[channel][k]
                } else {
                    weights[k][channel]
                };
                acc += taps[k] * weight;
            }
            *slot = acc * sigmoid(acc);
        }
        let (hist_0, rest) = state.conv_hist.split_at_mut(1);
        let (hist_1, hist_2) = rest.split_at_mut(1);
        hist_0[0].clone_from(&hist_1[0]);
        hist_1[0].clone_from(&hist_2[0]);
        state.conv_hist[2].copy_from_slice(qkv_raw);

        let (q_raw, rest) = convolved.split_at(key_dim);
        let (k_raw, v_raw) = rest.split_at(key_dim);
    let (ssm_a, dt_bias, norm_weight) = (&state.ssm_a, &state.dt_bias, &state.norm_weight);
    let inv_sqrt_k = 1.0 / (state.head_k_dim as f32).sqrt();
    let mut core = vec![0.0f32; value_dim];

    for vh in 0..state.num_v_heads {
        let head = vh * state.head_k_dim..(vh + 1) * state.head_k_dim;
        l2norm_into(&mut state.scratch_q, &q_raw[head.clone()]);
        l2norm_into(&mut state.scratch_k, &k_raw[head]);
        for value in state.scratch_q.iter_mut() {
            *value *= inv_sqrt_k;
        }
        let q: &[f32] = &state.scratch_q;
        let k: &[f32] = &state.scratch_k;
            let v = &v_raw[vh * state.head_v_dim..(vh + 1) * state.head_v_dim];
            let beta = sigmoid(beta_raw[vh]);
            let decay = decay_factor(ssm_a[vh], ba_raw[vh] + dt_bias[vh]);
            let base = vh * state.head_v_dim * state.head_k_dim;

            // State is stored [value_dim, key_dim], equivalent to the reference [key_dim, value_dim].
            for row in 0..state.head_v_dim {
                let row_start = base + row * state.head_k_dim;
                let prediction = state.ssm[row_start..row_start + state.head_k_dim]
                    .iter()
                    .zip(k)
                    .map(|(s, key)| s * key)
                    .sum::<f32>();
                let delta = beta * (v[row] - prediction);
                for col in 0..state.head_k_dim {
                    let idx = row_start + col;
                    state.ssm[idx] = state.ssm[idx] * decay + delta * k[col];
                }
                core[vh * state.head_v_dim + row] = state.ssm[row_start..row_start + state.head_k_dim]
                    .iter()
                    .zip(q)
                    .map(|(s, query)| s * query)
                    .sum();
            }
        }

        // Qwen3.5 RMSNormGated uses its learned scale directly, then SiLU(z).
        for (head, output) in core.chunks_exact_mut(state.head_v_dim).enumerate() {
            let start = head * state.head_v_dim;
            let mean_square = output.iter().map(|x| x * x).sum::<f32>() / state.head_v_dim as f32;
            let scale = 1.0 / (mean_square + 1e-6).sqrt();
            for idx in 0..state.head_v_dim {
                let value_idx = start + idx;
                let gate = z_raw[value_idx];
                output[idx] = output[idx] * scale * norm_weight[idx] * gate * sigmoid(gate);
            }
        }
        cores.extend_from_slice(&core);
    }
    super::session::prof_acc(&mut prof.cpu_ms, t0);
    let t0 = super::session::prof_snap(prof, device);
    let gated = Tensor::new(cores, device)?
        .reshape((n_steps, value_dim))?
        .to_dtype(DType::F16)?;
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
