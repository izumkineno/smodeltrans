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

/// 单层 recurrent state：conv ring（(k-1) × qkv_dim）+ SSM 矩阵（num_v × head_vd × head_kd）。
pub(crate) struct DeltaNetState {
    pub conv_hist: Vec<Vec<f32>>,
    pub ssm: Vec<f32>,
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
            num_v_heads,
            head_v_dim,
            head_k_dim,
            qkv_dim,
        }
    }
}

fn l2norm_row(v: &[f32]) -> Vec<f32> {
    let inv_norm = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
    v.iter().map(|x| x * inv_norm).collect()
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

/// 单 token DeltaNet recurrent 前向；线性投影由 Candle 在所选设备执行，递推 state 为 f32。
/// 输入 hidden `[2048]`；输出 `[2048]`。state 原地更新。
pub(crate) fn delta_step(
    layer: &LinearLayerWeights,
    hidden: &[f32],
    state: &mut DeltaNetState,
    device: &Device,
) -> Result<Vec<f32>> {
    let hidden = Tensor::new(hidden, device)?
        .reshape((1, hidden.len()))?
        .to_dtype(DType::F32)?;
    let qkv = layer
        .attn_qkv
        .forward(&hidden)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    let z = layer
        .attn_gate
        .forward(&hidden)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    let key_dim = state.num_v_heads * state.head_k_dim;
    let value_dim = state.num_v_heads * state.head_v_dim;
    anyhow::ensure!(
        qkv.len() == 2 * key_dim + value_dim,
        "unexpected QKV projection width {}",
        qkv.len()
    );
    anyhow::ensure!(
        z.len() == value_dim,
        "unexpected z projection width {}",
        z.len()
    );
    anyhow::ensure!(
        state.qkv_dim == qkv.len(),
        "conv state/projection width mismatch"
    );

    // Causal depthwise Conv1d: each channel consumes its oldest-to-newest 4-tap window.
    let weights = layer.ssm_conv1d.to_vec2::<f32>()?;
    let transposed = weights.len() != 4;
    anyhow::ensure!(
        if transposed {
            weights.len() == qkv.len() && weights.iter().all(|row| row.len() == 4)
        } else {
            weights.iter().all(|row| row.len() == qkv.len())
        },
        "unexpected conv1d weight shape"
    );
    let mut convolved = vec![0.0f32; qkv.len()];
    for channel in 0..qkv.len() {
        let taps = [
            state.conv_hist[0][channel],
            state.conv_hist[1][channel],
            state.conv_hist[2][channel],
            qkv[channel],
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
        convolved[channel] = acc * sigmoid(acc);
    }
    let (hist_0, rest) = state.conv_hist.split_at_mut(1);
    let (hist_1, hist_2) = rest.split_at_mut(1);
    hist_0[0].clone_from(&hist_1[0]);
    hist_1[0].clone_from(&hist_2[0]);
    state.conv_hist[2].copy_from_slice(&qkv);

    let (q_raw, rest) = convolved.split_at(key_dim);
    let (k_raw, v_raw) = rest.split_at(key_dim);
    let ba = layer
        .ssm_alpha
        .forward(&hidden)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    let beta_logits = layer
        .ssm_beta
        .forward(&hidden)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    // GGUF `ssm_a` 已是 -exp(A_log)（llama.cpp 转换器对 HF `A_log` 做 `-exp` 后写入，
    // 实测 blk.0.ssm_a 首值 -0.78/-0.057 均为负数），此处直接使用，不再 exp。
    let ssm_a = layer.ssm_a.to_vec1::<f32>()?;
    let dt_bias = layer.ssm_dt_bias.to_vec1::<f32>()?;
    anyhow::ensure!(
        ba.len() == state.num_v_heads && beta_logits.len() == state.num_v_heads,
        "unexpected DeltaNet gate width"
    );
    let norm_weight = layer.ssm_norm_weight.to_vec1::<f32>()?;
    let mut core = vec![0.0f32; value_dim];

    for vh in 0..state.num_v_heads {
        let mut q = l2norm_row(&q_raw[vh * state.head_k_dim..(vh + 1) * state.head_k_dim]);
        let k = l2norm_row(&k_raw[vh * state.head_k_dim..(vh + 1) * state.head_k_dim]);
        let inv_sqrt_k = 1.0 / (state.head_k_dim as f32).sqrt();
        for value in &mut q {
            *value *= inv_sqrt_k;
        }
        let v = &v_raw[vh * state.head_v_dim..(vh + 1) * state.head_v_dim];
        let beta = sigmoid(beta_logits[vh]);
        let decay = decay_factor(ssm_a[vh], ba[vh] + dt_bias[vh]);
        let base = vh * state.head_v_dim * state.head_k_dim;

        // State is stored [value_dim, key_dim], equivalent to the reference [key_dim, value_dim].
        for row in 0..state.head_v_dim {
            let row_start = base + row * state.head_k_dim;
            let prediction = state.ssm[row_start..row_start + state.head_k_dim]
                .iter()
                .zip(&k)
                .map(|(s, key)| s * key)
                .sum::<f32>();
            let delta = beta * (v[row] - prediction);
            for col in 0..state.head_k_dim {
                let idx = row_start + col;
                state.ssm[idx] = state.ssm[idx] * decay + delta * k[col];
            }
            core[vh * state.head_v_dim + row] = state.ssm[row_start..row_start + state.head_k_dim]
                .iter()
                .zip(&q)
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
            let gate = z[value_idx];
            output[idx] = output[idx] * scale * norm_weight[idx] * gate * sigmoid(gate);
        }
    }
    let gated = Tensor::new(core, device)?
        .reshape((1, value_dim))?
        .to_dtype(DType::F32)?;
    Ok(layer
        .ssm_out
        .forward(&gated)?
        .squeeze(0)?
        .to_vec1::<f32>()?)
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
