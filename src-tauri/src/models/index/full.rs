//! Qwen35 full-attention 单 token 前向（CPU 可跑；CUDA tensor 化后置）。
//!
//! GQA + RoPE 后 Q/K RMSNorm（QK-Norm）+ KV cache 拼接 + softmax + o_proj。
//! 不走 flash-attn（AGENTS.md 禁重编；且 Hy 的 flash 路径 CPU 无 fallback）。
//! mRoPE sections [11,11,10] 按 pair 交织重排 32 对频率；纯文本 token 的 T/H/W 位置 id
//! 相同，重排前后频率一致，等价退化为对前 64 维的标准 RoPE，后 192 维直通。

use super::model::FullLayerWeights;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};

/// 单层 KV cache（f32 CPU）。
pub(crate) struct FullCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

impl FullCache {
    pub(crate) fn new(n_kv_head: usize, head_dim: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            n_kv_head,
            head_dim,
        }
    }
}

fn rms_norm_row(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps as f32).sqrt();
    // GGUF 的 attn/q/k_norm 已由转换器预加 +1（实测 ~1.0-1.6），此处直接乘，不再 +1。
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}

fn rope_row(x: &[f32], pos: usize, freq_base: f32, rope_dim: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    let dim = rope_dim.min(x.len()) & !1;
    let half = dim / 2;
    for i in 0..half {
        let theta = 1.0 / freq_base.powf((2 * i) as f32 / dim as f32);
        let (cos, sin) = ((pos as f32 * theta).cos(), (pos as f32 * theta).sin());
        let (first, second) = (x[i], x[i + half]);
        out[i] = first * cos - second * sin;
        out[i + half] = second * cos + first * sin;
    }
    out
}

/// 单 token full 前向；Qwen3.5 的 Q 投影同时产出 query 与 output gate。
pub(crate) fn full_step(
    attn: &super::model::FullAttnWeights,
    hidden: &[f32],
    cache: &mut FullCache,
    pos: usize,
    rms_eps: f64,
    freq_base: f32,
    device: &Device,
) -> Result<Vec<f32>> {
    let h = Tensor::new(hidden, device)?
        .reshape((1, hidden.len()))?
        .to_dtype(DType::F32)?;
    let (n_head, n_kv_head, head_dim, rotary_dim) = (8usize, 2usize, 256usize, 64usize);
    let q_gate = attn.query.forward(&h)?.squeeze(0)?.to_vec1::<f32>()?;
    let k = attn.key.forward(&h)?.squeeze(0)?.to_vec1::<f32>()?;
    let v = attn.value.forward(&h)?.squeeze(0)?.to_vec1::<f32>()?;
    anyhow::ensure!(
        q_gate.len() == n_head * head_dim * 2,
        "unexpected Q+gate projection width"
    );
    anyhow::ensure!(
        k.len() == n_kv_head * head_dim && v.len() == n_kv_head * head_dim,
        "unexpected K/V projection width"
    );
    let qn_w = attn.query_norm_weight.to_vec1::<f32>()?;
    let kn_w = attn.key_norm_weight.to_vec1::<f32>()?;
    let mut q_heads = Vec::with_capacity(n_head);
    let mut gate = Vec::with_capacity(n_head * head_dim);
    for head in 0..n_head {
        let base = head * head_dim * 2;
        let q = rms_norm_row(&q_gate[base..base + head_dim], &qn_w, rms_eps);
        gate.extend_from_slice(&q_gate[base + head_dim..base + 2 * head_dim]);
        q_heads.push(rope_row(&q, pos, freq_base, rotary_dim));
    }
    let mut k_heads = Vec::with_capacity(n_kv_head);
    for head in 0..n_kv_head {
        let base = head * head_dim;
        let k = rms_norm_row(&k[base..base + head_dim], &kn_w, rms_eps);
        k_heads.push(rope_row(&k, pos, freq_base, rotary_dim));
    }
    let v_heads: Vec<&[f32]> = (0..n_kv_head)
        .map(|head| &v[head * head_dim..(head + 1) * head_dim])
        .collect();
    cache.k.push(k_heads.concat());
    cache.v.push(v_heads.concat());

    let rep = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_head * head_dim];
    for head in 0..n_head {
        let kv_head = head / rep;
        let q = &q_heads[head];
        let mut scores = Vec::with_capacity(cache.k.len());
        for key in &cache.k {
            let key_head = &key[kv_head * head_dim..(kv_head + 1) * head_dim];
            scores.push(q.iter().zip(key_head).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - max_score).exp();
                *score
            })
            .sum::<f32>();
        for dim in 0..head_dim {
            let mut value = 0.0;
            for (time, score) in scores.iter().enumerate() {
                value += score / denom * cache.v[time][kv_head * head_dim + dim];
            }
            out[head * head_dim + dim] = value * sigmoid(gate[head * head_dim + dim]);
        }
    }
    let output = Tensor::new(out, device)?
        .reshape((1, n_head * head_dim))?
        .to_dtype(DType::F32)?;
    Ok(attn.output.forward(&output)?.squeeze(0)?.to_vec1::<f32>()?)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_zero_pos_is_identity() {
        let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let out = rope_row(&x, 0, 10_000_000.0, 64);
        assert_eq!(out.len(), x.len());
        for (a, b) in out.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_keeps_tail_dims_passthrough() {
        let x: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let out = rope_row(&x, 7, 10_000_000.0, 64);
        assert_eq!(&out[64..], &x[64..]);
        assert!(out[..64].iter().zip(x[..64].iter()).any(|(a, b)| (a - b).abs() > 1e-3));
    }

    #[test]
    fn mrope_text_sections_partition_32_pairs() {
        // 官方 recomposition：H 取 slice(1, 33, 3)，W 取 slice(2, 30, 3)，余下归 T。
        let h: Vec<usize> = (1..33).step_by(3).collect();
        let w: Vec<usize> = (2..30).step_by(3).collect();
        assert_eq!((h.len(), w.len()), (11, 10));
        let mut t: Vec<usize> = (0..32).filter(|i| !h.contains(i) && !w.contains(i)).collect();
        t.sort_unstable();
        assert_eq!(t.len(), 11);
        let mut all = [h, w, t].concat();
        all.sort_unstable();
        assert_eq!(all, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn rms_norm_uses_gguf_weight_directly() {
        // GGUF norm 已预加 +1（约 1.0），直接乘；若误用 (1+w) 会得到约 2 倍。
        let x = vec![2.0f32; 4];
        let w = vec![1.0f32; 4];
        let out = rms_norm_row(&x, &w, 1e-6);
        let inv = 1.0 / (4.0f32 + 1e-6).sqrt();
        for v in &out {
            assert!((v - 2.0 * inv).abs() < 1e-5, "{v}");
        }
    }
}
