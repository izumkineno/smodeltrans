//! Qwen35 full-attention 单 token 前向（CPU 可跑；CUDA tensor 化后置）。
//!
//! GQA + RoPE 后 Q/K RMSNorm（QK-Norm）+ KV cache 拼接 + softmax + o_proj。
//! 不走 flash-attn（AGENTS.md 禁重编；且 Hy 的 flash 路径 CPU 无 fallback）。
//! mRoPE sections [11,11,10] 按 pair 交织重排 32 对频率；纯文本 token 的 T/H/W 位置 id
//! 相同，重排前后频率一致，等价退化为对前 64 维的标准 RoPE，后 192 维直通。

use super::model::{FullAttnWeights, FullLayerWeights};
#[cfg(any(test, feature = "flash-attn"))]
use super::model::INDEX_ROTARY_DIM;
use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;
#[cfg(feature = "flash-attn")]
use candle_nn::ops::{rms_norm, sigmoid as tensor_sigmoid};
#[cfg(any(test, feature = "flash-attn"))]
use candle_nn::rotary_emb::rope;

/// 单层 KV cache：CPU Vec（非 flash fallback）+ GPU tensor（flash 路径，按需翻倍扩容）。
pub(crate) struct FullCache {
    #[cfg_attr(feature = "flash-attn", allow(dead_code))]
    pub k: Vec<Vec<f32>>,
    #[cfg_attr(feature = "flash-attn", allow(dead_code))]
    pub v: Vec<Vec<f32>>,
    #[allow(dead_code)]
    pub tk: Option<Tensor>,
    #[allow(dead_code)]
    pub tv: Option<Tensor>,
}

impl FullCache {
    pub(crate) fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            tk: None,
            tv: None,
        }
    }
}

#[cfg_attr(feature = "flash-attn", allow(dead_code))]
fn rms_norm_row(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps as f32).sqrt();
    // GGUF 的 attn/q/k_norm 已由转换器预加 +1（实测 ~1.0-1.6），此处直接乘，不再 +1。
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}

#[cfg_attr(feature = "flash-attn", allow(dead_code))]
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

/// batch full 前向；hidden (S, 2048) F16 GPU in / out。
/// flash 路径全程 GPU tensor（rope 预计算表 + tensor KV + flash_attn），零 CPU 回传；
/// 非 flash 编译走 CPU 逐 step（pos = start_pos + s）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn full_step(
    layer: &FullLayerWeights,
    hidden: &Tensor,
    cache: &mut FullCache,
    start_pos: usize,
    rms_eps: f64,
    freq_base: f32,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    device: &Device,
    prof: &mut super::session::StepProfile,
) -> Result<Tensor> {
    let _ = freq_base;
    #[cfg(feature = "flash-attn")]
    {
        full_step_flash(
            layer, hidden, cache, start_pos, rms_eps, rope_cos, rope_sin, device, prof,
        )
    }
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = (rope_cos, rope_sin);
        full_step_cpu(
            &layer.attn,
            hidden,
            cache,
            start_pos,
            rms_eps,
            freq_base,
            device,
            prof,
        )
    }
}

#[cfg(feature = "flash-attn")]
#[allow(clippy::too_many_arguments)]
fn full_step_flash(
    layer: &FullLayerWeights,
    hidden: &Tensor,
    cache: &mut FullCache,
    start_pos: usize,
    rms_eps: f64,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    device: &Device,
    prof: &mut super::session::StepProfile,
) -> Result<Tensor> {
    let attn = &layer.attn;
    let (n_head, n_kv_head, head_dim) = (layer.n_head, layer.n_kv_head, layer.head_dim);
    let t0 = super::session::prof_snap(prof, device);
    let qg = attn.query.forward(hidden)?;
    let k = attn.key.forward(hidden)?;
    let v = attn.value.forward(hidden)?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    let s = hidden.dim(0)?;
    let dt = hidden.dtype();
    let t0 = super::session::prof_snap(prof, device);
    let qg_heads = qg.reshape((s, n_head, 2 * head_dim))?.contiguous()?;
    let q_part = qg_heads
        .narrow(2, 0, head_dim)?
        .contiguous()?
        .reshape((1, s, n_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let gate = qg_heads
        .narrow(2, head_dim, head_dim)?
        .contiguous()?
        .reshape((s, n_head * head_dim))?;
    let k = k
        .reshape((1, s, n_kv_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = v
        .reshape((1, s, n_kv_head, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let qn = attn.query_norm_weight.to_dtype(dt)?.contiguous()?;
    let kn = attn.key_norm_weight.to_dtype(dt)?.contiguous()?;
    let q = rms_norm(&q_part, &qn, rms_eps as f32)?.contiguous()?;
    let k = rms_norm(&k, &kn, rms_eps as f32)?.contiguous()?;
    let cos = rope_cos
        .narrow(0, start_pos, s)?
        .to_dtype(dt)?
        .contiguous()?;
    let sin = rope_sin
        .narrow(0, start_pos, s)?
        .to_dtype(dt)?
        .contiguous()?;
    let q_rot = rope(&q.narrow(3, 0, INDEX_ROTARY_DIM)?.contiguous()?, &cos, &sin)?;
    let q = Tensor::cat(
        &[
            &q_rot,
            &q.narrow(3, INDEX_ROTARY_DIM, head_dim - INDEX_ROTARY_DIM)?
                .contiguous()?,
        ],
        3,
    )?
    .contiguous()?;
    let k_rot = rope(&k.narrow(3, 0, INDEX_ROTARY_DIM)?.contiguous()?, &cos, &sin)?;
    let k = Tensor::cat(
        &[
            &k_rot,
            &k.narrow(3, INDEX_ROTARY_DIM, head_dim - INDEX_ROTARY_DIM)?
                .contiguous()?,
        ],
        3,
    )?
    .contiguous()?;
    let total_len = start_pos + s;
    let max_ctx = rope_cos.dim(0)?;
    let cache_cap = match &cache.tk {
        Some(t) => t.dim(2)?,
        None => 0,
    };
    if cache_cap < total_len {
        let new_cap = (if cache_cap == 0 {
            total_len.max(64).next_power_of_two()
        } else {
            cache_cap.saturating_mul(2).max(total_len)
        })
        .min(max_ctx);
        anyhow::ensure!(
            new_cap >= total_len,
            "index full KV cache exceeded rope table ({total_len} > {new_cap})"
        );
        let new_k = Tensor::zeros((1, n_kv_head, new_cap, head_dim), DType::F16, device)?;
        let new_v = Tensor::zeros((1, n_kv_head, new_cap, head_dim), DType::F16, device)?;
        if start_pos > 0 {
            if let (Some(old_k), Some(old_v)) = (cache.tk.as_ref(), cache.tv.as_ref()) {
                new_k.slice_set(&old_k.narrow(2, 0, start_pos)?.contiguous()?, 2, 0)?;
                new_v.slice_set(&old_v.narrow(2, 0, start_pos)?.contiguous()?, 2, 0)?;
            }
        }
        cache.tk = Some(new_k);
        cache.tv = Some(new_v);
    }
    {
        let ck = cache.tk.as_mut().expect("index KV cache k missing");
        ck.slice_set(&k.to_dtype(DType::F16)?.contiguous()?, 2, start_pos)?;
        let cv = cache.tv.as_mut().expect("index KV cache v missing");
        cv.slice_set(&v.to_dtype(DType::F16)?.contiguous()?, 2, start_pos)?;
    }
    let kf = cache
        .tk
        .as_ref()
        .expect("index KV cache k missing")
        .narrow(2, 0, total_len)?
        .transpose(1, 2)?;
    let vf = cache
        .tv
        .as_ref()
        .expect("index KV cache v missing")
        .narrow(2, 0, total_len)?
        .transpose(1, 2)?;
    let qf = q.transpose(1, 2)?.contiguous()?;
    let attn_out = flash_attn(&qf, &kf, &vf, 1.0 / (head_dim as f32).sqrt(), true)?;
    let attn_out = attn_out.reshape((s, n_head * head_dim))?;
    let gated = attn_out.broadcast_mul(&tensor_sigmoid(&gate)?)?;
    let out = attn.output.forward(&gated)?;
    super::session::prof_acc(&mut prof.attn_ms, t0);
    Ok(out)
}

#[cfg_attr(feature = "flash-attn", allow(dead_code))]
fn full_step_cpu(
    attn: &FullAttnWeights,
    hidden: &Tensor,
    cache: &mut FullCache,
    start_pos: usize,
    rms_eps: f64,
    freq_base: f32,
    device: &Device,
    prof: &mut super::session::StepProfile,
) -> Result<Tensor> {
    let qn = attn.query_norm_weight.to_vec1::<f32>()?;
    let kn = attn.key_norm_weight.to_vec1::<f32>()?;
    let t0 = super::session::prof_snap(prof, device);
    let q_gate = attn.query.forward(hidden)?;
    let k = attn.key.forward(hidden)?;
    let v = attn.value.forward(hidden)?;
    let (n_head, n_kv_head, head_dim, rotary_dim) = (8usize, 2usize, 256usize, 64usize);
    let rows = Tensor::cat(&[q_gate, k, v], 1)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    let n_steps = rows.len();
    let rep = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut outs = Vec::with_capacity(n_steps * n_head * head_dim);
    let t0 = super::session::prof_snap(prof, device);
    for (step, row) in rows.iter().enumerate() {
        let pos = start_pos + step;
        let (q_gate, rest) = row.split_at(n_head * head_dim * 2);
        let (k, v) = rest.split_at(n_kv_head * head_dim);
        anyhow::ensure!(
            q_gate.len() == n_head * head_dim * 2,
            "unexpected Q+gate projection width"
        );
        anyhow::ensure!(
            k.len() == n_kv_head * head_dim && v.len() == n_kv_head * head_dim,
            "unexpected K/V projection width"
        );
        let mut q_heads = Vec::with_capacity(n_head);
        let mut gate = Vec::with_capacity(n_head * head_dim);
        for head in 0..n_head {
            let base = head * head_dim * 2;
            let q = rms_norm_row(&q_gate[base..base + head_dim], &qn, rms_eps);
            gate.extend_from_slice(&q_gate[base + head_dim..base + 2 * head_dim]);
            q_heads.push(rope_row(&q, pos, freq_base, rotary_dim));
        }
        let mut k_heads = Vec::with_capacity(n_kv_head);
        for head in 0..n_kv_head {
            let base = head * head_dim;
            let k = rms_norm_row(&k[base..base + head_dim], &kn, rms_eps);
            k_heads.push(rope_row(&k, pos, freq_base, rotary_dim));
        }
        let v_heads: Vec<&[f32]> = (0..n_kv_head)
            .map(|head| &v[head * head_dim..(head + 1) * head_dim])
            .collect();
        cache.k.push(k_heads.concat());
        cache.v.push(v_heads.concat());

        let mut out = vec![0.0f32; n_head * head_dim];
        let mut scores = Vec::new();
        for head in 0..n_head {
            let kv_head = head / rep;
            let q = &q_heads[head];
            scores.clear();
            scores.reserve(cache.k.len());
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
        outs.extend_from_slice(&out);
    }
    super::session::prof_acc(&mut prof.cpu_ms, t0);
    let t0 = super::session::prof_snap(prof, device);
    let out_t = Tensor::new(outs, device)?
        .reshape((n_steps, n_head * head_dim))?
        .to_dtype(DType::F16)?;
    let out = attn.output.forward(&out_t)?;
    super::session::prof_acc(&mut prof.proj_ms, t0);
    Ok(out)
}

#[cfg_attr(feature = "flash-attn", allow(dead_code))]
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

    #[test]
    fn tensor_partial_rope_matches_rope_row() {
        use super::super::model::precompute_freqs_cis;
        let device = Device::Cpu;
        let vals: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let xs = Tensor::new(vals.clone(), &device)
            .unwrap()
            .reshape((1, 1, 1, 256))
            .unwrap();
        let (cos_all, sin_all) =
            precompute_freqs_cis(INDEX_ROTARY_DIM, 10_000.0, 16, &device).unwrap();
        let pos = 7usize;
        let cos = cos_all.narrow(0, pos, 1).unwrap();
        let sin = sin_all.narrow(0, pos, 1).unwrap();
        let rot = rope(&xs.narrow(3, 0, INDEX_ROTARY_DIM).unwrap().contiguous().unwrap(), &cos, &sin).unwrap();
        let out = Tensor::cat(&[&rot, &xs.narrow(3, INDEX_ROTARY_DIM, 256 - INDEX_ROTARY_DIM).unwrap()], 3).unwrap();
        let got = out.reshape(256).unwrap().to_vec1::<f32>().unwrap();
        let want = rope_row(&vals, pos, 10_000.0, INDEX_ROTARY_DIM);
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
