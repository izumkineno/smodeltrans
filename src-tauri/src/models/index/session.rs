//! Index 会话：单轮无记忆（translation 场景够用），贪心解码。
use super::{
    deltanet::{DeltaNetState, delta_step},
    full::{FullCache, full_step},
    model::{IndexLayer, ModelWeights},
};
use crate::{model_config::GenerationConfig, model_support::CancellationToken};
use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;
pub(crate) struct IndexSession {
    tokenizer: Tokenizer,
    model: ModelWeights,
    device: Device,
    eos_ids: [u32; 2],
}
/// 每 token 步进的双 states（linear recurrent + full KV）。
/// hidden 全程驻 GPU，只在 DeltaNet 递推 / Full GQA / 最终 argmax 处过 CPU。
struct IndexStates {
    delta: Vec<Option<DeltaNetState>>,
    full: Vec<FullCache>,
    attn_norms: Vec<Tensor>,
    post_norms: Vec<Tensor>,
    output_norm: Tensor,
}
#[allow(dead_code)]
fn rms_norm_vec(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps as f32).sqrt();
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}

/// env 门控的前向分段计时（`SMODELTRANS_INDEX_PROFILE=1` 开启；关闭时零开销）。
/// GPU 是异步的，分段必须先 `synchronize` 否则数字无意义。
#[derive(Default)]
pub(crate) struct StepProfile {
    pub enabled: bool,
    pub proj_ms: f64,
    pub cpu_ms: f64,
    pub ffn_ms: f64,
    pub norm_ms: f64,
    pub steps: u64,
}

fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SMODELTRANS_INDEX_PROFILE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

pub(crate) fn prof_snap(prof: &StepProfile, device: &Device) -> Option<std::time::Instant> {
    if prof.enabled {
        let _ = device.synchronize();
        Some(std::time::Instant::now())
    } else {
        None
    }
}

pub(crate) fn prof_acc(slot: &mut f64, t0: Option<std::time::Instant>) {
    if let Some(t) = t0 {
        *slot += t.elapsed().as_secs_f64() * 1000.0;
    }
}
impl IndexSession {
    pub(crate) fn new(model_path: &Path, device: &Device) -> Result<Self> {
        let (weights, tokenizer) = ModelWeights::open(model_path, device, 32768)?;
        // Local tokenizer_config.json / generation_config.json: endoftext and im_end are EOS.
        Ok(Self {
            tokenizer,
            model: weights,
            device: device.clone(),
            eos_ids: [248044, 248046],
        })
    }
    fn new_states(&self) -> Result<IndexStates> {
        let f16 = candle_core::DType::F16;
        let mut delta = Vec::with_capacity(self.model.layers.len());
        let mut full = Vec::with_capacity(self.model.layers.len());
        let mut attn_norms = Vec::with_capacity(self.model.layers.len());
        let mut post_norms = Vec::with_capacity(self.model.layers.len());
        for layer in &self.model.layers {
            let common = match layer {
                IndexLayer::Linear(w) => &w.common,
                IndexLayer::Full(w) => &w.common,
            };
            attn_norms.push(common.attn_norm_weight.to_dtype(f16)?);
            post_norms.push(common.post_norm_weight.to_dtype(f16)?);
            match layer {
                IndexLayer::Linear(w) => {
                    let mut st = DeltaNetState::zeros(16, 128, 128, 6144);
                    st.ssm_a = w.ssm_a.to_vec1::<f32>()?;
                    st.dt_bias = w.ssm_dt_bias.to_vec1::<f32>()?;
                    st.norm_weight = w.ssm_norm_weight.to_vec1::<f32>()?;
                    st.conv_weight = w.ssm_conv1d.to_vec2::<f32>()?;
                    st.conv_transposed = st.conv_weight.len() != 4;
                    delta.push(Some(st));
                    full.push(FullCache::new(0, 0, Vec::new(), Vec::new()));
                }
                IndexLayer::Full(w) => {
                    delta.push(None);
                    full.push(FullCache::new(
                        2,
                        256,
                        w.attn.query_norm_weight.to_vec1::<f32>()?,
                        w.attn.key_norm_weight.to_vec1::<f32>()?,
                    ));
                }
            }
        }
        let output_norm = self.model.output_norm_weight.to_dtype(f16)?;
        Ok(IndexStates {
            delta,
            full,
            attn_norms,
            post_norms,
            output_norm,
        })
    }

    /// 单轮翻译：编码 → prefill（逐 token 步进，简化版）→ 贪心解码到 EOS/上限。
    pub(crate) fn translate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        cancellation: &CancellationToken,
        mut on_chunk: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        cancellation
            .check()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        // Match the local tokenizer_config.json chat_template used by the OpenAI server.
        let chat_prompt = format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let ids = self
            .tokenizer
            .encode(chat_prompt.as_str(), true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(!ids.is_empty(), "prompt produced no tokenizer ids");
        let mut states = self.new_states()?;
        let mut prof = StepProfile {
            enabled: profile_enabled(),
            ..Default::default()
        };
        // prefill：整串一次 batch 前向（投影走 mmq 快路径；DeltaNet/GQA 在 CPU 逐 step 推进状态）。
        let emb = self.embed_ids(&ids)?;
        let seq_len = ids.len();
        let batch = emb.squeeze(0)?;
        let batch_logits = self.forward_batch(&batch, &mut states, 0, &mut prof)?;
        let mut logits = batch_logits.narrow(0, seq_len - 1, 1)?;
        let mut out_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let max_new = config.max_new_tokens.min(1024);
        for _ in 0..max_new {
            cancellation
                .check()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let next = argmax_u32(&logits)?;
            if self.eos_ids.contains(&next) {
                break;
            }
            out_ids.push(next);
            let piece = self
                .tokenizer
                .decode(&[next], false)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            text.push_str(&piece);
            on_chunk(&piece)?;
            let hid = self.embed_ids(&[next])?.squeeze(0)?;
            logits = self.forward_batch(
                &hid,
                &mut states,
                ids.len() + out_ids.len() - 1,
                &mut prof,
            )?;
        }
        if prof.enabled {
            println!(
                "index_profile steps={} proj={:.0}ms cpu={:.0}ms ffn={:.0}ms norm={:.0}ms",
                prof.steps, prof.proj_ms, prof.cpu_ms, prof.ffn_ms, prof.norm_ms,
            );
        }
        Ok(text)
    }

    fn embed_ids(&self, ids: &[u32]) -> Result<Tensor> {
        let t = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let emb = self
            .model
            .token_embd
            .embedding(&t)?
            .to_dtype(candle_core::DType::F16)?;
        Ok(emb)
    }

    /// batch 前向（S=seq 为 prefill，S=1 为解码步）；hidden/logits 均为 F16 GPU tensor。
    fn forward_batch(
        &self,
        hidden: &Tensor,
        states: &mut IndexStates,
        start_pos: usize,
        prof: &mut StepProfile,
    ) -> Result<Tensor> {
        prof.steps += hidden.dim(0)? as u64;
        let mut h = hidden.clone();
        for (index, ((layer, dst), fc)) in self
            .model
            .layers
            .iter()
            .zip(states.delta.iter_mut())
            .zip(states.full.iter_mut())
            .enumerate()
        {
            match (layer, dst) {
                (IndexLayer::Linear(w), Some(st)) => {
                    let t0 = prof_snap(prof, &self.device);
                    let n = rms_norm_gpu(&h, &states.attn_norms[index], self.model.rms_norm_eps)?;
                    prof_acc(&mut prof.norm_ms, t0);
                    let o = delta_step(w, &n, st, &self.device, prof)?;
                    h = (&h + &o)?;
                    let t0 = prof_snap(prof, &self.device);
                    let n2 =
                        rms_norm_gpu(&h, &states.post_norms[index], self.model.rms_norm_eps)?;
                    prof_acc(&mut prof.norm_ms, t0);
                    let f = ffn_swiglu_gpu(&w.common, &n2, &self.device, prof)?;
                    h = (&h + &f)?;
                }
                (IndexLayer::Full(w), _) => {
                    let t0 = prof_snap(prof, &self.device);
                    let n = rms_norm_gpu(&h, &states.attn_norms[index], self.model.rms_norm_eps)?;
                    prof_acc(&mut prof.norm_ms, t0);
                    let o = full_step(
                        &w.attn,
                        &n,
                        fc,
                        start_pos,
                        self.model.rms_norm_eps,
                        self.model.freq_base,
                        &self.device,
                        prof,
                    )?;
                    h = (&h + &o)?;
                    let t0 = prof_snap(prof, &self.device);
                    let n2 =
                        rms_norm_gpu(&h, &states.post_norms[index], self.model.rms_norm_eps)?;
                    prof_acc(&mut prof.norm_ms, t0);
                    let f = ffn_swiglu_gpu(&w.common, &n2, &self.device, prof)?;
                    h = (&h + &f)?;
                }
                _ => anyhow::bail!("layer/state mismatch"),
            }
        }
        let t0 = prof_snap(prof, &self.device);
        let normed = rms_norm_gpu(&h, &states.output_norm, self.model.rms_norm_eps)?;
        prof_acc(&mut prof.norm_ms, t0);
        Ok(self.model.output_proj.forward(&normed)?)
    }
}

fn rms_norm_gpu(hidden: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // 融合单 op（launch+alloc 最少）；GGUF 权重已预 +1，直接乘即等价。
    Ok(candle_nn::ops::rms_norm(hidden, weight, eps as f32)?)
}

fn argmax_u32(logits: &Tensor) -> Result<u32> {
    Ok(logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?)
}

fn ffn_swiglu_gpu(
    common: &super::model::BlockCommon,
    hidden: &Tensor,
    device: &Device,
    prof: &mut StepProfile,
) -> Result<Tensor> {
    use candle_core::Module;
    let t0 = prof_snap(prof, device);
    let gate = common.ffn_gate.forward(hidden)?;
    let up = common.ffn_up.forward(hidden)?;
    let gated = candle_nn::ops::silu(&gate)?.broadcast_mul(&up)?;
    let out = common.ffn_down.forward(&gated)?;
    prof_acc(&mut prof.ffn_ms, t0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_vec_uses_gguf_weight_directly() {
        let x = vec![2.0f32; 8];
        let w = vec![1.0f32; 8];
        let out = rms_norm_vec(&x, &w, 1e-6);
        let inv = 1.0 / (4.0f32 + 1e-6).sqrt();
        for v in &out {
            assert!((v - 2.0 * inv).abs() < 1e-5, "{v}");
        }
    }
}
