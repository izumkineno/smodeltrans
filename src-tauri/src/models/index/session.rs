//! Index 会话：单轮无记忆（translation 场景够用），贪心解码。
use super::{
    deltanet::{DeltaNetState, delta_step},
    full::{FullCache, full_step},
    model::{BlockCommon, IndexLayer, ModelWeights},
};
use crate::{model_config::GenerationConfig, model_support::CancellationToken};
use anyhow::{Context, Result};
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
struct IndexStates {
    delta: Vec<Option<DeltaNetState>>,
    full: Vec<FullCache>,
}
fn rms_norm_vec(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps as f32).sqrt();
    // 与 full.rs 同理：GGUF attn/post/output_norm 已预加 +1，直接乘。
    x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
}
fn add_vec(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    anyhow::ensure!(a.len() == b.len(), "residual dim mismatch");
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
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
    fn new_states(&self) -> IndexStates {
        let mut delta = Vec::with_capacity(self.model.layers.len());
        let mut full = Vec::with_capacity(self.model.layers.len());
        for layer in &self.model.layers {
            match layer {
                IndexLayer::Linear(_) => {
                    delta.push(Some(DeltaNetState::zeros(16, 128, 128, 6144)));
                    full.push(FullCache::new(0, 0));
                }
                IndexLayer::Full(_) => {
                    delta.push(None);
                    full.push(FullCache::new(2, 256));
                }
            }
        }
        IndexStates { delta, full }
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
        let mut states = self.new_states();
        // prefill：逐 token 步进（朴素版；chunk-64 优化后置 P2b）
        let mut hidden = self.embed_ids(&ids)?;
        let mut logits = None;
        for (pos, hid) in hidden.iter().enumerate() {
            cancellation
                .check()
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            logits = Some(self.forward_token(hid, &mut states, pos)?);
        }
        let mut out_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let max_new = config.max_new_tokens.min(1024);
        let mut logits = logits.context("prefill produced no logits")?;
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
            let hid = self.embed_ids(&[next])?.remove(0);
            logits = self.forward_token(&hid, &mut states, ids.len() + out_ids.len() - 1)?;
        }
        Ok(text)
    }

    fn embed_ids(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let t = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let emb = self
            .model
            .token_embd
            .embedding(&t)?
            .to_dtype(candle_core::DType::F32)?;
        let v = emb.to_vec3::<f32>()?;
        Ok(v.into_iter().next().unwrap_or_default())
    }

    fn forward_token(
        &self,
        hidden: &[f32],
        states: &mut IndexStates,
        pos: usize,
    ) -> Result<Tensor> {
        let mut h = hidden.to_vec();
        for ((layer, dst), fc) in self
            .model
            .layers
            .iter()
            .zip(states.delta.iter_mut())
            .zip(states.full.iter_mut())
        {
            match (layer, dst) {
                (IndexLayer::Linear(w), Some(st)) => {
                    let n = rms_norm_vec(
                        &h,
                        &w.common.attn_norm_weight.to_vec1::<f32>()?,
                        self.model.rms_norm_eps,
                    );
                    let o = delta_step(w, &n, st, &self.device)?;
                    h = add_vec(&h, &o)?;
                    let n2 = rms_norm_vec(
                        &h,
                        &w.common.post_norm_weight.to_vec1::<f32>()?,
                        self.model.rms_norm_eps,
                    );
                    let f = ffn_swiglu(&w.common, &n2, &self.device)?;
                    h = add_vec(&h, &f)?;
                }
                (IndexLayer::Full(w), _) => {
                    let n = rms_norm_vec(
                        &h,
                        &w.common.attn_norm_weight.to_vec1::<f32>()?,
                        self.model.rms_norm_eps,
                    );
                    let o = full_step(
                        &w.attn,
                        &n,
                        fc,
                        pos,
                        self.model.rms_norm_eps,
                        self.model.freq_base,
                        &self.device,
                    )?;
                    h = add_vec(&h, &o)?;
                    let n2 = rms_norm_vec(
                        &h,
                        &w.common.post_norm_weight.to_vec1::<f32>()?,
                        self.model.rms_norm_eps,
                    );
                    let f = ffn_swiglu(&w.common, &n2, &self.device)?;
                    h = add_vec(&h, &f)?;
                }
                _ => anyhow::bail!("layer/state mismatch"),
            }
        }
        let normed = rms_norm_vec(
            &h,
            &self.model.output_norm_weight.to_vec1::<f32>()?,
            self.model.rms_norm_eps,
        );
        let t = Tensor::new(normed, &self.device)?
            .reshape((1, h.len()))?
            .to_dtype(candle_core::DType::F32)?;
        Ok(self.model.output_proj.forward(&t)?)
    }
}

fn argmax_u32(logits: &Tensor) -> Result<u32> {
    Ok(logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?)
}

fn ffn_swiglu(
    common: &super::model::BlockCommon,
    hidden: &[f32],
    device: &Device,
) -> Result<Vec<f32>> {
    use candle_core::Module;
    let h = Tensor::new(hidden, device)?
        .reshape((1, hidden.len()))?
        .to_dtype(candle_core::DType::F32)?;
    let gate = common.ffn_gate.forward(&h)?;
    let up = common.ffn_up.forward(&h)?;
    let gated = candle_nn::ops::silu(&gate)?.broadcast_mul(&up)?;
    let down = common.ffn_down.forward(&gated)?;
    Ok(down.squeeze(0)?.to_vec1::<f32>()?)
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
