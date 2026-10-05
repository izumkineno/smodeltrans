//! Index-Translate 的 Qwen35 GGUF 加载与自回归推理实现。
//!
//! P0 取证结论（`models/Index-Translate-2B.f16.gguf`，335 tensor / 37 meta）：
//! - `general.architecture = "qwen35"`，`context_length = 262144`，8 q-head / 2 kv-head
//! - `ssm.conv_kernel = 4`，`ssm.state_size = 128`，`ssm.group_count = 16`，`inner_size = 2048`
//! - linear 层 18 个（`blk.N.ssm_alpha/beta` Separate 投影 `[2048,16]`，`attn_qkv [2048,6144]`，
//!   `ssm_conv1d [4,6144]`，`ssm_a/dt [16]`，`ssm_norm [128]`，`ssm_out [2048,2048]`）
//! - full 层 7 个（`attn_q/k/v/output` + `q/k_norm`，`head_dim = 256`）
//! - `rope.dimension_sections = [11,11,10,0]`（mRoPE），`freq_base = 10_000_000`
//! - tokenizer `gpt2/qwen35`，vocab 248320，merges 247587，eos = pad = 248044
//! - 第 25 层为 MTP（`nextn.*`），文本翻译不加载

use anyhow::{Context, Result};
use candle_core::{
    DType, Device, Module, Tensor,
    quantized::{
        GgmlDType, QMatMul,
        gguf_file::{Content, Value},
    },
};
use std::{fs::File, io::BufReader, path::Path};
use tokenizers::Tokenizer;

/// GGUF arch 标识（P0 取证值，非 `qwen3_5`）。
pub(crate) const INDEX_ARCH: &str = "qwen35";
/// 线性层数（P0：18 个 `ssm_*` 桶）。
pub(crate) const INDEX_LINEAR_LAYERS: usize = 18;
/// 全注意力层数（Qwen3.5 2B 的 24 层中每 4 层 1 个，共 6 个）。
pub(crate) const INDEX_FULL_LAYERS: usize = 6;

/// Qwen3.5 full-attention 的 QKVO + QK-Norm 与 Q 输出门控（GQA）。
pub(crate) struct FullAttnWeights {
    pub query: QMatMul,
    pub key: QMatMul,
    pub value: QMatMul,
    pub output: QMatMul,
    pub query_norm_weight: Tensor,
    pub key_norm_weight: Tensor,
}
/// 每 blk 共有的 norm + SwiGLU FFN（P0：linear/full 层均有 `ffn_gate/up/down` + 双 norm）。
pub(crate) struct BlockCommon {
    pub attn_norm_weight: Tensor,
    pub post_norm_weight: Tensor,
    pub ffn_gate: QMatMul,
    pub ffn_up: QMatMul,
    pub ffn_down: QMatMul,
}
/// Qwen35 linear-attention（Gated DeltaNet）单层权重（见模块头注 R2 对应关系）。
pub(crate) struct LinearLayerWeights {
    pub attn_qkv: QMatMul,
    pub attn_gate: QMatMul,
    pub ssm_alpha: QMatMul,
    pub ssm_beta: QMatMul,
    pub ssm_conv1d: Tensor,
    pub ssm_a: Tensor,
    pub ssm_dt_bias: Tensor,
    pub ssm_norm_weight: Tensor,
    pub ssm_out: QMatMul,
    pub common: BlockCommon,
}
/// Qwen35 full-attention block = GQA + 共有 FFN。
pub(crate) struct FullLayerWeights {
    pub attn: FullAttnWeights,
    pub common: BlockCommon,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

/// Full 层参与 RoPE 的前 64 维（mRoPE 文本退化：后 192 维直通）。
pub(crate) const INDEX_ROTARY_DIM: usize = 64;

/// Qwen35 traffo block：linear（18/24）或 full（6/24 + MTP 除外）。
pub(crate) enum IndexLayer {
    Linear(LinearLayerWeights),
    Full(FullLayerWeights),
}

/// Index 模型权重全集。
pub(crate) struct ModelWeights {
    pub token_embd: QMatMul,
    pub output_norm_weight: Tensor,
    pub output_proj: QMatMul,
    pub layers: Vec<IndexLayer>,
    pub vocab_size: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub freq_base: f32,
    pub max_seq_len: usize,
    pub rope_cos: Tensor,
    pub rope_sin: Tensor,
}

pub(crate) fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn metadata<'a>(content: &'a Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .with_context(|| format!("GGUF is missing metadata '{key}'"))
}
pub(crate) fn is_index_gguf(path: &Path) -> Result<bool> {
    let mut reader = BufReader::new(File::open(path)?);
    let content = Content::read(&mut reader)?;
    Ok(metadata(&content, "general.architecture")?.to_string()? == INDEX_ARCH)
}
fn load_qmat<R: std::io::Read + std::io::Seek>(
    content: &Content,
    reader: &mut R,
    device: &Device,
    key: &str,
) -> Result<QMatMul> {
    let qtensor = content.tensor(reader, key, device)?;
    match qtensor.dtype() {
        // Float weights keep the previous behavior: dequantize + TensorF16,
        // whose forward casts inputs (plain Tensor has no such cast).
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
            let tensor = qtensor
                .dequantize(device)
                .and_then(|tensor| tensor.to_dtype(DType::F16))
                .with_context(|| format!("failed to load F16 projection '{key}'"))?;
            Ok(QMatMul::TensorF16(tensor))
        }
        // Quantized weights stay quantized: VRAM ~= file size, quantized CUDA matmul.
        _ => Ok(QMatMul::from_qtensor(qtensor)?),
    }
}
fn load_f32<R: std::io::Read + std::io::Seek>(
    content: &Content,
    reader: &mut R,
    device: &Device,
    key: &str,
) -> Result<Tensor> {
    Ok(content
        .tensor(reader, key, device)?
        .dequantize(device)?
        .to_dtype(DType::F32)?)
}

impl ModelWeights {
    /// 是否为 linear-attention 层：存在 `ssm_alpha.weight` 即判定（P0：18 层命中）。
    fn is_linear_layer(content: &Content, layer: usize) -> bool {
        content
            .tensor_infos
            .contains_key(&format!("blk.{layer}.ssm_alpha.weight"))
    }
    fn load_common<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        device: &Device,
        layer: usize,
    ) -> Result<BlockCommon> {
        let p = |name: &str| format!("blk.{layer}.{name}");
        Ok(BlockCommon {
            attn_norm_weight: load_f32(content, reader, device, &p("attn_norm.weight"))?,
            post_norm_weight: load_f32(content, reader, device, &p("post_attention_norm.weight"))?,
            ffn_gate: load_qmat(content, reader, device, &p("ffn_gate.weight"))?,
            ffn_up: load_qmat(content, reader, device, &p("ffn_up.weight"))?,
            ffn_down: load_qmat(content, reader, device, &p("ffn_down.weight"))?,
        })
    }
    fn load_linear<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        device: &Device,
        layer: usize,
    ) -> Result<LinearLayerWeights> {
        let p = |name: &str| format!("blk.{layer}.{name}");
        Ok(LinearLayerWeights {
            attn_qkv: load_qmat(content, reader, device, &p("attn_qkv.weight"))?,
            attn_gate: load_qmat(content, reader, device, &p("attn_gate.weight"))?,
            ssm_alpha: load_qmat(content, reader, device, &p("ssm_alpha.weight"))?,
            ssm_beta: load_qmat(content, reader, device, &p("ssm_beta.weight"))?,
            ssm_conv1d: load_f32(content, reader, device, &p("ssm_conv1d.weight"))?,
            ssm_a: load_f32(content, reader, device, &p("ssm_a"))?,
            ssm_dt_bias: load_f32(content, reader, device, &p("ssm_dt.bias"))?,
            ssm_norm_weight: load_f32(content, reader, device, &p("ssm_norm.weight"))?,
            ssm_out: load_qmat(content, reader, device, &p("ssm_out.weight"))?,
            common: Self::load_common(content, reader, device, layer)?,
        })
    }
    fn load_full<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        device: &Device,
        layer: usize,
    ) -> Result<FullLayerWeights> {
        let p = |name: &str| format!("blk.{layer}.{name}");
        Ok(FullLayerWeights {
            attn: FullAttnWeights {
                query: load_qmat(content, reader, device, &p("attn_q.weight"))?,
                key: load_qmat(content, reader, device, &p("attn_k.weight"))?,
                value: load_qmat(content, reader, device, &p("attn_v.weight"))?,
                output: load_qmat(content, reader, device, &p("attn_output.weight"))?,
                query_norm_weight: load_f32(content, reader, device, &p("attn_q_norm.weight"))?,
                key_norm_weight: load_f32(content, reader, device, &p("attn_k_norm.weight"))?,
            },
            common: Self::load_common(content, reader, device, layer)?,
            n_head: 8,
            n_kv_head: 2,
            head_dim: 256,
        })
    }

    pub(crate) fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        device: &Device,
        max_seq_len: usize,
    ) -> Result<Self> {
        let architecture = metadata(content, "general.architecture")?.to_string()?;
        anyhow::ensure!(
            architecture == INDEX_ARCH,
            "unsupported GGUF architecture '{architecture}' (expected '{INDEX_ARCH}')"
        );
        let head_count = metadata(content, "qwen35.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = metadata(content, "qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let context_length = metadata(content, "qwen35.context_length")?.to_u32()? as usize;
        let rms_norm_eps =
            metadata(content, "qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let freq_base = metadata(content, "qwen35.rope.freq_base")?.to_f32()?;
        anyhow::ensure!(
            head_count == 8 && head_count_kv == 2,
            "unexpected GQA shape"
        );
        // head_dim = 256（full 层 attn_q [2048, 8*256] 隐含；linear 层 ssm_norm [128] 为 value head dim）
        let head_dim = 256;
        // 文本 24 层（blk.0..23）+ MTP 1 层（`nextn.*`，跳过）；按 ssm_alpha 存在性路由
        let mut layers = Vec::with_capacity(24);
        for layer in 0..24 {
            if Self::is_linear_layer(content, layer) {
                layers.push(IndexLayer::Linear(Self::load_linear(
                    content, reader, device, layer,
                )?));
            } else {
                layers.push(IndexLayer::Full(Self::load_full(
                    content, reader, device, layer,
                )?));
            }
        }
        let linear = layers
            .iter()
            .filter(|layer| matches!(layer, IndexLayer::Linear(_)))
            .count();
        let full = layers
            .iter()
            .filter(|layer| matches!(layer, IndexLayer::Full(_)))
            .count();
        anyhow::ensure!(
            linear == INDEX_LINEAR_LAYERS,
            "expected {INDEX_LINEAR_LAYERS} linear layers, got {linear}"
        );
        anyhow::ensure!(
            full == INDEX_FULL_LAYERS,
            "expected {INDEX_FULL_LAYERS} full layers, got {full}"
        );
        let token_embd = load_qmat(content, reader, device, "token_embd.weight")?;
        let output_proj = token_embd.clone();
        let max_seq_len = max_seq_len.min(context_length);
        let (rope_cos, rope_sin) =
            precompute_freqs_cis(INDEX_ROTARY_DIM, freq_base, max_seq_len, device)?;
        Ok(Self {
            token_embd,
            output_norm_weight: load_f32(content, reader, device, "output_norm.weight")?,
            output_proj,
            layers,
            vocab_size: 248320,
            head_count,
            head_count_kv,
            head_dim,
            rms_norm_eps,
            freq_base,
            max_seq_len,
            rope_cos,
            rope_sin,
        })
    }

    /// 从同一 GGUF 同时加载权重与官方 tokenizer 元数据。
    pub(crate) fn open(
        path: &Path,
        device: &Device,
        max_seq_len: usize,
    ) -> Result<(Self, Tokenizer)> {
        let mut reader = BufReader::new(File::open(path)?);
        let content = Content::read(&mut reader)?;
        let weights = Self::from_gguf(&content, &mut reader, device, max_seq_len)?;
        let tokenizer = candle_core::quantized::tokenizer::TokenizerFromGguf::from_gguf(&content)?;
        Ok((weights, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_constant_matches_p0_forensics() {
        assert_eq!(INDEX_ARCH, "qwen35");
        assert_eq!(INDEX_LINEAR_LAYERS, 18);
        assert_eq!(INDEX_FULL_LAYERS, 6);
    }
}
