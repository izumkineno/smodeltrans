//! Hy-MT2 的 Hunyuan-Dense GGUF 加载与自回归推理实现。
//!
//! 本模块直接读取 Candle GGUF 内容，使用量化矩阵执行 Transformer 层，
//! 并提供单次推理和保留 KV cache 的多轮会话两种入口。模型输出使用贪心
//! 解码，再通过 [`stream`] 模块按稳定的 Unicode 字符增量输出。

use super::generation::{HyDecoderState, HyTokenRng};
use crate::model_config::{GenerationConfig, MAX_TOP_K};
use anyhow::{Context, Result};
use candle_core::{
    DType, Device, IndexOp, Module, Tensor,
    quantized::{
        QMatMul, QStorage, QTensor,
        gguf_file::{Content, Value},
    },
};
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;
use candle_nn::ops::{rms_norm, silu};
use std::{
    borrow::Cow,
    fs::File,
    io::BufReader,
    path::Path,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};
use tokenizers::{
    Tokenizer,
    tokenizer::{Encoding, step_decode_stream},
};
#[cfg(not(feature = "flash-attn"))]
fn require_flash_attn() -> Result<()> {
    anyhow::bail!("Hy requires feature \"flash-attn\" when feature \"cuda\" is enabled");
}

#[cfg(feature = "flash-attn")]
fn require_flash_attn() -> Result<()> {
    Ok(())
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{:.2}ms", duration.as_secs_f64() * 1000.0)
    }
}

/// Hy tokenizer 使用的句子结束 token ID。
const EOS_TOKEN_ID: u32 = 120020;
/// Upper bound for the experimental GPU sampler's candidate set.
///
/// Candle's cumulative-sum implementation materializes an `n × n` helper
/// matrix, so unbounded `top_k` would turn a sampling option into an
/// unbounded GPU allocation. The value is shared with settings validation.

/// Hard metadata limits applied before RoPE or model-weight allocation.
const MAX_HY_CONTEXT_LENGTH: usize = 262_144;
const MAX_HY_EMBEDDING_LENGTH: usize = 32_768;
const MAX_HY_BLOCK_COUNT: usize = 256;
const MAX_HY_HEAD_COUNT: usize = 512;

/// Bound request text before tokenizer and attention allocations.
const MAX_HY_PROMPT_BYTES: usize = 1_048_576;
const MAX_HY_VOCAB_SIZE: usize = 1_000_000;
const MAX_HY_TOKEN_BYTES: usize = 1_048_576;
const MAX_HY_TOKEN_BYTES_TOTAL: usize = 64 * 1024 * 1024;

/// 使用 RMS 归一化的权重和数值稳定项。
#[derive(Clone, Debug)]
struct SimpleRmsNorm {
    /// 逐隐藏维度相乘的可训练缩放权重。
    weight: Tensor,
    /// 防止平方均值为零时除零的 epsilon。
    eps: f64,
}

impl SimpleRmsNorm {
    /// 使用已经从 GGUF 读取的权重创建归一化层。
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// 对最后一个维度执行 RMS normalization。
    ///
    /// 输入和输出都保持相同的 shape,因此前面的维度会保持不变。
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        rms_norm(&xs.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// Query/key projections can share input quantization when their GGUF dtypes
/// match.
enum QueryKeyProjection {
    Fused(QMatMul),
    Separate { query: QMatMul, key: QMatMul },
}
/// 一个 Hunyuan-Dense Transformer block 的全部量化权重。
struct LayerWeights {
    /// Query/key projections, fused when their quantized dtypes match.
    attention_qk: QueryKeyProjection,
    /// Value projection.
    attention_wv: QMatMul,
    /// Attention output projection.
    attention_wo: QMatMul,
    /// RMS normalization before attention.
    attention_norm: SimpleRmsNorm,
    /// RMS normalization for each query head.
    query_norm: SimpleRmsNorm,
    /// RMS normalization for each key head.
    key_norm: SimpleRmsNorm,
    /// Fused SwiGLU gate/up projections.
    feed_forward_gate_up: QMatMul,
    /// SwiGLU down projection.
    feed_forward_down: QMatMul,
    /// RMS normalization before the feed-forward network.
    ffn_norm: SimpleRmsNorm,
    /// Number of query heads.
    n_head: usize,
    /// Number of key/value heads; GQA replicates these when smaller.
    n_kv_head: usize,
    /// Hidden dimension of each attention head.
    head_dim: usize,
    /// Intermediate dimension of the fused gate/up projection.
    feed_forward_size: usize,
    /// Precomputed RoPE cosine table.
    cos: Tensor,
    /// Precomputed RoPE sine table.
    sin: Tensor,
}

/// Hy 会话独占的单层 key/value cache。
type LayerCache = Option<(Arc<Tensor>, Arc<Tensor>)>;
static HY_SELECTION_PROFILE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("SMODELTRANS_PROFILE_HY").is_some());

fn hy_selection_profile_enabled() -> bool {
    *HY_SELECTION_PROFILE_ENABLED
}

static HY_LOGIT_VALIDATION_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("SMODELTRANS_VALIDATE_HY_LOGITS").is_some());

static HY_TRANSFER_TRACE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("SMODELTRANS_TRACE_HY_TRANSFERS").is_some());

fn trace_hy_transfer(direction: &str, operation: &str, bytes: usize) {
    if *HY_TRANSFER_TRACE_ENABLED {
        eprintln!(
            "[hy-transfer] direction={direction} bytes={bytes} count=1 operation={operation}"
        );
    }
}
fn validate_finite_logits(logits: &Tensor) -> Result<()> {
    let logits = logits.to_dtype(DType::F32)?;
    let equal_to_self = logits.broadcast_eq(&logits)?.to_dtype(DType::F32)?;
    let infinity = Tensor::full(f32::INFINITY, logits.shape(), logits.device())?;
    let below_infinity = logits
        .abs()?
        .broadcast_lt(&infinity)?
        .to_dtype(DType::F32)?;
    let finite = equal_to_self.broadcast_mul(&below_infinity)?;
    trace_hy_transfer(
        "d2h",
        "finite_logits_validation",
        std::mem::size_of::<f32>(),
    );
    let minimum = finite.min_all()?.to_scalar::<f32>()?;
    anyhow::ensure!(
        minimum >= 1.0,
        "Hy backend returned non-finite logits; validation failed"
    );
    Ok(())
}

#[derive(Default)]
struct HySelectionProfile {
    enabled: bool,
    flatten: Duration,
    penalty_math: Duration,
    greedy_argmax: Duration,
    sample_top_k: Duration,
    sample_math: Duration,
    sample_select: Duration,
}

impl HySelectionProfile {
    fn new() -> Self {
        Self {
            enabled: hy_selection_profile_enabled(),
            ..Self::default()
        }
    }

    fn print(&self) {
        if self.enabled {
            eprintln!(
                "[profile] hy-select: flatten={} penalty_math={} greedy_argmax={} sample_top_k={} sample_math={} sample_select={}",
                format_duration(self.flatten),
                format_duration(self.penalty_math),
                format_duration(self.greedy_argmax),
                format_duration(self.sample_top_k),
                format_duration(self.sample_math),
                format_duration(self.sample_select),
            );
        }
    }
}
/// Device-side scalar and RNG state reused by every selection in one request.
///
/// Configuration scalars are uploaded once. Sampling thresholds are generated
/// with the existing CPU RNG in advance and uploaded as one device buffer, so
/// the decode loop does not perform a scalar H2D upload per token.
pub(super) struct HySelectionTensors {
    repetition_penalty: Tensor,
    frequency_penalty: Tensor,
    temperature: Tensor,
    top_p: Option<Tensor>,
    thresholds: Option<Tensor>,
}

/// Return the first true position of a device-side mask, or the final
/// position when no true value exists.
///
/// The difference between the mask and its zero-prefixed predecessor contains
/// one `1` at the first true position. A final-position sentinel makes the
/// no-match behavior deterministic without reading a device scalar on the
/// host; valid cumulative masks normally hit before that sentinel.
fn first_true_or_last_index(mask: &Tensor) -> Result<Tensor> {
    let mask = mask.to_dtype(DType::F32)?;
    let length = mask.dims1()?;
    anyhow::ensure!(length > 0, "cannot find a position in an empty mask");
    let positions = Tensor::arange(0u32, u32::try_from(length)?, mask.device())?;
    let last_index = positions.narrow(0, length - 1, 1)?.squeeze(0)?;
    let sentinel = positions.broadcast_eq(&last_index)?.to_dtype(DType::F32)?;
    let mask = mask.broadcast_maximum(&sentinel)?;
    let shifted = if length == 1 {
        Tensor::zeros(1, DType::F32, mask.device())?
    } else {
        let zero = Tensor::zeros(1, DType::F32, mask.device())?;
        let prefix = mask.narrow(0, 0, length - 1)?;
        Tensor::cat(&[&zero, &prefix], 0)?
    };
    Ok(mask.broadcast_sub(&shifted)?.argmax(0)?)
}
static HY_LAYER_PROFILE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("SMODELTRANS_PROFILE_HY_LAYERS").is_some());

#[derive(Default)]
struct HyLayerProfile {
    attention: Duration,
    attention_projection: Duration,
    attention_preprocess: Duration,
    attention_cache: Duration,
    attention_flash: Duration,
    attention_output: Duration,
    feed_forward: Duration,
}

impl HyLayerProfile {
    fn enabled() -> bool {
        *HY_LAYER_PROFILE_ENABLED
    }

    fn print(&self, device: &Device, layer_count: usize) -> candle_core::Result<()> {
        if !Self::enabled() {
            return Ok(());
        }
        device.synchronize()?;
        eprintln!(
            "[profile] hy-layers: count={} attention={} projection={} preprocess={} cache={} flash={} output={} feed_forward={} total={}",
            layer_count,
            format_duration(self.attention),
            format_duration(self.attention_projection),
            format_duration(self.attention_preprocess),
            format_duration(self.attention_cache),
            format_duration(self.attention_flash),
            format_duration(self.attention_output),
            format_duration(self.feed_forward),
            format_duration(self.attention + self.feed_forward),
        );
        Ok(())
    }
}

impl LayerWeights {
    /// 为当前序列位置截取 RoPE 表并应用旋转位置编码。
    fn apply_rotary_emb(&self, xs: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        let seq_len = xs.dim(2)?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)
    }

    /// 执行单个 block 的多头注意力和 KV cache 更新。
    ///
    /// `xs` 的布局为 `[batch, sequence, hidden]`。内部会转换为按头布局，
    /// 将当前 key/value 与历史 cache 拼接，然后使用 GQA 将 KV 头复制到
    /// query 头数量，最后恢复为隐藏层布局。
    fn forward_attn(
        &self,
        xs: &Tensor,
        index_pos: usize,
        cache: &mut LayerCache,
        mut profile: Option<&mut HyLayerProfile>,
    ) -> candle_core::Result<Tensor> {
        let projection_started = if profile.is_some() {
            xs.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let (batch, seq_len, hidden_size) = xs.dims3()?;
        let xs = xs.contiguous()?;
        let xs = self.attention_norm.forward(&xs)?;

        let (q, k) = match &self.attention_qk {
            QueryKeyProjection::Fused(weight) => {
                let query_size = self.n_head * self.head_dim;
                let key_size = self.n_kv_head * self.head_dim;
                let qk = weight.forward(&xs)?;
                let q = qk.narrow(2, 0, query_size)?;
                let q = if seq_len == 1 { q } else { q.contiguous()? };
                let q = q
                    .reshape((batch, seq_len, self.n_head, self.head_dim))?
                    .transpose(1, 2)?;
                let k = qk.narrow(2, query_size, key_size)?;
                let k = if seq_len == 1 { k } else { k.contiguous()? };
                let k = k
                    .reshape((batch, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                (q, k)
            }
            QueryKeyProjection::Separate { query, key } => {
                let q = query
                    .forward(&xs)?
                    .reshape((batch, seq_len, self.n_head, self.head_dim))?
                    .transpose(1, 2)?;
                let k = key
                    .forward(&xs)?
                    .reshape((batch, seq_len, self.n_kv_head, self.head_dim))?
                    .transpose(1, 2)?;
                (q, k)
            }
        };
        let v = self
            .attention_wv
            .forward(&xs)?
            .reshape((batch, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        if let Some(started) = projection_started {
            xs.device().synchronize()?;
            profile
                .as_deref_mut()
                .expect("layer profile must exist when timing projections")
                .attention_projection += started.elapsed();
        }

        // Hunyuan-Dense 在 RoPE 之后对每个注意力头的 Q/K 应用 RMS norm。
        let preprocess_started = if profile.is_some() {
            xs.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let q = self
            .query_norm
            .forward(&self.apply_rotary_emb(&q, index_pos)?)?
            .contiguous()?;
        let k = self
            .key_norm
            .forward(&self.apply_rotary_emb(&k, index_pos)?)?
            .contiguous()?;
        let v = v.contiguous()?;
        if let Some(started) = preprocess_started {
            xs.device().synchronize()?;
            profile
                .as_deref_mut()
                .expect("layer profile must exist when timing attention preprocessing")
                .attention_preprocess += started.elapsed();
        }

        let cache_started = if profile.is_some() {
            xs.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let total_len = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle_core::Error::Msg("attention sequence length overflow".into()))?;
        let max_context = self.cos.dim(0)?;
        if total_len > max_context {
            return Err(candle_core::Error::Msg(
                "sequence exceeds model context length".to_owned(),
            ));
        }

        let cache_dtype = DType::F16;
        let cache_cap = match cache.as_ref() {
            Some((cached_k, _)) => cached_k.dim(2)?,
            None => 0,
        };
        if cache_cap < total_len {
            let new_cap = if cache_cap == 0 {
                total_len.max(64).next_power_of_two().min(max_context)
            } else {
                cache_cap.saturating_mul(2).max(total_len).min(max_context)
            };
            let new_k = Tensor::zeros(
                (batch, self.n_kv_head, new_cap, self.head_dim),
                cache_dtype,
                k.device(),
            )?;
            let new_v = Tensor::zeros(
                (batch, self.n_kv_head, new_cap, self.head_dim),
                cache_dtype,
                v.device(),
            )?;
            if let Some((old_k, old_v)) = cache.take() {
                if index_pos > 0 {
                    new_k.slice_set(&old_k.narrow(2, 0, index_pos)?.contiguous()?, 2, 0)?;
                    new_v.slice_set(&old_v.narrow(2, 0, index_pos)?.contiguous()?, 2, 0)?;
                }
            }
            *cache = Some((Arc::new(new_k), Arc::new(new_v)));
        }

        let (cached_k, cached_v) = cache.as_mut().expect("cache must be initialized");
        let k = k.to_dtype(cache_dtype)?.contiguous()?;
        if Arc::strong_count(cached_k) > 1 {
            *cached_k = Arc::new((**cached_k).copy()?);
        }
        if Arc::strong_count(cached_v) > 1 {
            *cached_v = Arc::new((**cached_v).copy()?);
        }
        let v = v.to_dtype(cache_dtype)?.contiguous()?;
        cached_k.slice_set(&k, 2, index_pos)?;
        cached_v.slice_set(&v, 2, index_pos)?;
        if let Some(started) = cache_started {
            xs.device().synchronize()?;
            profile
                .as_deref_mut()
                .expect("layer profile must exist when timing KV cache")
                .attention_cache += started.elapsed();
        }
        #[cfg(feature = "flash-attn")]
        {
            let flash_started = if profile.is_some() {
                xs.device().synchronize()?;
                Some(Instant::now())
            } else {
                None
            };
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = cached_k.narrow(2, 0, total_len)?.transpose(1, 2)?;
            let v = cached_v.narrow(2, 0, total_len)?.transpose(1, 2)?;
            let flash_output = flash_attn(&q, &k, &v, 1.0 / (self.head_dim as f32).sqrt(), true)?;
            if let Some(started) = flash_started {
                xs.device().synchronize()?;
                profile
                    .as_deref_mut()
                    .expect("layer profile must exist when timing FlashAttention")
                    .attention_flash += started.elapsed();
            }
            let output_started = if profile.is_some() {
                xs.device().synchronize()?;
                Some(Instant::now())
            } else {
                None
            };
            let output = flash_output.reshape((batch, seq_len, hidden_size))?;
            let output = self.attention_wo.forward(&output)?;
            if let Some(started) = output_started {
                output.device().synchronize()?;
                profile
                    .as_deref_mut()
                    .expect("layer profile must exist when timing attention output")
                    .attention_output += started.elapsed();
            }
            return Ok(output);
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            return Err(candle_core::Error::Msg(
                "Hy attention requires feature \"flash-attn\"".to_owned(),
            ));
        }
    }

    /// 执行一个 Transformer block：注意力残差之后再执行 SwiGLU 前馈网络。
    fn forward(
        &self,
        xs: &Tensor,
        index_pos: usize,
        cache: &mut LayerCache,
        mut profile: Option<&mut HyLayerProfile>,
    ) -> candle_core::Result<Tensor> {
        let attn_started = if profile.is_some() {
            xs.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let residual = xs.clone();
        let attn = self.forward_attn(&xs, index_pos, cache, profile.as_deref_mut())?;
        let xs = (attn + residual)?;
        if let Some(started) = attn_started {
            xs.device().synchronize()?;
            profile
                .as_deref_mut()
                .expect("layer profile must exist when timing attention")
                .attention += started.elapsed();
        }

        let feed_forward_started = if profile.is_some() {
            xs.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let residual = xs.clone();
        let xs = self.ffn_norm.forward(&xs)?.contiguous()?;
        // Gate and up projections share the same quantized input.
        let gate_up = self.feed_forward_gate_up.forward(&xs)?;
        let gate = gate_up.narrow(2, 0, self.feed_forward_size)?;
        let up = gate_up.narrow(2, self.feed_forward_size, self.feed_forward_size)?;
        let mlp_input = (silu(&gate)? * up)?;
        let mlp = self.feed_forward_down.forward(&mlp_input.contiguous()?)?;
        let output = (mlp + residual)?;
        if let Some(started) = feed_forward_started {
            output.device().synchronize()?;
            profile
                .as_deref_mut()
                .expect("layer profile must exist when timing feed-forward")
                .feed_forward += started.elapsed();
        }
        Ok(output)
    }
}

/// Hy 模型的嵌入层、Transformer blocks 以及只读输出资源。
struct ModelWeights {
    /// 将 token ID 映射到隐藏向量的量化嵌入矩阵。
    token_embeddings: QMatMul,
    /// 按模型深度排列的 Transformer blocks。
    layers: Vec<LayerWeights>,
    /// 输出 logits 前的 RMS normalization。
    output_norm: SimpleRmsNorm,
    /// 将最终隐藏向量投影到词表的量化矩阵。
    output: QMatMul,
    /// 词表行数，用于分配 GPU 常驻 token 计数表。
    vocab_size: usize,
}

/// 从 GGUF 元数据中读取指定键，并在缺失时附带键名返回错误。
fn metadata<'a>(
    content: &'a Content,
    key: &str,
) -> Result<&'a candle_core::quantized::gguf_file::Value> {
    content
        .metadata
        .get(key)
        .with_context(|| format!("missing GGUF metadata key '{key}'"))
}

fn validate_vocab_metadata(content: &Content, embedding_length: usize) -> Result<usize> {
    let embedding_info = content
        .tensor_infos
        .get("token_embd.weight")
        .context("GGUF is missing token_embd.weight metadata")?;
    let (vocab_size, embedding_width) = embedding_info.shape.dims2()?;
    anyhow::ensure!(
        vocab_size > 0 && vocab_size <= MAX_HY_VOCAB_SIZE && embedding_width == embedding_length,
        "invalid Hunyuan embedding shape"
    );
    let values = match metadata(content, "tokenizer.ggml.tokens")? {
        Value::Array(values) => values,
        _ => anyhow::bail!("GGUF tokenizer vocabulary is not an array"),
    };
    anyhow::ensure!(
        values.len() == vocab_size,
        "GGUF tokenizer vocabulary does not match embedding rows"
    );
    let mut total_bytes = 0usize;
    for value in values {
        let token = value.to_string()?;
        anyhow::ensure!(
            token.len() <= MAX_HY_TOKEN_BYTES,
            "GGUF tokenizer token exceeds the byte limit"
        );
        total_bytes = total_bytes
            .checked_add(token.len())
            .context("GGUF tokenizer byte count overflowed")?;
    }
    anyhow::ensure!(
        total_bytes <= MAX_HY_TOKEN_BYTES_TOTAL,
        "GGUF tokenizer vocabulary exceeds the byte limit"
    );
    Ok(vocab_size)
}

fn load_norm_weight<R: std::io::Read + std::io::Seek>(
    content: &Content,
    reader: &mut R,
    device: &Device,
    key: &str,
) -> Result<Tensor> {
    let weight = content
        .tensor(reader, key, device)?
        .dequantize(device)?
        .to_dtype(DType::F16)?;
    Ok(weight)
}
/// Combine quantized matrices along their output-row dimension.
///
/// `QMatMul` stores quantized weights as `[output, input]`. Concatenating
/// complete raw matrices therefore preserves every quantized row while
/// sharing the input quantization across the fused projections.
fn fuse_quantized_rows(tensors: &[QTensor]) -> Result<QMatMul> {
    anyhow::ensure!(
        !tensors.is_empty(),
        "cannot fuse an empty quantized matrix list"
    );
    let first = &tensors[0];
    let (first_rows, cols) = first.shape().dims2()?;
    let dtype = first.dtype();
    let device = first.device();
    anyhow::ensure!(
        first_rows > 0 && cols > 0,
        "cannot fuse an empty quantized matrix"
    );

    let mut rows = first_rows;
    let mut data = first.data()?.into_owned();
    for tensor in &tensors[1..] {
        let (tensor_rows, tensor_cols) = tensor.shape().dims2()?;
        anyhow::ensure!(
            tensor_cols == cols,
            "cannot fuse quantized matrices with different input widths: {cols} vs {tensor_cols}"
        );
        anyhow::ensure!(
            tensor.dtype() == dtype,
            "cannot fuse quantized matrices with different dtypes: {dtype:?} vs {:?}",
            tensor.dtype()
        );
        rows = rows
            .checked_add(tensor_rows)
            .ok_or_else(|| anyhow::anyhow!("fused quantized row count overflow"))?;
        data.extend_from_slice(tensor.data()?.as_ref());
    }

    let storage = QStorage::from_data(Cow::Owned(data), &device, dtype)?;
    let tensor = QTensor::new(storage, (rows, cols))?;
    Ok(QMatMul::from_qtensor(tensor)?)
}
/// 为所有可能的序列位置预计算 RoPE 的余弦和正弦表。
///
/// 返回的两个 tensor 形状都是 `[max_seq_len, head_dim / 2]`，其中每一行
/// 对应一个绝对位置。
fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
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

impl ModelWeights {
    /// 根据 GGUF 元数据和量化 tensor 构造完整的 Hy 模型。
    ///
    /// 这里同时校验模型架构、GQA 头数以及最大上下文长度，并为每个 block
    /// 建立权重引用。权重由 Candle 按需从 GGUF reader 加载到指定设备。
    fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        device: &Device,
        max_seq_len: usize,
    ) -> Result<Self> {
        let architecture = metadata(content, "general.architecture")?
            .to_string()?
            .to_owned();
        anyhow::ensure!(
            architecture == "hunyuan-dense",
            "unsupported GGUF architecture '{architecture}'"
        );

        let head_count =
            metadata(content, "hunyuan-dense.attention.head_count")?.to_u32()? as usize;
        let head_count_kv =
            metadata(content, "hunyuan-dense.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length =
            metadata(content, "hunyuan-dense.embedding_length")?.to_u32()? as usize;
        let block_count = metadata(content, "hunyuan-dense.block_count")?.to_u32()? as usize;
        let context_length = metadata(content, "hunyuan-dense.context_length")?.to_u32()? as usize;
        let rms_norm_eps =
            metadata(content, "hunyuan-dense.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = metadata(content, "hunyuan-dense.rope.freq_base")?.to_f32()?;
        anyhow::ensure!(
            head_count > 0 && head_count <= MAX_HY_HEAD_COUNT,
            "invalid Hunyuan attention head count: {head_count}"
        );
        anyhow::ensure!(
            head_count_kv > 0 && head_count_kv <= head_count,
            "invalid Hunyuan KV head count: {head_count_kv}"
        );
        anyhow::ensure!(
            embedding_length > 0 && embedding_length <= MAX_HY_EMBEDDING_LENGTH,
            "invalid Hunyuan embedding length: {embedding_length}"
        );
        anyhow::ensure!(
            block_count > 0 && block_count <= MAX_HY_BLOCK_COUNT,
            "invalid Hunyuan block count: {block_count}"
        );
        anyhow::ensure!(
            context_length > 0 && context_length <= MAX_HY_CONTEXT_LENGTH,
            "invalid Hunyuan context length: {context_length}"
        );
        anyhow::ensure!(
            rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
            "invalid Hunyuan RMS norm epsilon: {rms_norm_eps}"
        );
        anyhow::ensure!(
            rope_freq_base.is_finite() && rope_freq_base > 0.0,
            "invalid Hunyuan RoPE frequency base: {rope_freq_base}"
        );
        anyhow::ensure!(
            embedding_length.is_multiple_of(head_count),
            "Hunyuan embedding length must be divisible by head count"
        );
        let head_dim = embedding_length / head_count;
        anyhow::ensure!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "Hunyuan head dimension must be positive and even: {head_dim}"
        );
        anyhow::ensure!(
            head_count.is_multiple_of(head_count_kv),
            "invalid Hunyuan GQA head counts"
        );
        anyhow::ensure!(
            max_seq_len > 0 && max_seq_len <= context_length,
            "sequence exceeds model context length"
        );
        let embedding_info = content
            .tensor_infos
            .get("token_embd.weight")
            .context("GGUF is missing token_embd.weight metadata")?;
        let (vocab_size, embedding_width) = embedding_info.shape.dims2()?;
        anyhow::ensure!(
            vocab_size > 0 && vocab_size <= MAX_HY_VOCAB_SIZE,
            "invalid Hunyuan vocabulary size: {vocab_size}"
        );
        anyhow::ensure!(
            embedding_width == embedding_length,
            "Hunyuan embedding width does not match metadata"
        );
        anyhow::ensure!(
            (EOS_TOKEN_ID as usize) < vocab_size,
            "Hy EOS token ID is outside the model vocabulary"
        );
        if let Some(output_info) = content.tensor_infos.get("output.weight") {
            let output_shape = output_info.shape.dims2()?;
            anyhow::ensure!(
                output_shape == (vocab_size, embedding_length),
                "Hunyuan output projection shape does not match token embeddings"
            );
        }

        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, max_seq_len, device)?;
        let cos = cos.to_dtype(DType::F16)?;
        let sin = sin.to_dtype(DType::F16)?;
        let token_embeddings =
            QMatMul::from_qtensor(content.tensor(reader, "token_embd.weight", device)?)?;
        let output_norm = SimpleRmsNorm::new(
            load_norm_weight(content, reader, device, "output_norm.weight")?,
            rms_norm_eps,
        );
        // Hy 将 LM head 与 token_embd.weight 绑定；没有独立 output.weight 时
        // 直接克隆 QMatMul 句柄，避免再次分配整份词表权重。
        let output = if content.tensor_infos.contains_key("output.weight") {
            QMatMul::from_qtensor(content.tensor(reader, "output.weight", device)?)?
        } else {
            token_embeddings.clone()
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_qk_tensors = [
                content.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?,
                content.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
            ];
            let attention_qk = if attention_qk_tensors[0].dtype() == attention_qk_tensors[1].dtype()
            {
                QueryKeyProjection::Fused(fuse_quantized_rows(&attention_qk_tensors)?)
            } else {
                let [query, key] = attention_qk_tensors;
                QueryKeyProjection::Separate {
                    query: QMatMul::from_qtensor(query)?,
                    key: QMatMul::from_qtensor(key)?,
                }
            };
            let attention_wv = QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_v.weight"),
                device,
            )?)?;
            let attention_wo = QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.attn_output.weight"),
                device,
            )?)?;
            let attention_norm = SimpleRmsNorm::new(
                load_norm_weight(
                    content,
                    reader,
                    device,
                    &format!("{prefix}.attn_norm.weight"),
                )?,
                rms_norm_eps,
            );
            let query_norm = SimpleRmsNorm::new(
                load_norm_weight(
                    content,
                    reader,
                    device,
                    &format!("{prefix}.attn_q_norm.weight"),
                )?,
                rms_norm_eps,
            );
            let key_norm = SimpleRmsNorm::new(
                load_norm_weight(
                    content,
                    reader,
                    device,
                    &format!("{prefix}.attn_k_norm.weight"),
                )?,
                rms_norm_eps,
            );
            let feed_forward_gate_up_tensors = [
                content.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?,
                content.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
            ];
            let (feed_forward_size, _) = feed_forward_gate_up_tensors[0].shape().dims2()?;
            let feed_forward_gate_up = fuse_quantized_rows(&feed_forward_gate_up_tensors)?;
            let feed_forward_down = QMatMul::from_qtensor(content.tensor(
                reader,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?)?;
            let ffn_norm = SimpleRmsNorm::new(
                load_norm_weight(
                    content,
                    reader,
                    device,
                    &format!("{prefix}.ffn_norm.weight"),
                )?,
                rms_norm_eps,
            );
            layers.push(LayerWeights {
                attention_qk,
                attention_wv,
                attention_wo,
                attention_norm,
                query_norm,
                key_norm,
                feed_forward_gate_up,
                feed_forward_down,
                ffn_norm,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                feed_forward_size,
                cos: cos.clone(),
                sin: sin.clone(),
            });
        }
        Ok(Self {
            token_embeddings,
            layers,
            output_norm,
            output,
            vocab_size,
        })
    }

    /// 执行一次隐藏状态前向传播；cache 由调用方会话持有。
    fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        caches: &mut [LayerCache],
    ) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let mut hidden = self
            .token_embeddings
            .embedding(input_ids)?
            .to_dtype(DType::F16)?;
        let layer_profile_enabled = HyLayerProfile::enabled();
        let mut layer_profile = HyLayerProfile::default();
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(
                &hidden,
                index_pos,
                cache,
                layer_profile_enabled.then_some(&mut layer_profile),
            )?;
        }
        layer_profile.print(input_ids.device(), self.layers.len())?;
        let output_norm_started = if layer_profile_enabled {
            input_ids.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let hidden = self.output_norm.forward(&hidden)?;
        if let Some(started) = output_norm_started {
            hidden.device().synchronize()?;
            eprintln!(
                "[profile] hy-output: norm={}",
                format_duration(started.elapsed())
            );
        }
        let hidden = hidden.i((.., seq_len - 1, ..))?.contiguous()?;
        let output_started = if layer_profile_enabled {
            hidden.device().synchronize()?;
            Some(Instant::now())
        } else {
            None
        };
        let logits = self.output.forward(&hidden)?;
        if let Some(started) = output_started {
            logits.device().synchronize()?;
            eprintln!(
                "[profile] hy-output: projection={}",
                format_duration(started.elapsed())
            );
        }
        Ok(logits)
    }
}
/// 使用首轮 Hy 对话模板编码 system 与 user 提示词。
fn encode_prompt(
    tokenizer: &Tokenizer,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<Encoding> {
    encode_turn(tokenizer, system_prompt, user_prompt, true)
}

/// 使用后续轮次模板编码 user 提示词，不重复添加 system 与会话起始标记。
fn encode_followup_prompt(tokenizer: &Tokenizer, user_prompt: &str) -> Result<Encoding> {
    encode_turn(tokenizer, "", user_prompt, false)
}

/// 按 Hy 1.8B 官方首轮/续轮模板拼接角色提示词。
fn format_turn_prompt(system_prompt: &str, user_prompt: &str, first_turn: bool) -> String {
    const BEGIN: &str = "<｜hy_begin▁of▁sentence｜>";
    const SYSTEM_SUFFIX: &str = "<｜hy_place▁holder▁no▁3｜>";
    const USER_PREFIX: &str = "<｜hy_User｜>";
    const ASSISTANT_PREFIX: &str = "<｜hy_Assistant｜>";

    let system_len = if first_turn && !system_prompt.is_empty() {
        system_prompt.len() + SYSTEM_SUFFIX.len()
    } else {
        0
    };
    let mut formatted = String::with_capacity(
        if first_turn { BEGIN.len() } else { 0 }
            + system_len
            + USER_PREFIX.len()
            + user_prompt.len()
            + ASSISTANT_PREFIX.len(),
    );
    if first_turn {
        formatted.push_str(BEGIN);
        if !system_prompt.is_empty() {
            formatted.push_str(system_prompt);
            formatted.push_str(SYSTEM_SUFFIX);
        }
    }
    formatted.push_str(USER_PREFIX);
    formatted.push_str(user_prompt);
    formatted.push_str(ASSISTANT_PREFIX);
    formatted
}

/// 根据是否为首轮选择 Hy 的角色模板并进行 tokenization。
fn encode_turn(
    tokenizer: &Tokenizer,
    system_prompt: &str,
    user_prompt: &str,
    first_turn: bool,
) -> Result<Encoding> {
    tokenizer
        .encode(
            format_turn_prompt(system_prompt, user_prompt, first_turn),
            first_turn,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn decode_token_delta(
    tokenizer: &Tokenizer,
    token_ids: &mut Vec<u32>,
    prefix: &mut String,
    prefix_index: &mut usize,
    token_id: u32,
) -> Result<Vec<u8>> {
    let chunk = step_decode_stream(
        tokenizer,
        vec![token_id],
        true,
        token_ids,
        prefix,
        prefix_index,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(chunk.unwrap_or_default().into_bytes())
}

/// Immutable Hy model resources plus tokenizer state.
///
/// Request cache and conversation state are owned by the Hy session driver,
/// never by this immutable model resource.
pub(crate) struct HySession {
    tokenizer: Tokenizer,
    model: ModelWeights,
    device: Device,
}

impl HySession {
    /// Open GGUF resources once; no request state is retained here.
    pub(crate) fn new(model_path: &Path, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            device.is_cuda(),
            "Hy currently requires an explicit CUDA device"
        );
        require_flash_attn()?;
        let mut reader = BufReader::new(File::open(model_path)?);
        let content = Content::read(&mut reader)?;
        let embedding_length =
            metadata(&content, "hunyuan-dense.embedding_length")?.to_u32()? as usize;
        validate_vocab_metadata(&content, embedding_length)?;
        let tokenizer = candle_core::quantized::tokenizer::TokenizerFromGguf::from_gguf(&content)?;
        let context_length = metadata(&content, "hunyuan-dense.context_length")?.to_u32()? as usize;
        anyhow::ensure!(context_length > 0, "GGUF context length must be positive");
        let model = ModelWeights::from_gguf(&content, &mut reader, device, context_length)?;
        Ok(Self {
            tokenizer,
            model,
            device: device.clone(),
        })
    }

    fn max_context_length(&self) -> Result<usize> {
        self.model
            .layers
            .first()
            .context("Hy model has no transformer layers")?
            .cos
            .dim(0)
            .map_err(Into::into)
    }

    pub(super) fn has_context_capacity(&self, position: usize) -> Result<bool> {
        Ok(position < self.max_context_length()?)
    }

    pub(super) fn new_state(&self) -> Result<HyGenerationState> {
        HyGenerationState::new(
            self.model.layers.len(),
            self.max_context_length()?,
            &self.device,
        )
    }

    pub(super) fn prepare_penalty_state(
        &self,
        state: &mut HyGenerationState,
        position: usize,
        config: &GenerationConfig,
    ) -> Result<()> {
        let enabled = config.repetition_penalty != 1.0 || config.frequency_penalty != 0.0;
        state.prepare_penalty_counts(enabled, position, self.model.vocab_size, &self.device)
    }

    pub(super) fn prepare_selection(
        &self,
        config: &GenerationConfig,
        rng: &mut dyn HyTokenRng,
    ) -> Result<HySelectionTensors> {
        let thresholds = if config.sampling {
            let mut values = Vec::with_capacity(config.max_new_tokens);
            for _ in 0..config.max_new_tokens {
                values.push(rng.next_f64());
            }
            Some(Tensor::new(values.as_slice(), &self.device)?)
        } else {
            None
        };
        trace_hy_transfer(
            "h2d",
            "selection_config_scalars",
            3 * std::mem::size_of::<f32>()
                + if config.top_p < 1.0 {
                    std::mem::size_of::<f64>()
                } else {
                    0
                },
        );
        if config.sampling {
            trace_hy_transfer(
                "h2d",
                "sampling_threshold_buffer",
                config.max_new_tokens * std::mem::size_of::<f64>(),
            );
        }
        Ok(HySelectionTensors {
            repetition_penalty: Tensor::new(config.repetition_penalty, &self.device)?,
            frequency_penalty: Tensor::new(config.frequency_penalty, &self.device)?,
            temperature: Tensor::new(config.temperature, &self.device)?,
            top_p: (config.top_p < 1.0)
                .then(|| Tensor::new(config.top_p as f64, &self.device))
                .transpose()?,
            thresholds,
        })
    }

    fn prompt_token_ids(
        &self,
        state: &HyGenerationState,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<Vec<u32>> {
        let prompt_bytes = system_prompt
            .len()
            .checked_add(user_prompt.len())
            .context("Hy prompt byte length overflowed usize")?;
        anyhow::ensure!(
            prompt_bytes <= MAX_HY_PROMPT_BYTES,
            "Hy prompt exceeds the {}-byte limit",
            MAX_HY_PROMPT_BYTES
        );
        let encoding = if state.first_turn {
            encode_prompt(&self.tokenizer, system_prompt, user_prompt)?
        } else {
            encode_followup_prompt(&self.tokenizer, user_prompt)?
        };
        let ids = encoding.get_ids().to_vec();
        anyhow::ensure!(
            ids.len() <= self.max_context_length()?,
            "Hy prompt exceeds the model context length"
        );
        Ok(ids)
    }

    pub(super) fn prompt_token_count(
        &self,
        state: &HyGenerationState,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<usize> {
        Ok(self
            .prompt_token_ids(state, system_prompt, user_prompt)?
            .len())
    }

    fn apply_token_penalties(
        &self,
        logits: Tensor,
        state: &HyGenerationState,
        config: &GenerationConfig,
        selection: &HySelectionTensors,
        profile: &mut HySelectionProfile,
    ) -> Result<Tensor> {
        if config.repetition_penalty == 1.0 && config.frequency_penalty == 0.0 {
            return Ok(logits);
        }
        let math_started = Instant::now();
        let logits_dtype = logits.dtype();
        let mut adjusted = logits.to_dtype(DType::F32)?;
        let counts = state
            .penalty_counts
            .as_ref()
            .context("Hy penalty state was not prepared before token selection")?;
        let active = counts.gt(0f32)?;
        if config.repetition_penalty != 1.0 {
            let positive = adjusted.ge(0f32)?;
            let divided = adjusted.broadcast_div(&selection.repetition_penalty)?;
            let multiplied = adjusted.broadcast_mul(&selection.repetition_penalty)?;
            let transformed = positive.where_cond(&divided, &multiplied)?;
            adjusted = active.where_cond(&transformed, &adjusted)?;
        }
        if config.frequency_penalty != 0.0 {
            let penalties = counts.broadcast_mul(&selection.frequency_penalty)?;
            let transformed = adjusted.broadcast_sub(&penalties)?;
            adjusted = active.where_cond(&transformed, &adjusted)?;
        }
        if profile.enabled {
            profile.penalty_math += math_started.elapsed();
        }
        Ok(adjusted.to_dtype(logits_dtype)?)
    }

    fn prefill_with_ids(
        &self,
        state: &mut HyGenerationState,
        ids: &[u32],
        position: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(!ids.is_empty(), "prompt produced no tokenizer ids");
        let max_context = self.max_context_length()?;
        let end = position
            .checked_add(ids.len())
            .context("Hy prompt position overflowed")?;
        anyhow::ensure!(
            end <= max_context,
            "Hy prompt exceeds the model context length"
        );
        let input_tensor = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        trace_hy_transfer(
            "h2d",
            "prompt_token_ids",
            ids.len() * std::mem::size_of::<u32>(),
        );
        state.record_token_ids(&input_tensor, position)?;
        let logits = self
            .model
            .forward(&input_tensor, position, &mut state.kv_cache)?;
        state.first_turn = false;
        Ok(logits)
    }

    pub(super) fn prefill(
        &self,
        state: &mut HyGenerationState,
        system_prompt: &str,
        user_prompt: &str,
        position: usize,
    ) -> Result<(Tensor, Vec<u32>)> {
        let ids = self.prompt_token_ids(state, system_prompt, user_prompt)?;
        let logits = self.prefill_with_ids(state, &ids, position)?;
        Ok((logits, ids))
    }

    pub(super) fn step(
        &self,
        state: &mut HyGenerationState,
        token: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        let max_context = self.max_context_length()?;
        let end = position
            .checked_add(1)
            .context("Hy generation position overflowed usize")?;
        anyhow::ensure!(
            end <= max_context,
            "Hy generation exceeds the model context length"
        );
        anyhow::ensure!(
            token.elem_count() == 1,
            "Hy generation step requires exactly one device token"
        );
        let input = token.flatten_all()?.reshape((1, 1))?;
        state.record_token_ids(&input, position)?;
        Ok(self.model.forward(&input, position, &mut state.kv_cache)?)
    }

    /// Replay one assistant turn after a memory trim with one batched H2D copy.
    pub(super) fn replay_assistant_tokens(
        &self,
        state: &mut HyGenerationState,
        token_ids: &[u32],
        position: usize,
    ) -> Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let tokens = Tensor::new(token_ids, &self.device)?;
        trace_hy_transfer(
            "h2d",
            "replay_assistant_token_ids",
            token_ids.len() * std::mem::size_of::<u32>(),
        );
        for offset in 0..tokens.elem_count() {
            let token = tokens.narrow(0, offset, 1)?.reshape((1, 1))?;
            self.step(state, &token, position + offset)?;
        }
        Ok(())
    }

    pub(super) fn select_token(
        &self,
        state: &HyGenerationState,
        logits: &Tensor,
        config: &GenerationConfig,
        selection: &HySelectionTensors,
        selection_step: usize,
    ) -> Result<(u32, Tensor)> {
        let mut profile = HySelectionProfile::new();
        let flatten_started = Instant::now();
        let logits = logits.flatten_all()?.contiguous()?;
        if profile.enabled {
            profile.flatten += flatten_started.elapsed();
        }
        anyhow::ensure!(logits.elem_count() > 0, "Hy logits tensor was empty");
        if *HY_LOGIT_VALIDATION_ENABLED {
            validate_finite_logits(&logits)?;
        }
        if !config.sampling {
            let logits =
                self.apply_token_penalties(logits, state, config, selection, &mut profile)?;
            let argmax_started = Instant::now();
            let selected = logits.argmax(0)?;
            trace_hy_transfer("d2h", "greedy_argmax_token_id", std::mem::size_of::<u32>());
            let token_id = selected.to_scalar::<u32>()?;
            if profile.enabled {
                profile.greedy_argmax += argmax_started.elapsed();
                profile.print();
            }
            let input = selected.unsqueeze(0)?.unsqueeze(0)?;
            return Ok((token_id, input));
        }
        if config.top_k == 0 {
            anyhow::bail!("Hy CUDA sampled selection requires top_k > 0; top_k=0 is unsupported");
        }
        anyhow::ensure!(
            config.top_k <= MAX_TOP_K,
            "Hy CUDA sampled selection supports top_k <= {MAX_TOP_K}"
        );
        let logits = self.apply_token_penalties(logits, state, config, selection, &mut profile)?;
        let logits = logits.to_dtype(DType::F32)?;
        let device = logits.device().clone();
        let candidate_started = Instant::now();
        let candidate_count = config.top_k.min(logits.elem_count());
        let vocab_size = u32::try_from(logits.elem_count())?;
        let positions = Tensor::arange(0u32, vocab_size, &device)?;
        let negative_infinity = Tensor::full(f32::NEG_INFINITY, logits.shape(), &device)?;
        let mut masked_logits = logits;
        let candidate_ids = Tensor::zeros(candidate_count, DType::U32, &device)?;
        let candidate_logits = Tensor::zeros(candidate_count, DType::F32, &device)?;
        for candidate_index in 0..candidate_count {
            let selected_id = masked_logits.argmax(0)?;
            let selected_id_1d = selected_id.unsqueeze(0)?;
            let selected_logit = masked_logits.index_select(&selected_id_1d, 0)?;
            candidate_ids.slice_set(&selected_id_1d, 0, candidate_index)?;
            candidate_logits.slice_set(&selected_logit, 0, candidate_index)?;
            let selected = positions.broadcast_eq(&selected_id)?;
            masked_logits = selected.where_cond(&negative_infinity, &masked_logits)?;
        }
        if profile.enabled {
            profile.sample_top_k += candidate_started.elapsed();
        }

        let math_started = Instant::now();
        let scaled_logits = candidate_logits.broadcast_div(&selection.temperature)?;
        let max_logit = scaled_logits.max_all()?;
        let centered_logits = scaled_logits.broadcast_sub(&max_logit)?;
        let weights = centered_logits.to_dtype(DType::F64)?.exp()?;
        let total = weights.sum_all()?;
        let mut probabilities = weights.broadcast_div(&total)?;
        if let Some(top_p) = &selection.top_p {
            let boundary =
                first_true_or_last_index(&probabilities.cumsum(0)?.broadcast_ge(top_p)?)?;
            let positions = Tensor::arange(0u32, candidate_count as u32, &device)?;
            let boundary = boundary.broadcast_as(positions.shape())?;
            let keep = positions.broadcast_le(&boundary)?.to_dtype(DType::F64)?;
            probabilities = probabilities.broadcast_mul(&keep)?;
            let retained_total = probabilities.sum_all()?;
            probabilities = probabilities.broadcast_div(&retained_total)?;
        }
        if profile.enabled {
            profile.sample_math += math_started.elapsed();
        }

        let select_started = Instant::now();
        let cumulative = probabilities.cumsum(0)?;
        let thresholds = selection
            .thresholds
            .as_ref()
            .context("sampled Hy selection has no device threshold buffer")?;
        anyhow::ensure!(
            selection_step < thresholds.elem_count(),
            "Hy sampling threshold index exceeded the prepared buffer"
        );
        let threshold = thresholds.narrow(0, selection_step, 1)?;
        let selected_position =
            first_true_or_last_index(&cumulative.broadcast_gt(&threshold)?)?.unsqueeze(0)?;
        let selected = candidate_ids.index_select(&selected_position, 0)?;
        trace_hy_transfer("d2h", "sample_token_id", std::mem::size_of::<u32>());
        let token_id = selected.squeeze(0)?.to_scalar::<u32>()?;
        if profile.enabled {
            profile.sample_select += select_started.elapsed();
            profile.print();
        }
        Ok((token_id, selected.unsqueeze(0)?))
    }

    pub(super) fn decode_token(
        &self,
        decoder: &mut HyDecoderState,
        token_id: u32,
    ) -> Result<Vec<u8>> {
        decode_token_delta(
            &self.tokenizer,
            &mut decoder.token_ids,
            &mut decoder.prefix,
            &mut decoder.prefix_index,
            token_id,
        )
    }

    pub(super) fn is_stop_token(&self, token_id: u32) -> bool {
        token_id == EOS_TOKEN_ID
    }
}
/// Hy 会话独占的 copy-on-write GPU 状态。
///
/// KV cache 和 token history 驻留在模型 device。Penalty counts 仅在当前
/// request 启用 penalty 时按需创建，避免默认路径增加无意义的 GPU scatter。
#[derive(Clone)]
pub(super) struct HyGenerationState {
    kv_cache: Vec<LayerCache>,
    first_turn: bool,
    token_history: Arc<Tensor>,
    penalty_counts: Option<Arc<Tensor>>,
}

impl HyGenerationState {
    pub(super) fn new(
        layer_count: usize,
        max_context_length: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(max_context_length > 0, "Hy context length must be positive");
        Ok(Self {
            kv_cache: (0..layer_count).map(|_| None).collect(),
            first_turn: true,
            token_history: Arc::new(Tensor::zeros(max_context_length, DType::U32, device)?),
            penalty_counts: None,
        })
    }

    fn writable_tensor(slot: &mut Arc<Tensor>) -> Result<&mut Tensor> {
        if Arc::strong_count(slot) > 1 {
            *slot = Arc::new((**slot).copy()?);
        }
        Ok(Arc::get_mut(slot).expect("unique Hy GPU tensor after copy-on-write"))
    }

    /// Prepare a device-side vocabulary count table only when penalties are used.
    pub(super) fn prepare_penalty_counts(
        &mut self,
        enabled: bool,
        position: usize,
        vocab_size: usize,
        device: &Device,
    ) -> Result<()> {
        if !enabled {
            self.penalty_counts = None;
            return Ok(());
        }
        if self.penalty_counts.is_some() {
            return Ok(());
        }
        anyhow::ensure!(
            position <= self.token_history.dim(0)?,
            "Hy penalty history position exceeds the model context length"
        );
        let counts = Tensor::zeros(vocab_size, DType::F32, device)?;
        if position > 0 {
            let history = self.token_history.narrow(0, 0, position)?;
            let ones = Tensor::ones(history.shape().clone(), DType::F32, device)?;
            counts.scatter_add_set(&history, &ones, 0)?;
        }
        self.penalty_counts = Some(Arc::new(counts));
        Ok(())
    }

    /// 将一段 device token IDs 写入 GPU history，并更新已启用的 counts。
    pub(super) fn record_token_ids(&mut self, ids: &Tensor, position: usize) -> Result<()> {
        let ids = ids.flatten_all()?.contiguous()?;
        let length = ids.elem_count();
        let end = position
            .checked_add(length)
            .context("Hy token history position overflowed")?;
        anyhow::ensure!(
            end <= self.token_history.dim(0)?,
            "Hy token history exceeds the model context length"
        );
        Self::writable_tensor(&mut self.token_history)?.slice_set(&ids, 0, position)?;
        if let Some(counts) = &mut self.penalty_counts {
            let ones = Tensor::ones(ids.shape().clone(), DType::F32, ids.device())?;
            Self::writable_tensor(counts)?.scatter_add_set(&ids, &ones, 0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HyGenerationState, SimpleRmsNorm, format_turn_prompt, fuse_quantized_rows};
    use candle_core::{
        Device, Module, Tensor,
        quantized::{GgmlDType, QMatMul, QTensor},
    };

    #[test]
    fn formats_first_turn_prompt_with_begin_marker() {
        assert_eq!(
            format_turn_prompt("", "hello", true),
            "<｜hy_begin▁of▁sentence｜><｜hy_User｜>hello<｜hy_Assistant｜>"
        );
    }

    #[test]
    fn formats_first_turn_prompt_with_system_role() {
        assert_eq!(
            format_turn_prompt("system", "hello", true),
            "<｜hy_begin▁of▁sentence｜>system<｜hy_place▁holder▁no▁3｜><｜hy_User｜>hello<｜hy_Assistant｜>"
        );
    }

    #[test]
    fn formats_followup_prompt_without_system_or_begin_marker() {
        assert_eq!(
            format_turn_prompt("ignored", "hello", false),
            "<｜hy_User｜>hello<｜hy_Assistant｜>"
        );
    }

    #[test]
    fn first_true_or_last_index_has_deterministic_fallback() -> anyhow::Result<()> {
        let mask = Tensor::new(&[0u8, 0, 1, 1], &Device::Cpu)?;
        let index = super::first_true_or_last_index(&mask)?.to_scalar::<u32>()?;
        assert_eq!(index, 2);

        let no_match = Tensor::new(&[0u8, 0, 0, 0], &Device::Cpu)?;
        let index = super::first_true_or_last_index(&no_match)?.to_scalar::<u32>()?;
        assert_eq!(index, 3);
        Ok(())
    }

    #[test]
    fn gpu_token_history_and_penalty_counts_are_copy_on_write() -> anyhow::Result<()> {
        let mut state = HyGenerationState::new(0, 8, &Device::Cpu)?;
        state.prepare_penalty_counts(true, 0, 16, &Device::Cpu)?;
        let ids = Tensor::new(&[2u32, 2, 5], &Device::Cpu)?;
        state.record_token_ids(&ids, 1)?;

        let history = state.token_history.flatten_all()?.to_vec1::<u32>()?;
        assert_eq!(history, vec![0, 2, 2, 5, 0, 0, 0, 0]);
        let counts = state.penalty_counts.as_ref().unwrap().to_vec1::<f32>()?;
        assert_eq!(counts[2], 2.0);
        assert_eq!(counts[5], 1.0);

        let mut clone = state.clone();
        let next = Tensor::new(&[7u32], &Device::Cpu)?;
        clone.record_token_ids(&next, 4)?;
        assert_eq!(
            state.penalty_counts.as_ref().unwrap().to_vec1::<f32>()?[7],
            0.0
        );
        assert_eq!(
            clone.penalty_counts.as_ref().unwrap().to_vec1::<f32>()?[7],
            1.0
        );
        Ok(())
    }
    #[test]
    fn fused_quantized_rows_match_separate_cpu_matmuls() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let make_weight = |rows: usize, offset: f32| {
            Tensor::from_vec(
                (0..rows * 256)
                    .map(|index| ((index % 31) as f32 - 15.0 + offset) / 16.0)
                    .collect(),
                (rows, 256),
                &device,
            )
        };
        let weight_a = make_weight(2, 0.0)?;
        let weight_b = make_weight(3, 0.5)?;
        let quantized = [
            QTensor::quantize(&weight_a, GgmlDType::Q4K)?,
            QTensor::quantize(&weight_b, GgmlDType::Q4K)?,
        ];
        let fused = fuse_quantized_rows(&quantized)?;
        let [quantized_a, quantized_b] = quantized;
        let separate_a = QMatMul::from_qtensor(quantized_a)?;
        let separate_b = QMatMul::from_qtensor(quantized_b)?;
        let input = Tensor::from_vec(
            (0..256)
                .map(|index| (index as f32 - 128.0) / 64.0)
                .collect(),
            (1, 256),
            &device,
        )?;

        let fused_output = fused.forward(&input)?.to_vec2::<f32>()?;
        let separate_a = separate_a.forward(&input)?.to_vec2::<f32>()?;
        let separate_b = separate_b.forward(&input)?.to_vec2::<f32>()?;
        let expected = [separate_a[0].as_slice(), separate_b[0].as_slice()].concat();
        assert_eq!(fused_output[0].len(), expected.len());
        for (actual, expected) in fused_output[0].iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }
    #[test]
    fn fused_rms_norm_matches_reference_formula() -> candle_core::Result<()> {
        let device = candle_core::Device::Cpu;
        let xs = candle_core::Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, //
                -2.0, -1.0, 0.5, 1.5,
            ],
            (2, 4),
            &device,
        )?;
        let weight = candle_core::Tensor::from_vec(vec![0.25f32, 0.5, 1.0, 2.0], (4,), &device)?;
        let norm = SimpleRmsNorm::new(weight, 1e-6);

        let actual: Vec<f32> = norm.forward(&xs)?.flatten_all()?.to_vec1()?;
        let expected = reference_rms_norm(
            &[1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 0.5, 1.5],
            &[0.25, 0.5, 1.0, 2.0],
            1e-6,
            4,
        );

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    fn reference_rms_norm(xs: &[f32], weight: &[f32], eps: f32, hidden: usize) -> Vec<f32> {
        xs.chunks(hidden)
            .flat_map(|row| {
                let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / hidden as f32;
                let scale = 1.0 / (mean_sq + eps).sqrt();
                row.iter()
                    .zip(weight.iter())
                    .map(|(value, weight)| value * scale * weight)
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
