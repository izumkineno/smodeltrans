//! PP-OCRv6 recognition heads.
//!
//! - Tiny tier: reshape projection + two-layer CTC (`conv1d` DW + BN + `1x1`
//!   + BN + fc1 + fc2) operating on the pooled backbone feature.
//! - Small/medium tiers: `EncoderWithLightSVTR` (PaddleOCR `rnn.py`) followed
//!   by the CTC linear projection.

use super::backbone::hswish;
use super::super::common::{Activation, Conv2dLayer, ConvNormAct};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::{BatchNorm, BatchNormConfig, LayerNorm, Linear, Module, ModuleT, VarBuilder, batch_norm, layer_norm, linear};


/// Exported Conv1d wrapper: the tiny reshape-CTC head ships its depthwise
/// (and pointwise) convolutions as 3D `[out, in/groups, kernel]` tensors.
/// Candle's conv2d needs 4D weights, so the loaded tensor is unsqueezed on
/// the height axis and the width convolution runs with a `(1, kernel)` kernel.
#[derive(Clone, Debug)]
struct Conv1dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    out_channels: usize,
    kernel: usize,
    groups: usize,
}

impl Conv1dLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        groups: usize,
        bias_name: Option<&str>,
    ) -> Result<Self> {
        let weight = vb
            .get((out_channels, in_channels / groups, kernel), "weight")
            .with_context(|| format!("load convolution weight `{}`", vb.prefix()))?
            .unsqueeze(2)?;
        let bias = match bias_name {
            Some(name) => Some(vb.get(out_channels, name)?),
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            out_channels,
            kernel,
            groups,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, channels, height, width) = xs.dims4()?;
        let padding = (self.kernel - 1) / 2;
        let mut padded = xs.clone();
        if padding != 0 {
            padded = padded.pad_with_zeros(3, padding, padding)?;
        }
        let mut output = padded.conv2d(&self.weight, 0, 1, 1, self.groups)?;
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(&bias.reshape((1, self.out_channels, 1, 1))?)?;
        }
        if output.dim(3)? != width {
            candle_core::bail!(
                "Conv1d width mismatch: input {width} -> {}",
                output.dim(3)?
            );
        }
        let _ = (batch, channels, height);
        Ok(output)
    }
}

/// Tiny tier reshape-CTC head: `conv1` (DW 1x5) -> `norm1` -> ReLU ->
/// `conv2` (1x1) -> `norm2` -> ReLU -> transpose -> `fc1` -> ReLU -> `fc2`.
#[derive(Clone, Debug)]
pub(super) struct TinyCtcHead {
    conv1: Conv1dLayer,
    norm1: BatchNorm,
    conv2: Conv1dLayer,
    norm2: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl TinyCtcHead {
    pub(super) fn load(
        vb: VarBuilder,
        in_channels: usize,
        hidden_size: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let conv1 = Conv1dLayer::load(
            vb.pp("conv1"),
            in_channels,
            in_channels,
            5,
            in_channels,
            None,
        )
        .context("load tiny CTC head depthwise convolution")?;
        let norm1 = batch_norm(
            in_channels,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp("norm1"),
        )
        .context("load tiny CTC head first batch norm")?;
        let conv2 = Conv1dLayer::load(
            vb.pp("conv2"),
            in_channels,
            in_channels,
            1,
            1,
            None,
        )
        .context("load tiny CTC head pointwise convolution")?;
        let norm2 = batch_norm(
            in_channels,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp("norm2"),
        )
        .context("load tiny CTC head second batch norm")?;
        let fc1 = linear(in_channels, hidden_size, vb.pp("fc1"))
            .context("load tiny CTC head first projection")?;
        let fc2 = linear(hidden_size, out_channels, vb.pp("fc2"))
            .context("load tiny CTC head second projection")?;
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
            fc1,
            fc2,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = hswish(&self.norm1.forward_t(&xs, false)?)?;
        let xs = self.conv2.forward(&xs)?;
        let xs = hswish(&self.norm2.forward_t(&xs, false)?)?;
        let (batch, channels, _height, width) = xs.dims4()?;
        let xs = xs
            .squeeze(2)?
            .transpose(1, 2)?
            .reshape((batch, width, channels))?;
        let xs = self.fc1.forward(&xs)?;
        let out = self.fc2.forward(&xs)?;
        Ok(out)
    }
}

/// Light SVTR attention block (prenorm=False): LN -> attention -> add,
/// LN -> MLP -> add.
#[derive(Clone, Debug)]
struct SvtrBlock {
    layer_norm1: LayerNorm,
    self_attn: SvtrAttention,
    mlp: SvtrMlp,
    layer_norm2: LayerNorm,
}

impl SvtrBlock {
    fn load(vb: VarBuilder, hidden_size: usize, mlp_ratio: f32) -> Result<Self> {
        let layer_norm1 = layer_norm(hidden_size, 1e-5, vb.pp("layer_norm1"))
            .context("load LightSVTR first layer norm")?;
        let self_attn = SvtrAttention::load(vb.pp("self_attn"), hidden_size)?;
        let mlp = SvtrMlp::load(vb.pp("mlp"), hidden_size, mlp_ratio)?;
        let layer_norm2 = layer_norm(hidden_size, 1e-5, vb.pp("layer_norm2"))
            .context("load LightSVTR second layer norm")?;
        Ok(Self {
            layer_norm1,
            self_attn,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs.clone();
        let hidden_state = self.layer_norm1.forward(xs)?;
        let hidden_state = self.self_attn.forward(&hidden_state)?;
        let hidden_state = hidden_state.add(&residual)?;
        let residual = hidden_state.clone();
        let hidden_state = self.layer_norm2.forward(&hidden_state)?;
        let hidden_state = self.mlp.forward(&hidden_state)?;
        hidden_state.add(&residual)
    }
}

#[derive(Clone, Debug)]
struct SvtrAttention {
    qkv: Linear,
    projection: Linear,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl SvtrAttention {
    fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        const HEADS: usize = 8;
        anyhow::ensure!(
            hidden_size % HEADS == 0,
            "LightSVTR hidden size must divide evenly by {HEADS}"
        );
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))
            .context("load LightSVTR qkv projection")?;
        let projection = linear(hidden_size, hidden_size, vb.pp("projection"))
            .context("load LightSVTR output projection")?;
        let head_dim = hidden_size / HEADS;
        Ok(Self {
            qkv,
            projection,
            heads: HEADS,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, seq_len, hidden_size) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((batch, seq_len, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let query = qkv.narrow(0, 0, 1)?.squeeze(0)?.contiguous()?;
        let key = qkv.narrow(0, 1, 1)?.squeeze(0)?.contiguous()?;
        let value = qkv.narrow(0, 2, 1)?.squeeze(0)?.contiguous()?;
        let scale = Tensor::new(self.scale, xs.device())?;
        let scores = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .broadcast_mul(&scale)?;
        let attention = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let output = attention
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden_size))?;
        self.projection.forward(&output)
    }
}

#[derive(Clone, Debug)]
struct SvtrMlp {
    fc1: Linear,
    fc2: Linear,
}

impl SvtrMlp {
    fn load(vb: VarBuilder, hidden_size: usize, mlp_ratio: f32) -> Result<Self> {
        let intermediate = ((hidden_size as f32) * mlp_ratio).round() as usize;
        Ok(Self {
            fc1: linear(hidden_size, intermediate, vb.pp("fc1"))
                .context("load LightSVTR MLP fc1")?,
            fc2: linear(intermediate, hidden_size, vb.pp("fc2"))
                .context("load LightSVTR MLP fc2")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.fc1.forward(xs)?.silu()?;
        self.fc2.forward(&xs)
    }
}

/// `EncoderWithLightSVTR`: reduce + local DW + global attention with an
/// additive 1x1 skip, followed by the CTC linear projection.
#[derive(Clone, Debug)]
pub(super) struct LightSvtrHead {
    conv_reduce: ConvNormAct,
    local_conv: Conv2dLayer,
    local_norm: BatchNorm,
    svtr_block: Vec<SvtrBlock>,
    norm: LayerNorm,
    skip_conv: ConvNormAct,
    head: Linear,
}

impl LightSvtrHead {
    pub(super) fn load(
        vb: VarBuilder,
        in_channels: usize,
        hidden_size: usize,
        mlp_ratio: f32,
        depth: usize,
        local_kernel: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let conv_reduce = ConvNormAct::load(
            vb.pp("encoder").pp("conv_block").pp("1"),
            in_channels,
            hidden_size,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            Activation::Silu,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load LightSVTR reduce convolution")?;
        let skip_conv = ConvNormAct::load(
            vb.pp("encoder").pp("conv_block").pp("0"),
            in_channels,
            hidden_size,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            Activation::Silu,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load LightSVTR skip convolution")?;
        let local_conv = Conv2dLayer::load(
            vb.pp("encoder").pp("conv_block").pp("2"),
            hidden_size,
            hidden_size,
            (1, local_kernel),
            (1, 1),
            (0, local_kernel / 2),
            hidden_size,
            "convolution.weight",
            None,
        )
        .context("load LightSVTR local convolution")?;
        let local_norm = batch_norm(
            hidden_size,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp("encoder").pp("conv_block").pp("2").pp("normalization"),
        )
        .context("load LightSVTR local batch norm")?;
        let mut svtr_block = Vec::with_capacity(depth);
        for index in 0..depth {
            svtr_block.push(
                SvtrBlock::load(
                    vb.pp("encoder").pp("svtr_block").pp(index.to_string()),
                    hidden_size,
                    mlp_ratio,
                )
                .with_context(|| format!("load LightSVTR block {index}"))?,
            );
        }
        let norm = layer_norm(hidden_size, 1e-6, vb.pp("encoder").pp("norm"))
            .context("load LightSVTR output layer norm")?;
        let head = linear(hidden_size, out_channels, vb.pp("head"))
            .context("load LightSVTR CTC projection")?;
        Ok(Self {
            conv_reduce,
            local_conv,
            local_norm,
            svtr_block,
            norm,
            skip_conv,
            head,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let skip = self.skip_conv.forward(xs)?;
        let mut z = self.conv_reduce.forward(xs)?;
        let local = self.local_conv.forward(&z)?;
        let local = self.local_norm.forward_t(&local, false)?.silu()?;
        z = z.add(&local)?;
        let (batch, channels, height, width) = z.dims4()?;
        let mut z = z.flatten_from(2)?.transpose(1, 2)?;
        for block in &self.svtr_block {
            z = block.forward(&z)?;
        }
        let z = self.norm.forward(&z)?;
        let z = z
            .reshape((batch, height, width, channels))?
            .permute((0, 3, 1, 2))?;
        let z = z.add(&skip)?;
        let z = z.flatten_from(2)?.transpose(1, 2)?;
        let out = self.head.forward(&z)?;
        Ok(out)
    }
}
