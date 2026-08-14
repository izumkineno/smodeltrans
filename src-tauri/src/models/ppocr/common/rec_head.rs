//! Shared SVTR/CTC recognition head used by both recognizer variants.

use super::layers::{Activation, ConvNormAct};
use anyhow::{Context, Result};
use candle_core::{D, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, ops};
#[derive(Clone, Debug)]
struct RecAttention {
    qkv: Linear,
    projection: Linear,
    heads: usize,
    head_dim: usize,
    scale: f32,
}

impl RecAttention {
    fn load(vb: VarBuilder, hidden_size: usize, heads: usize) -> Result<Self> {
        anyhow::ensure!(
            hidden_size % heads == 0,
            "recognizer attention hidden size must divide evenly by heads"
        );
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))
            .context("load recognizer SVTR qkv projection")?;
        let projection = linear(hidden_size, hidden_size, vb.pp("projection"))
            .context("load recognizer SVTR output projection")?;
        let head_dim = hidden_size / heads;
        Ok(Self {
            qkv,
            projection,
            heads,
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
        let attention = ops::softmax(&scores, D::Minus1)?;
        let output = attention
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden_size))?;
        self.projection.forward(&output)
    }
}

#[derive(Clone, Debug)]
struct RecMlp {
    fc1: Linear,
    fc2: Linear,
}

impl RecMlp {
    fn load(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            fc1: linear(hidden_size, intermediate_size, vb.pp("fc1"))
                .context("load recognizer SVTR MLP fc1")?,
            fc2: linear(intermediate_size, hidden_size, vb.pp("fc2"))
                .context("load recognizer SVTR MLP fc2")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.fc1.forward(xs)?.silu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
struct RecSvtrBlock {
    self_attn: RecAttention,
    layer_norm1: LayerNorm,
    mlp: RecMlp,
    layer_norm2: LayerNorm,
}

impl RecSvtrBlock {
    fn load(vb: VarBuilder, hidden_size: usize, heads: usize, eps: f64) -> Result<Self> {
        let layer_norm1 = layer_norm(hidden_size, eps, vb.pp("layer_norm1"))
            .context("load recognizer SVTR first layer norm")?;
        let self_attn = RecAttention::load(vb.pp("self_attn"), hidden_size, heads)?;
        let mlp = RecMlp::load(vb.pp("mlp"), hidden_size, hidden_size * 2)?;
        let layer_norm2 = layer_norm(hidden_size, eps, vb.pp("layer_norm2"))
            .context("load recognizer SVTR second layer norm")?;
        Ok(Self {
            self_attn,
            layer_norm1,
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
pub(crate) struct RecHead {
    conv_block: [ConvNormAct; 5],
    svtr_block: [RecSvtrBlock; 2],
    norm: LayerNorm,
    head: Linear,
}

impl RecHead {
    pub(crate) fn load(vb: VarBuilder, backbone_channels: usize) -> Result<Self> {
        let mid_channels = backbone_channels / 8;
        let conv_block = [
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("0"),
                backbone_channels,
                mid_channels,
                (1, 3),
                (1, 1),
                (0, 1),
                1,
                Activation::Silu,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?,
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("1"),
                mid_channels,
                120,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                Activation::Silu,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?,
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("2"),
                120,
                backbone_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                Activation::Silu,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?,
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("3"),
                2 * backbone_channels,
                mid_channels,
                (1, 3),
                (1, 1),
                (0, 1),
                1,
                Activation::Silu,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?,
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("4"),
                mid_channels,
                120,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                Activation::Silu,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?,
        ];
        let svtr_block = [
            RecSvtrBlock::load(vb.pp("encoder").pp("svtr_block").pp("0"), 120, 8, 1e-6)?,
            RecSvtrBlock::load(vb.pp("encoder").pp("svtr_block").pp("1"), 120, 8, 1e-6)?,
        ];
        let norm = layer_norm(120, 1e-6, vb.pp("encoder").pp("norm"))?;
        let head = linear(120, 18_385, vb.pp("head"))?;
        Ok(Self {
            conv_block,
            svtr_block,
            norm,
            head,
        })
    }
    pub(crate) fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.conv_block[0].forward(hidden_states)?;
        let hidden_states = self.conv_block[1].forward(&hidden_states)?;
        let (batch_size, channels, height, width) = hidden_states.dims4()?;
        let mut hidden_states = hidden_states.flatten_from(2)?.transpose(1, 2)?;
        for block in &self.svtr_block {
            hidden_states = block.forward(&hidden_states)?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;
        let hidden_states = hidden_states
            .reshape((batch_size, height, width, channels))?
            .permute((0, 3, 1, 2))?;
        let hidden_states = self.conv_block[2].forward(&hidden_states)?;
        let hidden_states = Tensor::cat(&[&residual, &hidden_states], 1)?;
        let hidden_states = self.conv_block[3].forward(&hidden_states)?;
        let hidden_states = self.conv_block[4].forward(&hidden_states)?;
        let hidden_states = hidden_states.squeeze(2)?.transpose(1, 2)?;
        self.head.forward(&hidden_states)
    }
}
