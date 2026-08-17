//! PP-OCRv6 PPLCNetV4 backbone.
//!
//! Structural reference: PaddleOCR `ppocr/modeling/backbones/rec_lcnetv4.py`.
//! The shipped Transformers export fuses each `RepDWConv` (3x3 DW + 1x1 DW +
//! identity + post-sum BN) into a single depthwise convolution carrying
//! `weight` + `bias`; stride/channel-changing blocks keep the unfused
//! `convolution` + `normalization` form.  Channel mixers export as
//! `channel_conv1` (expand) and `channel_conv2` (compress).

use super::super::common::{Conv2dLayer, se_gate};
use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::{BatchNorm, BatchNormConfig, ModuleT, VarBuilder, batch_norm};
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
enum StemKind {
    Branch,
    Simple,
}

/// LCNetV4 channel mixer activation.
///
/// PaddleOCR's LCNetV4 block defaults to GELU.  The export-level
/// `hidden_act` describes the recognizer head (LightSVTR uses SiLU), not the
/// backbone; accepting it here corrupts every residual channel mixer.
#[derive(Clone, Copy, Debug)]
enum ChannelAct {
    Gelu,
    Silu,
    Hswish,
    Relu,
}

impl ChannelAct {
    fn from_config(backbone_config: &Value) -> Result<Self> {
        match backbone_config.get("hidden_act").and_then(Value::as_str) {
            None | Some("gelu") => Ok(Self::Gelu),
            Some("silu") | Some("swish") => Ok(Self::Silu),
            Some("hswish") | Some("hardswish") => Ok(Self::Hswish),
            Some("relu") => Ok(Self::Relu),
            Some(other) => anyhow::bail!("unsupported LCNetV4 hidden_act {other}"),
        }
    }

    fn apply(self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Silu => xs.silu(),
            Self::Hswish => hswish(xs),
            Self::Relu => xs.relu(),
        }
    }
}

/// One exported LCNetV4 block: token mixer, optional SE gate, and the
/// expand-act-compress channel mixer with an identity shortcut.
#[derive(Clone, Debug)]
struct LcNetV4Block {
    token_mixer: TokenMixer,
    se: Option<SeBlock>,
    expand: ConvBn,
    compress: ConvBn,
    residual: bool,
    act: ChannelAct,
}

impl LcNetV4Block {
    fn load(
        vb: VarBuilder,
        dw_size: usize,
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        use_se: bool,
        act: ChannelAct,
    ) -> Result<Self> {
        let residual = in_channels == out_channels && stride == (1, 1);
        let token_mixer = if residual {
            // Rep-fused depthwise convolution: single weight+bias tensor.
            let padding = (dw_size - 1) / 2;
            TokenMixer::Fused(Conv2dLayer::load(
                vb.pp("token_conv"),
                in_channels,
                in_channels,
                (dw_size, dw_size),
                (1, 1),
                (padding, padding),
                in_channels,
                "weight",
                Some("bias"),
            )?)
        } else {
            let padding = (dw_size - 1) / 2;
            TokenMixer::ConvBn(ConvBn::load(
                vb.pp("token_conv"),
                in_channels,
                in_channels,
                (dw_size, dw_size),
                stride,
                (padding, padding),
                in_channels,
            )?)
        };
        let se = if use_se {
            Some(SeBlock::load(
                vb.pp("token_squeeze_excitation"),
                in_channels,
            )?)
        } else {
            None
        };
        let hidden = in_channels
            .checked_mul(2)
            .context("LCNetV4 expand ratio overflow")?;
        let expand = ConvBn::load(
            vb.pp("channel_conv1"),
            in_channels,
            hidden,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
        )?;
        let compress = ConvBn::load(
            vb.pp("channel_conv2"),
            hidden,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
        )?;
        Ok(Self {
            token_mixer,
            se,
            expand,
            compress,
            residual,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.token_mixer.forward(xs)?;
        if let Some(se) = &self.se {
            xs = se.forward(&xs)?;
        }
        let hidden = self.expand.forward(&xs)?;
        let hidden = self.act.apply(&hidden)?;
        let hidden = self.compress.forward(&hidden)?;
        if self.residual {
            Ok(xs.add(&hidden)?)
        } else {
            Ok(hidden)
        }
    }
}

#[derive(Clone, Debug)]
enum TokenMixer {
    /// Rep-fused depthwise convolution (identity stride, same channels).
    Fused(Conv2dLayer),
    /// Plain depthwise conv + BN (stride or channel change).
    ConvBn(ConvBn),
}

impl TokenMixer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Fused(layer) => Ok(layer.forward(xs)?),
            Self::ConvBn(layer) => Ok(layer.forward(xs)?),
        }
    }
}

/// SE gate with `token_squeeze_excitation.convolutions.{0,2}` naming.
#[derive(Clone, Debug)]
struct SeBlock {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl SeBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let reduced = channels / 4;
        Ok(Self {
            conv1: Conv2dLayer::load(
                vb.pp("convolutions").pp("0"),
                channels,
                reduced,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                "weight",
                Some("bias"),
            )
            .context("load LCNetV4 SE squeeze convolution")?,
            conv2: Conv2dLayer::load(
                vb.pp("convolutions").pp("2"),
                reduced,
                channels,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                "weight",
                Some("bias"),
            )
            .context("load LCNetV4 SE excitation convolution")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        se_gate(xs, &self.conv1, &self.conv2, "LCNetV4 SE gating")
    }
}

/// Conv2D (no bias) followed by batch norm; `Conv2D_BN` in Paddle.
///
/// `same_pad` requests Paddle's asymmetric `"SAME"` padding used by the
/// 2x2 branch-stem convolutions: a single row/column on the right and bottom.
#[derive(Clone, Debug)]
struct ConvBn {
    convolution: Conv2dLayer,
    normalization: BatchNorm,
    same_pad: bool,
}

impl ConvBn {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        Self::load_with_same_pad(
            vb,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            groups,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_with_same_pad(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        same_pad: bool,
    ) -> Result<Self> {
        let convolution = Conv2dLayer::load(
            vb.clone(),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            groups,
            "convolution.weight",
            None,
        )?;
        let normalization = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp("normalization"),
        )?;
        Ok(Self {
            convolution,
            normalization,
            same_pad,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.same_pad {
            xs.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?
        } else {
            xs.clone()
        };
        let xs = self.convolution.forward(&xs)?;
        self.normalization
            .forward_t(&xs, false)
            .context("apply LCNetV4 conv-bn")
    }
}

#[derive(Clone, Debug)]
enum Stem {
    /// Multi-branch stem: stem1..stem4 + max pool, total stride 4.
    Branch {
        stem1: ConvBn,
        stem2a: ConvBn,
        stem2b: ConvBn,
        stem3: ConvBn,
        stem4: ConvBn,
    },
    /// Simple two-conv stem with GELU between, total stride 4.
    Simple { conv1: ConvBn, conv2: ConvBn },
}

impl Stem {
    fn load(vb: VarBuilder, kind: StemKind, mid: usize, out: usize) -> Result<Self> {
        match kind {
            StemKind::Branch => {
                let conv = |name: &str,
                            in_c: usize,
                            out_c: usize,
                            kernel: (usize, usize),
                            stride: (usize, usize)| {
                    ConvBn::load(
                        vb.pp(name),
                        in_c,
                        out_c,
                        kernel,
                        stride,
                        ((kernel.0 - 1) / 2, (kernel.1 - 1) / 2),
                        1,
                    )
                };
                Ok(Self::Branch {
                    stem1: conv("stem1", 3, mid, (3, 3), (2, 2)).context("load LCNetV4 stem1")?,
                    stem2a: ConvBn::load_with_same_pad(
                        vb.pp("stem2a"),
                        mid,
                        mid / 2,
                        (2, 2),
                        (1, 1),
                        (0, 0),
                        1,
                        true,
                    )
                    .context("load LCNetV4 stem2a")?,
                    stem2b: ConvBn::load_with_same_pad(
                        vb.pp("stem2b"),
                        mid / 2,
                        mid,
                        (2, 2),
                        (1, 1),
                        (0, 0),
                        1,
                        true,
                    )
                    .context("load LCNetV4 stem2b")?,
                    stem3: conv("stem3", mid * 2, mid, (3, 3), (2, 2))
                        .context("load LCNetV4 stem3")?,
                    stem4: conv("stem4", mid, out, (1, 1), (1, 1)).context("load LCNetV4 stem4")?,
                })
            }
            StemKind::Simple => {
                let conv1 = ConvBn::load(vb.pp("conv1"), 3, mid, (3, 3), (2, 2), (1, 1), 1)
                    .context("load LCNetV4 simple stem conv1")?;
                Ok(Self::Simple {
                    conv1,
                    conv2: ConvBn::load(vb.pp("conv2"), mid, out, (3, 3), (2, 2), (1, 1), 1)
                        .context("load LCNetV4 simple stem conv2")?,
                })
            }
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Branch {
                stem1,
                stem2a,
                stem2b,
                stem3,
                stem4,
            } => {
                // PaddleOCR's `StemBlock` uses `ConvBNAct` for every branch
                // convolution. The export retains conv + BN, so the ReLU is
                // an explicit part of the native graph.
                let xs = stem1.forward(xs)?.relu()?;
                let x2 = stem2a.forward(&xs)?.relu()?;
                let x2 = stem2b.forward(&x2)?.relu()?;
                let padded = xs.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
                let x1 = padded.max_pool2d_with_stride((2, 2), (1, 1))?;
                let xs = Tensor::cat(&[&x1, &x2], 1)?;
                let xs = stem3.forward(&xs)?.relu()?;
                Ok(stem4.forward(&xs)?.relu()?)
            }
            Self::Simple { conv1, conv2 } => {
                let raw = conv1.convolution.forward(xs)?;
                let xs = conv1.normalization.forward_t(&raw, false)?;
                let xs = xs.gelu()?;
                conv2.forward(&xs)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PpLcNetV4 {
    stem: Stem,
    stages: Vec<Vec<LcNetV4Block>>,
}
impl PpLcNetV4 {
    pub(super) fn load(vb: VarBuilder, backbone_config: &Value, det: bool) -> Result<Self> {
        let vb = vb.pp("encoder");
        let stem_channels = backbone_config
            .get("stem_channels")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .flatten()
            .filter(|values| values.len() == 3)
            .ok_or_else(|| {
                anyhow::anyhow!("backbone_config.stem_channels must be [3, mid, out]")
            })?;
        let mid = stem_channels[1];
        let out = stem_channels[2];
        let stem_type = backbone_config
            .get("stem_type")
            .and_then(Value::as_str)
            .unwrap_or(if det { "large" } else { "small" });
        let stem_kind = match stem_type {
            "large" | "branch" => StemKind::Branch,
            "small" | "simple" => StemKind::Simple,
            other => anyhow::bail!("unsupported LCNetV4 stem_type {other}"),
        };
        let stem =
            Stem::load(vb.pp("convolution"), stem_kind, mid, out).context("load LCNetV4 stem")?;
        let channel_act = ChannelAct::from_config(backbone_config)?;
        let block_configs = backbone_config
            .get("block_configs")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("backbone_config.block_configs is missing"))?;
        ensure!(
            block_configs.len() == 4,
            "LCNetV4 expects four stage block lists, found {}",
            block_configs.len()
        );
        let mut stages = Vec::with_capacity(4);
        for (stage_index, stage_value) in block_configs.iter().enumerate() {
            let stage_blocks = stage_value
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("stage {stage_index} block config is not a list"))?;
            let mut blocks = Vec::with_capacity(stage_blocks.len());
            for (block_index, block_value) in stage_blocks.iter().enumerate() {
                let entry = block_value
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("block config {stage_index}.{block_index}"))?;
                ensure!(
                    entry.len() == 5,
                    "LCNetV4 block config {stage_index}.{block_index} must have five fields"
                );
                let dw_size = entry[0]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("block {stage_index}.{block_index} dw_size"))?
                    as usize;
                let in_channels = entry[1]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("block {stage_index}.{block_index} in"))?
                    as usize;
                let out_channels = entry[2]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("block {stage_index}.{block_index} out"))?
                    as usize;
                let stride = parse_stride(&entry[3])
                    .with_context(|| format!("block {stage_index}.{block_index} stride"))?;
                let use_se = entry[4]
                    .as_bool()
                    .ok_or_else(|| anyhow::anyhow!("block {stage_index}.{block_index} use_se"))?;
                let vb = vb
                    .pp("blocks")
                    .pp(stage_index.to_string())
                    .pp("blocks")
                    .pp(block_index.to_string());
                blocks.push(
                    LcNetV4Block::load(
                        vb,
                        dw_size,
                        in_channels,
                        out_channels,
                        stride,
                        use_se,
                        channel_act,
                    )
                    .with_context(|| {
                        format!("load LCNetV4 stage {stage_index} block {block_index}")
                    })?,
                );
            }
            stages.push(blocks);
        }
        Ok(Self { stem, stages })
    }

    pub(super) fn forward_det(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let mut hidden = self.stem.forward(xs)?;
        let mut levels = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            hidden = forward_stage(stage, hidden)?;
            levels.push(hidden.clone());
        }
        Ok(levels)
    }

    pub(super) fn forward_rec(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.stem.forward(xs)?;
        for stage in &self.stages {
            hidden = forward_stage(stage, hidden)?;
        }
        ensure!(
            hidden.dim(2)? >= 3,
            "LCNetV4 recognizer feature height {} is below the [3, 2] pool kernel",
            hidden.dim(2)?
        );
        Ok(hidden.avg_pool2d((3, 2))?)
    }
}

fn parse_stride(value: &Value) -> Result<(usize, usize)> {
    if let Some(scalar) = value.as_u64() {
        let stride = scalar as usize;
        return Ok((stride, stride));
    }
    if let Some(pair) = value.as_array().filter(|pair| pair.len() == 2) {
        let stride = |index: usize| {
            pair[index]
                .as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("stride entry {index} is not an integer"))
        };
        return Ok((stride(0)?, stride(1)?));
    }
    anyhow::bail!("stride must be an integer or a [h, w] pair")
}

fn forward_stage(stage: &[LcNetV4Block], xs: Tensor) -> Result<Tensor> {
    let mut hidden = xs;
    for block in stage {
        hidden = block.forward(&hidden)?;
    }
    Ok(hidden)
}

/// Paddle hard-swish: `x * clip(x + 3.0, 0.0, 6.0) / 6.0`.
pub(super) fn hswish(xs: &Tensor) -> candle_core::Result<Tensor> {
    let three = Tensor::new(&[3.0f32], xs.device())?;
    let six = Tensor::new(&[6.0f32], xs.device())?;
    let clipped = xs.broadcast_add(&three)?.clamp(0.0f32, 6.0f32)?;
    Ok(xs.broadcast_mul(&clipped)?.broadcast_div(&six)?)
}
