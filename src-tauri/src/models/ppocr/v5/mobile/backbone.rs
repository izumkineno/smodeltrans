//! Mobile-family PPLCNetV3 backbone.

use super::super::super::common::{Conv2dLayer, se_gate};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::{BatchNorm, BatchNormConfig, ModuleT, VarBuilder, batch_norm};
/// PaddleOCR's `make_divisible`: round up to the nearest divisor with a guard
/// that prevents the result from falling below 90% of the requested value.
fn make_divisible(value: f32, divisor: usize) -> usize {
    let mut rounded = ((value + divisor as f32 / 2.0) as usize / divisor) * divisor;
    rounded = rounded.max(divisor);
    if (rounded as f32) < 0.9 * value {
        rounded += divisor;
    }
    rounded
}

/// Paddle hard-swish: `x * clip(x + 3.0, 0.0, 6.0) / 6.0`.
fn hswish(xs: &Tensor) -> Result<Tensor> {
    let three = Tensor::new(&[3.0f32], xs.device())?;
    let six = Tensor::new(&[6.0f32], xs.device())?;
    let clipped = xs.broadcast_add(&three)?.clamp(0.0f32, 6.0f32)?;
    xs.broadcast_mul(&clipped)?
        .broadcast_div(&six)
        .context("apply hardswish")
}
/// Paddle `LearnableAffineBlock`: `scale * x + bias` with scalar parameters.
#[derive(Clone, Debug)]
struct LearnableAffineBlock {
    scale: Tensor,
    bias: Tensor,
}

impl LearnableAffineBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get(1, "scale").context("load learnable affine scale")?,
            bias: vb.get(1, "bias").context("load learnable affine bias")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)?
            .broadcast_add(&self.bias)
            .context("apply learnable affine block")
    }
}

/// Paddle `ConvBNLayer`: convolution (no bias) followed by batch norm.
#[derive(Clone, Debug)]
struct ConvBn {
    convolution: Conv2dLayer,
    normalization: BatchNorm,
}

impl ConvBn {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        let padding = ((kernel.0 - 1) / 2, (kernel.1 - 1) / 2);
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
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.convolution.forward(xs)?;
        self.normalization
            .forward_t(&xs, false)
            .context("apply conv-bn")
    }
}
/// Paddle `LearnableRepLayer` in its exported (non-reparameterized) form:
/// identity BN + one 1x1 branch + four same-kernel branches, an affine `lab`,
/// and an optional hardswish with its own affine `act.lab`.
#[derive(Clone, Debug)]
struct LearnableRepLayer {
    identity: Option<BatchNorm>,
    conv_kxk: Vec<ConvBn>,
    conv_1x1: Option<ConvBn>,
    lab: LearnableAffineBlock,
    act_lab: LearnableAffineBlock,
    apply_act: bool,
}

impl LearnableRepLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: (usize, usize),
        groups: usize,
        apply_act: bool,
    ) -> Result<Self> {
        let identity = if out_channels == in_channels && stride == (1, 1) {
            Some(
                batch_norm(
                    in_channels,
                    BatchNormConfig {
                        eps: 1e-5,
                        ..Default::default()
                    },
                    vb.pp("identity"),
                )
                .context("load rep-layer identity batch norm")?,
            )
        } else {
            None
        };
        let conv_kxk = (0..4)
            .map(|index| {
                ConvBn::load(
                    vb.pp(format!("conv_symmetric.{index}")),
                    in_channels,
                    out_channels,
                    (kernel, kernel),
                    stride,
                    groups,
                )
                .with_context(|| format!("load rep-layer symmetric branch {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_1x1 = if kernel > 1 {
            Some(
                ConvBn::load(
                    vb.pp("conv_small_symmetric"),
                    in_channels,
                    out_channels,
                    (1, 1),
                    stride,
                    groups,
                )
                .context("load rep-layer small-symmetric branch")?,
            )
        } else {
            None
        };
        let lab = LearnableAffineBlock::load(vb.pp("lab"))?;
        let act_lab = LearnableAffineBlock::load(vb.pp("act").pp("lab"))?;
        Ok(Self {
            identity,
            conv_kxk,
            conv_1x1,
            lab,
            act_lab,
            apply_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = match &self.identity {
            Some(bn) => bn
                .forward_t(xs, false)
                .context("apply rep-layer identity")?,
            None => {
                // Pointwise layers change channels and carry no identity
                // branch; the accumulator must be zeroed at the branch
                // output shape, not the input shape.
                let first = self
                    .conv_kxk
                    .first()
                    .context("rep-layer has no symmetric branches")?
                    .forward(xs)?;
                first.zeros_like()?
            }
        };
        if let Some(conv) = &self.conv_1x1 {
            out = out.add(&conv.forward(xs)?)?;
        }
        for conv in &self.conv_kxk {
            out = out.add(&conv.forward(xs)?)?;
        }
        let out = self.lab.forward(&out)?;
        if self.apply_act {
            let out = hswish(&out)?;
            return self.act_lab.forward(&out);
        }
        Ok(out)
    }
}
/// Backbone squeeze-and-excitation module with `convolutions.{0,2}` naming.
#[derive(Clone, Debug)]
struct SELayer {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl SELayer {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let reduced = channels / 4;
        let conv1 = Conv2dLayer::load(
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
        .context("load SE squeeze convolution")?;
        let conv2 = Conv2dLayer::load(
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
        .context("load SE excitation convolution")?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        se_gate(xs, &self.conv1, &self.conv2, "backbone SE gating")
    }
}
/// One PPLCNetV3 residual stage: depthwise rep-layer, optional SE, pointwise
/// rep-layer.
#[derive(Clone, Debug)]
struct LcNetV3Block {
    depthwise: LearnableRepLayer,
    se: Option<SELayer>,
    pointwise: LearnableRepLayer,
}

impl LcNetV3Block {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        dw_kernel: usize,
        stride: (usize, usize),
        use_se: bool,
    ) -> Result<Self> {
        // Paddle only skips the activation on plain 2x2 downsampling blocks;
        // rectangular (2,1)/(1,2) recognizer strides still activate.
        let apply_act = stride != (2, 2);
        let depthwise = LearnableRepLayer::load(
            vb.pp("depthwise_convolution"),
            in_channels,
            in_channels,
            dw_kernel,
            stride,
            in_channels,
            apply_act,
        )
        .context("load LCNetV3 depthwise convolution")?;
        let se = if use_se {
            Some(SELayer::load(
                vb.pp("squeeze_excitation_module"),
                in_channels,
            )?)
        } else {
            None
        };
        let pointwise = LearnableRepLayer::load(
            vb.pp("pointwise_convolution"),
            in_channels,
            out_channels,
            1,
            (1, 1),
            1,
            true,
        )
        .context("load LCNetV3 pointwise convolution")?;
        Ok(Self {
            depthwise,
            se,
            pointwise,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.depthwise.forward(xs)?;
        let xs = match &self.se {
            Some(se) => se.forward(&xs)?,
            None => xs,
        };
        self.pointwise.forward(&xs)
    }
}
struct LcNetBlockSpec {
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    stride: (usize, usize),
    use_se: bool,
}

const fn lcnet_block(
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    stride: (usize, usize),
    use_se: bool,
) -> LcNetBlockSpec {
    LcNetBlockSpec {
        kernel,
        in_channels,
        out_channels,
        stride,
        use_se,
    }
}
const NET_CONFIG_DET: [&[LcNetBlockSpec]; 5] = [
    &[lcnet_block(3, 16, 32, (1, 1), false)],
    &[
        lcnet_block(3, 32, 64, (2, 2), false),
        lcnet_block(3, 64, 64, (1, 1), false),
    ],
    &[
        lcnet_block(3, 64, 128, (2, 2), false),
        lcnet_block(3, 128, 128, (1, 1), false),
    ],
    &[
        lcnet_block(3, 128, 256, (2, 2), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
    ],
    &[
        lcnet_block(5, 256, 512, (2, 2), true),
        lcnet_block(5, 512, 512, (1, 1), true),
        lcnet_block(5, 512, 512, (1, 1), false),
        lcnet_block(5, 512, 512, (1, 1), false),
    ],
];

// The exported recognizer weights carry a `[2, 1]` stride at index 2 of the
// final stage (its depthwise branch has no identity batch norm, and the
// shipped `config.json` `block_configs` lists `[2, 1]` there).  At the
// recognizer's 48-pixel input that leaves a 3-pixel feature height, exactly
// matching the `avg_pool2d([3, 2])` kernel.
const NET_CONFIG_REC: [&[LcNetBlockSpec]; 5] = [
    &[lcnet_block(3, 16, 32, (1, 1), false)],
    &[
        lcnet_block(3, 32, 64, (1, 1), false),
        lcnet_block(3, 64, 64, (1, 1), false),
    ],
    &[
        lcnet_block(3, 64, 128, (2, 1), false),
        lcnet_block(3, 128, 128, (1, 1), false),
    ],
    &[
        lcnet_block(3, 128, 256, (1, 2), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
        lcnet_block(5, 256, 256, (1, 1), false),
    ],
    &[
        lcnet_block(5, 256, 512, (2, 1), true),
        lcnet_block(5, 512, 512, (1, 1), true),
        lcnet_block(5, 512, 512, (2, 1), false),
        lcnet_block(5, 512, 512, (1, 1), false),
    ],
];
/// PPLCNetV3 backbone; the detector variant emits four level projections
/// through its `layer_list`, the recognizer variant emits the raw final stage.
#[derive(Clone, Debug)]
pub(super) struct PpLcNetV3 {
    stem: ConvBn,
    blocks: [Vec<LcNetV3Block>; 5],
    det_layers: Option<[Conv2dLayer; 4]>,
}

impl PpLcNetV3 {
    pub(super) fn load(model_vb: VarBuilder, det: bool, scale: f32) -> Result<Self> {
        // The encoder lives under `model.backbone.encoder`; the detector's
        // four level projections live beside the backbone under `model.layer.*`.
        let vb = model_vb.pp("backbone").pp("encoder");
        let divisor = 16;
        let stem = ConvBn::load(
            vb.pp("convolution"),
            3,
            make_divisible(16.0 * scale, divisor),
            (3, 3),
            (2, 2),
            1,
        )
        .context("load PPLCNetV3 stem")?;
        let config = if det {
            NET_CONFIG_DET.as_slice()
        } else {
            NET_CONFIG_REC.as_slice()
        };
        let mut stages = Vec::with_capacity(config.len());
        for (stage_index, stage) in config.iter().enumerate() {
            let mut stage_blocks = Vec::with_capacity(stage.len());
            for (layer_index, block) in stage.iter().enumerate() {
                let vb = vb
                    .pp(format!("blocks.{stage_index}"))
                    .pp(format!("layers.{layer_index}"));
                let in_channels = make_divisible(block.in_channels as f32 * scale, divisor);
                let out_channels = make_divisible(block.out_channels as f32 * scale, divisor);
                stage_blocks.push(LcNetV3Block::load(
                    vb,
                    in_channels,
                    out_channels,
                    block.kernel,
                    block.stride,
                    block.use_se,
                )?);
            }
            stages.push(stage_blocks);
        }
        let mut stages = stages.into_iter();
        let blocks = [
            stages.next().context("PPLCNetV3 stage 1 is missing")?,
            stages.next().context("PPLCNetV3 stage 2 is missing")?,
            stages.next().context("PPLCNetV3 stage 3 is missing")?,
            stages.next().context("PPLCNetV3 stage 4 is missing")?,
            stages.next().context("PPLCNetV3 stage 5 is missing")?,
        ];
        let det_layers = if det {
            const MV_CHANNELS: [usize; 4] = [16, 24, 56, 480];
            let backbone_out = [
                make_divisible(
                    config[1]
                        .last()
                        .context("PPLCNetV3 stage 2 is empty")?
                        .out_channels as f32
                        * scale,
                    divisor,
                ),
                make_divisible(
                    config[2]
                        .last()
                        .context("PPLCNetV3 stage 3 is empty")?
                        .out_channels as f32
                        * scale,
                    divisor,
                ),
                make_divisible(
                    config[3]
                        .last()
                        .context("PPLCNetV3 stage 4 is empty")?
                        .out_channels as f32
                        * scale,
                    divisor,
                ),
                make_divisible(
                    config[4]
                        .last()
                        .context("PPLCNetV3 stage 5 is empty")?
                        .out_channels as f32
                        * scale,
                    divisor,
                ),
            ];
            Some([
                Conv2dLayer::load(
                    model_vb.pp("layer.0"),
                    backbone_out[0],
                    (MV_CHANNELS[0] as f32 * scale) as usize,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    1,
                    "weight",
                    Some("bias"),
                )?,
                Conv2dLayer::load(
                    model_vb.pp("layer.1"),
                    backbone_out[1],
                    (MV_CHANNELS[1] as f32 * scale) as usize,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    1,
                    "weight",
                    Some("bias"),
                )?,
                Conv2dLayer::load(
                    model_vb.pp("layer.2"),
                    backbone_out[2],
                    (MV_CHANNELS[2] as f32 * scale) as usize,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    1,
                    "weight",
                    Some("bias"),
                )?,
                Conv2dLayer::load(
                    model_vb.pp("layer.3"),
                    backbone_out[3],
                    (MV_CHANNELS[3] as f32 * scale) as usize,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    1,
                    "weight",
                    Some("bias"),
                )?,
            ])
        } else {
            None
        };
        Ok(Self {
            stem,
            blocks,
            det_layers,
        })
    }

    pub(super) fn forward_det(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let xs = self.stem.forward(xs)?;
        let xs = forward_stage(&self.blocks[0], xs)?;
        let mut hidden = xs;
        let mut features = Vec::with_capacity(4);
        for stage in self.blocks.iter().skip(1) {
            hidden = forward_stage(stage, hidden)?;
            features.push(hidden.clone());
        }
        let layers = self
            .det_layers
            .as_ref()
            .context("PPLCNetV3 detector projections are missing")?;
        let mut outputs = Vec::with_capacity(features.len());
        for (index, feature) in features.into_iter().enumerate() {
            outputs.push(layers[index].forward(&feature)?);
        }
        Ok(outputs)
    }

    pub(super) fn forward_rec(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.stem.forward(xs)?;
        let mut hidden = xs;
        for stage in &self.blocks {
            hidden = forward_stage(stage, hidden)?;
        }
        Ok(hidden)
    }
}

fn forward_stage(stage: &[LcNetV3Block], xs: Tensor) -> Result<Tensor> {
    let mut hidden = xs;
    for block in stage {
        hidden = block.forward(&hidden)?;
    }
    Ok(hidden)
}
