//! PP-OCRv6 RepLKFPN detection neck (tiny/small tiers).
//!
//! Structural reference: PaddleOCR `db_fpn.py` `RepLKFPN`.  The exported
//! weights carry the dilated reparameterized depthwise convolutions already
//! fused (`depthwise_convolution.weight` + `bias`), followed by a 1x1
//! pointwise projection and an SE gate per level.

use super::super::common::{Conv2dLayer, se_gate};
use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::VarBuilder;

/// One RepLKFPN level: fused DW + 1x1 projection + residual SE gate.
#[derive(Clone, Debug)]
struct InputLayer {
    depthwise: Conv2dLayer,
    pointwise: Conv2dLayer,
    se: NeckSeBlock,
}

impl InputLayer {
    fn load(
        vb: VarBuilder,
        channels: usize,
        kernel: usize,
        se_name: &str,
    ) -> Result<Self> {
        let padding = (kernel - 1) / 2;
        let depthwise = Conv2dLayer::load(
            vb.pp("depthwise_convolution"),
            channels,
            channels,
            (kernel, kernel),
            (1, 1),
            (padding, padding),
            channels,
            "weight",
            Some("bias"),
        )
        .context("load RepLKFPN level depthwise convolution")?;
        let pointwise = Conv2dLayer::load(
            vb.pp("pointwise_convolution"),
            channels,
            channels / 4,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            "weight",
            None,
        )
        .context("load RepLKFPN level pointwise convolution")?;
        let se = NeckSeBlock::load(vb.pp(se_name), channels / 4)?;
        Ok(Self {
            depthwise,
            pointwise,
            se,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.depthwise.forward(xs)?;
        let xs = self.pointwise.forward(&xs)?;
        let gate = self.se.forward(&xs)?;
        Ok(xs.add(&gate)?)
    }
}

/// One RepLKFPN 1x1 insert layer with its own SE gate.
#[derive(Clone, Debug)]
struct InsertLayer {
    in_conv: Conv2dLayer,
    se: NeckSeBlock,
}

impl InsertLayer {
    fn load(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let in_conv = Conv2dLayer::load(
            vb.pp("in_conv"),
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            "weight",
            None,
        )
        .context("load RepLKFPN insert convolution")?;
        let se = NeckSeBlock::load(vb.pp("squeeze_excitation_block"), out_channels)?;
        Ok(Self { in_conv, se })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.in_conv.forward(xs)?;
        let gate = self.se.forward(&xs)?;
        Ok(xs.add(&gate)?)
    }
}

/// Neck SE gate with `conv1`/`conv2` naming.
#[derive(Clone, Debug)]
struct NeckSeBlock {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl NeckSeBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let reduced = channels / 4;
        Ok(Self {
            conv1: Conv2dLayer::load(
                vb.pp("conv1"),
                channels,
                reduced,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                "weight",
                Some("bias"),
            )
            .context("load neck SE squeeze convolution")?,
            conv2: Conv2dLayer::load(
                vb.pp("conv2"),
                reduced,
                channels,
                (1, 1),
                (1, 1),
                (0, 0),
                1,
                "weight",
                Some("bias"),
            )
            .context("load neck SE excitation convolution")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        se_gate(xs, &self.conv1, &self.conv2, "RepLKFPN SE gating")
    }
}

/// RepLKFPN: four 1x1 inserts, top-down nearest fusion, four DW+1x1+SE
/// projections, and a nearest-upsampled four-level concatenation.
#[derive(Clone, Debug)]
pub(super) struct RepLkFpn {
    insert_conv: [InsertLayer; 4],
    input_conv: [InputLayer; 4],
}

impl RepLkFpn {
    pub(super) fn load(
        vb: VarBuilder,
        in_channels: [usize; 4],
        out_channels: usize,
        dilated_kernel_size: usize,
    ) -> Result<Self> {
        let insert_conv = [
            InsertLayer::load(vb.pp("insert_conv.0"), in_channels[0], out_channels)?,
            InsertLayer::load(vb.pp("insert_conv.1"), in_channels[1], out_channels)?,
            InsertLayer::load(vb.pp("insert_conv.2"), in_channels[2], out_channels)?,
            InsertLayer::load(vb.pp("insert_conv.3"), in_channels[3], out_channels)?,
        ];
        let input_conv = [
            InputLayer::load(
                vb.pp("input_conv.0"),
                out_channels,
                dilated_kernel_size,
                "squeeze_excitation_module",
            )?,
            InputLayer::load(
                vb.pp("input_conv.1"),
                out_channels,
                dilated_kernel_size,
                "squeeze_excitation_module",
            )?,
            InputLayer::load(
                vb.pp("input_conv.2"),
                out_channels,
                dilated_kernel_size,
                "squeeze_excitation_module",
            )?,
            InputLayer::load(
                vb.pp("input_conv.3"),
                out_channels,
                dilated_kernel_size,
                "squeeze_excitation_module",
            )?,
        ];
        Ok(Self {
            insert_conv,
            input_conv,
        })
    }

    pub(super) fn forward(&self, features: &[Tensor]) -> Result<Tensor> {
        ensure!(
            features.len() == 4,
            "RepLKFPN expects four backbone feature levels"
        );
        let in5 = self.insert_conv[3].forward(&features[3])?;
        let in4 = self.insert_conv[2].forward(&features[2])?;
        let in3 = self.insert_conv[1].forward(&features[1])?;
        let in2 = self.insert_conv[0].forward(&features[0])?;
        let out4 = in4.add(&upsample_2x(&in5)?)?;
        let out3 = in3.add(&upsample_2x(&out4)?)?;
        let out2 = in2.add(&upsample_2x(&out3)?)?;
        let p2 = self.input_conv[0].forward(&out2)?;
        let p3 = self.input_conv[1].forward(&out3)?;
        let p4 = self.input_conv[2].forward(&out4)?;
        let p5 = self.input_conv[3].forward(&in5)?;
        let target_h = p2.dim(2)?;
        let target_w = p2.dim(3)?;
        let p5 = p5.interpolate2d(target_h, target_w)?;
        let p4 = p4.interpolate2d(target_h, target_w)?;
        let p3 = p3.interpolate2d(target_h, target_w)?;
        Tensor::cat(&[&p5, &p4, &p3, &p2], 1).context("concat RepLKFPN pyramid")
    }
}

fn upsample_2x(xs: &Tensor) -> Result<Tensor> {
    let target_h = xs.dim(2)?.saturating_mul(2);
    let target_w = xs.dim(3)?.saturating_mul(2);
    Ok(xs.interpolate2d(target_h, target_w)?)
}
