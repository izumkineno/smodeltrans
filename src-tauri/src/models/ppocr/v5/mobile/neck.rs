//! Mobile-family RSEFPN detection neck.

use super::super::super::common::{Conv2dLayer, se_gate};
use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::VarBuilder;
/// Neck squeeze-and-excitation module with `conv1`/`conv2` naming.
#[derive(Clone, Debug)]
struct NeckSeBlock {
    conv1: Conv2dLayer,
    conv2: Conv2dLayer,
}

impl NeckSeBlock {
    fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let reduced = channels / 4;
        let conv1 = Conv2dLayer::load(
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
        .context("load neck SE squeeze convolution")?;
        let conv2 = Conv2dLayer::load(
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
        .context("load neck SE excitation convolution")?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        se_gate(xs, &self.conv1, &self.conv2, "neck SE gating")
    }
}
/// One RSEFPN level: convolution followed by a residual SE gate.
#[derive(Clone, Debug)]
struct RseLayer {
    in_conv: Conv2dLayer,
    se: NeckSeBlock,
}

impl RseLayer {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        let padding = (kernel - 1) / 2;
        let in_conv = Conv2dLayer::load(
            vb.pp("in_conv"),
            in_channels,
            out_channels,
            (kernel, kernel),
            (1, 1),
            (padding, padding),
            1,
            "weight",
            None,
        )
        .context("load RSE layer input convolution")?;
        let se = NeckSeBlock::load(vb.pp("squeeze_excitation_block"), out_channels)?;
        Ok(Self { in_conv, se })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.in_conv.forward(xs)?;
        x.add(&self.se.forward(&x)?)
            .context("apply RSE layer shortcut")
    }
}

/// RSEFPN neck: four 1x1 input projections, a top-down nearest fusion, four
/// 3x3 projections, and a nearest-upsampled four-level concatenation.
#[derive(Clone, Debug)]
pub(super) struct RseFpn {
    insert_conv: [RseLayer; 4],
    input_conv: [RseLayer; 4],
}

impl RseFpn {
    pub(super) fn load(
        vb: VarBuilder,
        in_channels: [usize; 4],
        out_channels: usize,
    ) -> Result<Self> {
        let quarter = out_channels / 4;
        let insert_conv = [
            RseLayer::load(vb.pp("insert_conv.0"), in_channels[0], out_channels, 1)?,
            RseLayer::load(vb.pp("insert_conv.1"), in_channels[1], out_channels, 1)?,
            RseLayer::load(vb.pp("insert_conv.2"), in_channels[2], out_channels, 1)?,
            RseLayer::load(vb.pp("insert_conv.3"), in_channels[3], out_channels, 1)?,
        ];
        let input_conv = [
            RseLayer::load(vb.pp("input_conv.0"), out_channels, quarter, 3)?,
            RseLayer::load(vb.pp("input_conv.1"), out_channels, quarter, 3)?,
            RseLayer::load(vb.pp("input_conv.2"), out_channels, quarter, 3)?,
            RseLayer::load(vb.pp("input_conv.3"), out_channels, quarter, 3)?,
        ];
        Ok(Self {
            insert_conv,
            input_conv,
        })
    }

    pub(super) fn forward(&self, features: &[Tensor]) -> Result<Tensor> {
        ensure!(
            features.len() == 4,
            "RSEFPN expects four backbone feature levels"
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
        Tensor::cat(&[&p5, &p4, &p3, &p2], 1).context("concat RSEFPN pyramid")
    }
}

fn upsample_2x(xs: &Tensor) -> Result<Tensor> {
    let target_h = xs.dim(2)?.saturating_mul(2);
    let target_w = xs.dim(3)?.saturating_mul(2);
    Ok(xs.interpolate2d(target_h, target_w)?)
}
