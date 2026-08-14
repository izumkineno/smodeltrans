//! Plain DB detection head shared by the v5 mobile and all v6 detectors.

use super::layers::{Activation, ConvNormAct, ConvTranspose2dLayer, ConvTransposeNormAct};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::{VarBuilder, ops};
/// The plain DB head: 3x3 down, two 2x2 stride-2 transposed convolutions, and
/// a final sigmoid at the input resolution.
#[derive(Clone, Debug)]
pub(crate) struct DbHead {
    conv_down: ConvNormAct,
    conv_up: ConvTransposeNormAct,
    conv_final: ConvTranspose2dLayer,
}

impl DbHead {
    pub(crate) fn load(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        let mid = in_channels / 4;
        let conv_down = ConvNormAct::load(
            vb.pp("conv_down"),
            in_channels,
            mid,
            (3, 3),
            (1, 1),
            (1, 1),
            1,
            Activation::Relu,
            false,
            "convolution.weight",
            "convolution.bias",
            "norm",
        )
        .context("load DB head down convolution")?;
        let conv_up =
            ConvTransposeNormAct::load(vb.pp("conv_up"), mid, mid, 2, 2, Activation::Relu, true)
                .context("load DB head up convolution")?;
        let conv_final = ConvTranspose2dLayer::load(
            vb.pp("conv_final"),
            mid,
            1,
            2,
            2,
            0,
            0,
            1,
            "weight",
            Some("bias"),
        )
        .context("load DB head final convolution")?;
        Ok(Self {
            conv_down,
            conv_up,
            conv_final,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_down.forward(xs)?;
        let xs = self.conv_up.forward(&xs)?;
        let xs = self.conv_final.forward(&xs)?;
        ops::sigmoid(&xs).context("sigmoid DB probability map")
    }
}
