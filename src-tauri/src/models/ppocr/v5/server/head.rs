//! Server-family DB++ detection head.

use super::super::super::common::{
    Activation, Conv2dLayer, ConvNormAct, ConvTranspose2dLayer, ConvTransposeNormAct,
};
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder, ops};
#[derive(Clone, Debug)]
struct DetSegmentationHead {
    conv_down: ConvNormAct,
    conv_up: ConvTransposeNormAct,
    conv_final: ConvTranspose2dLayer,
}

impl DetSegmentationHead {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_down = ConvNormAct::load(
            vb.pp("conv_down"),
            256,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            1,
            Activation::Relu,
            false,
            "convolution.weight",
            "convolution.bias",
            "norm",
        )?;
        let conv_up =
            ConvTransposeNormAct::load(vb.pp("conv_up"), 64, 64, 2, 2, Activation::Relu, true)?;
        let conv_final = ConvTranspose2dLayer::load(
            vb.pp("conv_final"),
            64,
            1,
            2,
            2,
            0,
            0,
            1,
            "weight",
            Some("bias"),
        )?;
        Ok(Self {
            conv_down,
            conv_up,
            conv_final,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let xs = self.conv_down.forward(xs)?;
        let xs = self.conv_up.forward(&xs)?;
        let feature = xs.clone();
        let xs = ops::sigmoid(&self.conv_final.forward(&xs)?)?;
        Ok((xs, feature))
    }
}

#[derive(Clone, Debug)]
struct DetLocalModule {
    convolution_backbone: ConvNormAct,
    convolution_final: Conv2dLayer,
}

impl DetLocalModule {
    fn load(vb: VarBuilder) -> Result<Self> {
        let convolution_backbone = ConvNormAct::load(
            vb.pp("convolution_backbone"),
            65,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            1,
            Activation::Relu,
            false,
            "convolution.weight",
            "convolution.bias",
            "norm",
        )?;
        let convolution_final = Conv2dLayer::load(
            vb.pp("convolution_final"),
            64,
            1,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            "weight",
            Some("bias"),
        )?;
        Ok(Self {
            convolution_backbone,
            convolution_final,
        })
    }

    fn forward(&self, feature: &Tensor, init_map: &Tensor) -> candle_core::Result<Tensor> {
        let hidden_state = Tensor::cat(&[init_map, feature], 1)?;
        let hidden_state = self.convolution_backbone.forward(&hidden_state)?;
        self.convolution_final.forward(&hidden_state)
    }
}

#[derive(Clone, Debug)]
pub(super) struct DetHead {
    binarize_head: DetSegmentationHead,
    local_refinement_module: DetLocalModule,
}

impl DetHead {
    pub(super) fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            binarize_head: DetSegmentationHead::load(vb.pp("binarize_head"))?,
            local_refinement_module: DetLocalModule::load(vb.pp("local_refinement_module"))?,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (initial_map, feature) = self.binarize_head.forward(xs)?;
        let target_h = initial_map.dim(2)?;
        let target_w = initial_map.dim(3)?;
        let feature = feature.interpolate2d(target_h, target_w)?;
        let refined = ops::sigmoid(
            &self
                .local_refinement_module
                .forward(&feature, &initial_map)?,
        )?;
        let output = initial_map.add(&refined)?.affine(0.5, 0.0)?;
        Ok(output)
    }
}
