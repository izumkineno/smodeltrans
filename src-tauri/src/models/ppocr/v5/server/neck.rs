//! Server-family DB++ detection neck.

use super::super::super::common::{Activation, Conv2dLayer, ConvNormAct};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};
#[derive(Clone, Debug)]
struct DetIntraclassBlock {
    conv_reduce_channel: Conv2dLayer,
    vertical_long: [Conv2dLayer; 3],
    horizontal_long: [Conv2dLayer; 3],
    symmetric: [Conv2dLayer; 3],
    conv_final: ConvNormAct,
}

impl DetIntraclassBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv = |name: &str, kernel: (usize, usize), padding: (usize, usize)| {
            Conv2dLayer::load(
                vb.pp(name),
                32,
                32,
                kernel,
                (1, 1),
                padding,
                1,
                "weight",
                Some("bias"),
            )
        };
        let conv_reduce_channel = Conv2dLayer::load(
            vb.pp("conv_reduce_channel"),
            64,
            32,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            "weight",
            Some("bias"),
        )?;
        let vertical_long = [
            conv("vertical_long_to_small_conv_longratio", (7, 1), (3, 0))?,
            conv("vertical_long_to_small_conv_midratio", (5, 1), (2, 0))?,
            conv("vertical_long_to_small_conv_shortratio", (3, 1), (1, 0))?,
        ];
        let horizontal_long = [
            conv("horizontal_small_to_long_conv_longratio", (1, 7), (0, 3))?,
            conv("horizontal_small_to_long_conv_midratio", (1, 5), (0, 2))?,
            conv("horizontal_small_to_long_conv_shortratio", (1, 3), (0, 1))?,
        ];
        let symmetric = [
            conv("symmetric_conv_long_longratio", (7, 7), (3, 3))?,
            conv("symmetric_conv_long_midratio", (5, 5), (2, 2))?,
            conv("symmetric_conv_long_shortratio", (3, 3), (1, 1))?,
        ];
        let conv_final = ConvNormAct::load(
            vb.pp("conv_final"),
            32,
            64,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            Activation::Relu,
            true,
            "convolution.weight",
            "convolution.bias",
            "norm",
        )?;
        Ok(Self {
            conv_reduce_channel,
            vertical_long,
            horizontal_long,
            symmetric,
            conv_final,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs.clone();
        let hidden_state = self.conv_reduce_channel.forward(xs)?;
        let hidden_state = sum3(
            &self.symmetric[0].forward(&hidden_state)?,
            &self.vertical_long[0].forward(&hidden_state)?,
            &self.horizontal_long[0].forward(&hidden_state)?,
        )?;
        let hidden_state = sum3(
            &self.symmetric[1].forward(&hidden_state)?,
            &self.vertical_long[1].forward(&hidden_state)?,
            &self.horizontal_long[1].forward(&hidden_state)?,
        )?;
        let hidden_state = sum3(
            &self.symmetric[2].forward(&hidden_state)?,
            &self.vertical_long[2].forward(&hidden_state)?,
            &self.horizontal_long[2].forward(&hidden_state)?,
        )?;
        self.conv_final.forward(&hidden_state)?.add(&residual)
    }
}

fn sum3(a: &Tensor, b: &Tensor, c: &Tensor) -> candle_core::Result<Tensor> {
    (a + b)?.add(c)
}

#[derive(Clone, Debug)]
pub(crate) struct DetNeck {
    input_channel_adjustment: Vec<Conv2dLayer>,
    input_feature_projection: Vec<Conv2dLayer>,
    path_aggregation_head: Vec<Conv2dLayer>,
    path_aggregation_lateral: Vec<Conv2dLayer>,
    intraclass_blocks: Vec<DetIntraclassBlock>,
}

impl DetNeck {
    /// Load the DB++ path-aggregation neck (RepLKPAN export layout).
    ///
    /// `in_channels` are the four backbone levels and `out_channels` the neck
    /// width; the v5 server detector uses [128, 512, 1024, 2048] -> 256 while
    /// the v6 medium detector uses [128, 256, 512, 896] -> 256.
    pub(crate) fn load(
        vb: VarBuilder,
        in_channels: [usize; 4],
        out_channels: usize,
    ) -> Result<Self> {
        let quarter = out_channels / 4;
        let mut input_channel_adjustment = Vec::with_capacity(4);
        let mut input_feature_projection = Vec::with_capacity(4);
        let mut path_aggregation_lateral = Vec::with_capacity(4);
        for (index, channels) in in_channels.into_iter().enumerate() {
            input_channel_adjustment.push(
                Conv2dLayer::load(
                    vb.pp("input_channel_adjustment_convolution")
                        .pp(index.to_string()),
                    channels,
                    out_channels,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    1,
                    "weight",
                    None,
                )
                .with_context(|| format!("load detector neck channel adjustment {index}"))?,
            );
            input_feature_projection.push(
                Conv2dLayer::load(
                    vb.pp("input_feature_projection_convolution")
                        .pp(index.to_string()),
                    out_channels,
                    quarter,
                    (9, 9),
                    (1, 1),
                    (4, 4),
                    1,
                    "weight",
                    None,
                )
                .with_context(|| format!("load detector neck feature projection {index}"))?,
            );
            path_aggregation_lateral.push(
                Conv2dLayer::load(
                    vb.pp("path_aggregation_lateral_convolution")
                        .pp(index.to_string()),
                    quarter,
                    quarter,
                    (9, 9),
                    (1, 1),
                    (4, 4),
                    1,
                    "weight",
                    None,
                )
                .with_context(|| format!("load detector neck lateral convolution {index}"))?,
            );
        }
        let mut path_aggregation_head = Vec::with_capacity(3);
        for index in 0..3 {
            path_aggregation_head.push(
                Conv2dLayer::load(
                    vb.pp("path_aggregation_head_convolution")
                        .pp(index.to_string()),
                    quarter,
                    quarter,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    1,
                    "weight",
                    None,
                )
                .with_context(|| format!("load detector neck path aggregation head {index}"))?,
            );
        }
        let mut intraclass_blocks = Vec::with_capacity(4);
        for index in 0..4 {
            intraclass_blocks.push(
                DetIntraclassBlock::load(vb.pp("intraclass_blocks").pp(index.to_string()))
                    .with_context(|| format!("load detector neck intraclass block {index}"))?,
            );
        }
        Ok(Self {
            input_channel_adjustment,
            input_feature_projection,
            path_aggregation_head,
            path_aggregation_lateral,
            intraclass_blocks,
        })
    }

    pub(crate) fn forward(&self, feature_maps: &[Tensor]) -> candle_core::Result<Tensor> {
        if feature_maps.len() != 4 {
            candle_core::bail!("detector neck expects four HGNetV2 feature maps");
        }
        let channel_adjusted = self
            .input_channel_adjustment
            .iter()
            .zip(feature_maps)
            .map(|(layer, feature)| layer.forward(feature))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let mut top_down = vec![channel_adjusted[3].clone(); 4];
        for index in (0..3).rev() {
            let target_h = channel_adjusted[index].dim(2)?;
            let target_w = channel_adjusted[index].dim(3)?;
            let upsampled = top_down[index + 1].interpolate2d(target_h, target_w)?;
            top_down[index] = channel_adjusted[index].add(&upsampled)?;
        }

        let mut projected = Vec::with_capacity(4);
        for index in 0..4 {
            let source = if index < 3 {
                &top_down[index]
            } else {
                &channel_adjusted[3]
            };
            projected.push(self.input_feature_projection[index].forward(source)?);
        }

        let mut bottom_up = vec![projected[0].clone()];
        for index in 1..4 {
            let downsampled =
                self.path_aggregation_head[index - 1].forward(&bottom_up[index - 1])?;
            bottom_up.push(projected[index].add(&downsampled)?);
        }

        let mut lateral_refined = Vec::with_capacity(4);
        for index in 0..4 {
            let source = if index == 0 {
                &projected[0]
            } else {
                &bottom_up[index]
            };
            lateral_refined.push(self.path_aggregation_lateral[index].forward(source)?);
        }

        let mut refined = Vec::with_capacity(4);
        for (block, feature) in self.intraclass_blocks.iter().zip(lateral_refined) {
            refined.push(block.forward(&feature)?);
        }
        let target_h = refined[0].dim(2)?;
        let target_w = refined[0].dim(3)?;
        let mut upsampled = Vec::with_capacity(4);
        for (index, feature) in refined.iter().enumerate() {
            if [1usize, 2, 4, 8][index] == 1 {
                upsampled.push(feature.clone());
            } else {
                upsampled.push(feature.interpolate2d(target_h, target_w)?);
            }
        }
        upsampled.reverse();
        Tensor::cat(&upsampled.iter().collect::<Vec<_>>(), 1)
    }
}
