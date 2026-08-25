//! Server-family HGNetV2 backbone.

use super::super::super::common::{Activation, ConvNormAct};
use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
const HG_STAGE_IN: [usize; 4] = [48, 128, 512, 1024];
const HG_STAGE_MID: [usize; 4] = [48, 96, 192, 384];
const HG_STAGE_OUT: [usize; 4] = [128, 512, 1024, 2048];
const HG_STAGE_BLOCKS: [usize; 4] = [1, 1, 3, 1];
const HG_STAGE_LAYERS: [usize; 4] = [6, 6, 6, 6];
const HG_STAGE_KERNELS: [usize; 4] = [3, 3, 5, 5];
const HG_LIGHT_BLOCK: [bool; 4] = [false, false, true, true];

#[derive(Clone, Debug)]
struct HgEmbeddings {
    stem1: ConvNormAct,
    stem2a: ConvNormAct,
    stem2b: ConvNormAct,
    stem3: ConvNormAct,
    stem4: ConvNormAct,
}

impl HgEmbeddings {
    fn load(vb: VarBuilder, recognizer: bool) -> Result<Self> {
        let stem3_stride = if recognizer { (1, 1) } else { (2, 2) };
        // The nested HGNetV2 config keeps its default ReLU; the recognizer's
        // outer SiLU setting applies to the SVTR head, not this backbone.
        let activation = Activation::Relu;
        let stem1 = ConvNormAct::load(
            vb.pp("stem1"),
            3,
            32,
            (3, 3),
            (2, 2),
            (1, 1),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 embedder stem1")?;
        let stem2a = ConvNormAct::load(
            vb.pp("stem2a"),
            32,
            16,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 embedder stem2a")?;
        let stem2b = ConvNormAct::load(
            vb.pp("stem2b"),
            16,
            32,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 embedder stem2b")?;
        let stem3 = ConvNormAct::load(
            vb.pp("stem3"),
            64,
            32,
            (3, 3),
            stem3_stride,
            (1, 1),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 embedder stem3")?;
        let stem4 = ConvNormAct::load(
            vb.pp("stem4"),
            32,
            48,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 embedder stem4")?;
        Ok(Self {
            stem1,
            stem2a,
            stem2b,
            stem3,
            stem4,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        if pixel_values.dim(1)? != 3 {
            candle_core::bail!(
                "HGNetV2 expects three input channels, received {:?}",
                pixel_values.dims()
            );
        }
        let stem1_output = self.stem1.forward(pixel_values)?;
        let embedding = stem1_output
            .pad_with_zeros(2, 0, 1)?
            .pad_with_zeros(3, 0, 1)?;
        let pooled_emb = embedding.max_pool2d_with_stride((2, 2), (1, 1))?;
        let emb_stem_2a = self.stem2a.forward(&embedding)?;
        let emb_stem_2a = emb_stem_2a
            .pad_with_zeros(2, 0, 1)?
            .pad_with_zeros(3, 0, 1)?;
        let emb_stem_2a = self.stem2b.forward(&emb_stem_2a)?;
        let embedding = Tensor::cat(&[&pooled_emb, &emb_stem_2a], 1)?;
        let embedding = self.stem3.forward(&embedding)?;
        self.stem4.forward(&embedding)
    }
}

#[derive(Clone, Debug)]
struct HgLightLayer {
    conv1: ConvNormAct,
    conv2: ConvNormAct,
}

impl HgLightLayer {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        activation: Activation,
    ) -> Result<Self> {
        let conv1 = ConvNormAct::load(
            vb.pp("conv1"),
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            Activation::None,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 light block pointwise convolution")?;
        let conv2 = ConvNormAct::load(
            vb.pp("conv2"),
            out_channels,
            out_channels,
            (kernel, kernel),
            (1, 1),
            (kernel / 2, kernel / 2),
            out_channels,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 light block depthwise convolution")?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        self.conv2.forward(&xs)
    }
}

#[derive(Clone, Debug)]
enum HgLayer {
    Standard(ConvNormAct),
    Light(HgLightLayer),
}

impl HgLayer {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        light: bool,
        activation: Activation,
    ) -> Result<Self> {
        if light {
            Ok(Self::Light(HgLightLayer::load(
                vb,
                in_channels,
                out_channels,
                kernel,
                activation,
            )?))
        } else {
            Ok(Self::Standard(ConvNormAct::load(
                vb,
                in_channels,
                out_channels,
                (kernel, kernel),
                (1, 1),
                (kernel / 2, kernel / 2),
                1,
                activation,
                false,
                "convolution.weight",
                "convolution.bias",
                "normalization",
            )?))
        }
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Standard(layer) => layer.forward(xs),
            Self::Light(layer) => layer.forward(xs),
        }
    }
}

#[derive(Clone, Debug)]
struct HgBasicLayer {
    layers: Vec<HgLayer>,
    aggregation: [ConvNormAct; 2],
    residual: bool,
}

impl HgBasicLayer {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        middle_channels: usize,
        out_channels: usize,
        layer_num: usize,
        kernel: usize,
        residual: bool,
        light: bool,
        activation: Activation,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_num);
        for i in 0..layer_num {
            layers.push(
                HgLayer::load(
                    vb.pp("layers").pp(i.to_string()),
                    if i == 0 { in_channels } else { middle_channels },
                    middle_channels,
                    kernel,
                    light,
                    activation,
                )
                .with_context(|| format!("load HGNetV2 basic layer convolution {i}"))?,
            );
        }
        let total_channels = in_channels + layer_num * middle_channels;
        let aggregation0 = ConvNormAct::load(
            vb.pp("aggregation").pp("0"),
            total_channels,
            out_channels / 2,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 aggregation squeeze convolution")?;
        let aggregation1 = ConvNormAct::load(
            vb.pp("aggregation").pp("1"),
            out_channels / 2,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            1,
            activation,
            false,
            "convolution.weight",
            "convolution.bias",
            "normalization",
        )
        .context("load HGNetV2 aggregation excitation convolution")?;
        Ok(Self {
            layers,
            aggregation: [aggregation0, aggregation1],
            residual,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let identity = xs.clone();
        let mut outputs = vec![xs.clone()];
        let mut hidden_state = xs.clone();
        for layer in &self.layers {
            hidden_state = layer.forward(&hidden_state)?;
            outputs.push(hidden_state.clone());
        }
        let hidden_state = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?;
        let hidden_state =
            self.aggregation[1].forward(&self.aggregation[0].forward(&hidden_state)?)?;
        if self.residual {
            hidden_state.add(&identity)
        } else {
            Ok(hidden_state)
        }
    }
}

#[derive(Clone, Debug)]
struct HgStage {
    downsample: Option<ConvNormAct>,
    blocks: Vec<HgBasicLayer>,
}

impl HgStage {
    fn load(vb: VarBuilder, stage_index: usize, recognizer: bool) -> Result<Self> {
        // The nested HGNetV2 backbone uses ReLU for both detector and recognizer.
        let activation = Activation::Relu;
        let downsample = if recognizer || stage_index > 0 {
            let stride = if recognizer {
                [(2, 1), (1, 2), (2, 1), (2, 1)][stage_index]
            } else {
                (2, 2)
            };
            Some(
                ConvNormAct::load(
                    vb.pp("downsample"),
                    HG_STAGE_IN[stage_index],
                    HG_STAGE_IN[stage_index],
                    (3, 3),
                    stride,
                    (1, 1),
                    HG_STAGE_IN[stage_index],
                    Activation::None,
                    false,
                    "convolution.weight",
                    "convolution.bias",
                    "normalization",
                )
                .with_context(|| {
                    format!("load HGNetV2 stage {stage_index} depthwise downsample")
                })?,
            )
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(HG_STAGE_BLOCKS[stage_index]);
        for block_index in 0..HG_STAGE_BLOCKS[stage_index] {
            blocks.push(
                HgBasicLayer::load(
                    vb.pp("blocks").pp(block_index.to_string()),
                    if block_index == 0 {
                        HG_STAGE_IN[stage_index]
                    } else {
                        HG_STAGE_OUT[stage_index]
                    },
                    HG_STAGE_MID[stage_index],
                    HG_STAGE_OUT[stage_index],
                    HG_STAGE_LAYERS[stage_index],
                    HG_STAGE_KERNELS[stage_index],
                    block_index != 0,
                    HG_LIGHT_BLOCK[stage_index],
                    activation,
                )
                .with_context(|| format!("load HGNetV2 stage {stage_index} block {block_index}"))?,
            );
        }
        Ok(Self { downsample, blocks })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden_state = match &self.downsample {
            Some(layer) => layer.forward(xs)?,
            None => xs.clone(),
        };
        for block in &self.blocks {
            hidden_state = block.forward(&hidden_state)?;
        }
        Ok(hidden_state)
    }
}

#[derive(Clone, Debug)]
pub(super) struct HgBackbone {
    embedder: HgEmbeddings,
    stages: Vec<HgStage>,
}

impl HgBackbone {
    pub(super) fn load(vb: VarBuilder, recognizer: bool) -> Result<Self> {
        let embedder = HgEmbeddings::load(vb.pp("embedder"), recognizer)?;
        let mut stages = Vec::with_capacity(4);
        for stage_index in 0..4 {
            stages.push(HgStage::load(
                vb.pp("encoder").pp("stages").pp(stage_index.to_string()),
                stage_index,
                recognizer,
            )?);
        }
        Ok(Self { embedder, stages })
    }

    pub(super) fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Vec<Tensor>> {
        let mut hidden_state = self.embedder.forward(pixel_values)?;
        let mut feature_maps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            hidden_state = stage.forward(&hidden_state)?;
            feature_maps.push(hidden_state.clone());
        }
        Ok(feature_maps)
    }
}
