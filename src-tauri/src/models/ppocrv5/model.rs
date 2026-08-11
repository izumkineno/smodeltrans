//! Native Candle PP-OCRv5 server detector and recognizer graphs.
//!
//! Detector and recognizer loading, forward passes, and graph-specific
//! preprocessing remain inside this PP-OCRv5 family boundary.

use std::path::Path;

use super::assets::{GraphRole, PpOcrV5Assets};

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, LayerNorm, Linear, Module, ModuleT, VarBuilder, batch_norm,
    layer_norm, linear, ops,
};

/// Shape metadata accompanying a native detector probability map.
#[derive(Clone, Debug)]
pub struct DetectorOutput {
    /// PP-OCRv5's final sigmoid probability map, shaped `[batch, 1, height, width]`.
    pub probabilities: Tensor,
    /// Materialized output dimensions for callers that do not want to inspect a Tensor.
    pub shape: Vec<usize>,
}

impl DetectorOutput {
    fn new(probabilities: Tensor) -> Self {
        Self {
            shape: probabilities.dims().to_vec(),
            probabilities,
        }
    }

    /// Borrow the raw probability tensor without doing post-processing.
    pub fn tensor(&self) -> &Tensor {
        &self.probabilities
    }
}

/// Shape metadata accompanying native recognizer per-time-step logits.
#[derive(Clone, Debug)]
pub struct RecognizerOutput {
    /// PP-OCRv5's per-time-step CTC logits tensor, shaped `[batch, time, vocab]`.
    pub logits: Tensor,
    /// Materialized output dimensions for callers that do not want to inspect a Tensor.
    pub shape: Vec<usize>,
}

impl RecognizerOutput {
    fn new(logits: Tensor) -> Self {
        Self {
            shape: logits.dims().to_vec(),
            logits,
        }
    }

    /// Borrow the raw logits tensor without softmax, decoding, or CTC post-processing.
    pub fn tensor(&self) -> &Tensor {
        &self.logits
    }
}

#[derive(Clone, Copy, Debug)]
enum Activation {
    None,
    Relu,
    Silu,
}

impl Activation {
    fn apply(self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::None => Ok(xs.clone()),
            Self::Relu => xs.relu(),
            Self::Silu => xs.silu(),
        }
    }
}

/// A rectangular/grouped convolution wrapper.
///
/// Candle's pinned convolution API accepts scalar padding and stride values.
/// Most PP-OCRv5 layers use symmetric geometry and can therefore execute those
/// operations directly. Rectangular kernels with asymmetric padding or stride
/// retain the explicit compatibility path.
#[derive(Clone, Debug)]
struct Conv2dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    out_channels: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
}

impl Conv2dLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> Result<Self> {
        anyhow::ensure!(groups > 0, "convolution groups must be positive");
        anyhow::ensure!(
            in_channels % groups == 0,
            "convolution input channels {} are not divisible by groups {}",
            in_channels,
            groups
        );
        anyhow::ensure!(
            stride.0 > 0 && stride.1 > 0,
            "convolution strides must be positive"
        );
        anyhow::ensure!(
            kernel.0 > 0 && kernel.1 > 0,
            "convolution kernels must be positive"
        );

        let qualified_weight = qualified_key(&vb, weight_name);
        let weight = vb
            .get(
                (out_channels, in_channels / groups, kernel.0, kernel.1),
                weight_name,
            )
            .with_context(|| format!("load convolution weight `{qualified_weight}`"))?;
        let bias = match bias_name {
            Some(name) => {
                let qualified_bias = qualified_key(&vb, name);
                Some(
                    vb.get(out_channels, name)
                        .with_context(|| format!("load convolution bias `{qualified_bias}`"))?,
                )
            }
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            out_channels,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut output = if self.padding.0 == self.padding.1 && self.stride.0 == self.stride.1 {
            xs.conv2d(&self.weight, self.padding.0, self.stride.0, 1, self.groups)?
        } else {
            let mut padded = xs.clone();
            if self.padding.0 != 0 {
                padded = padded.pad_with_zeros(2, self.padding.0, self.padding.0)?;
            }
            if self.padding.1 != 0 {
                padded = padded.pad_with_zeros(3, self.padding.1, self.padding.1)?;
            }
            let mut output = padded.conv2d(&self.weight, 0, 1, 1, self.groups)?;
            if self.stride.0 != 1 {
                output = stride_select(&output, 2, self.stride.0)?;
            }
            if self.stride.1 != 1 {
                output = stride_select(&output, 3, self.stride.1)?;
            }
            output
        };
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(&bias.reshape((1, self.out_channels, 1, 1))?)?;
        }
        Ok(output)
    }
}

fn stride_select(xs: &Tensor, dim: usize, stride: usize) -> candle_core::Result<Tensor> {
    let len = xs.dim(dim)?;
    if stride == 1 {
        return Ok(xs.clone());
    }
    let indices = Tensor::arange_step(0u32, len as u32, stride as u32, xs.device())?;
    xs.index_select(&indices, dim)
}

#[derive(Clone, Debug)]
struct ConvTranspose2dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    out_channels: usize,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
}

impl ConvTranspose2dLayer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> Result<Self> {
        let qualified_weight = qualified_key(&vb, weight_name);
        let weight = vb
            .get((in_channels, out_channels, kernel, kernel), weight_name)
            .with_context(|| format!("load transposed-convolution weight `{qualified_weight}`"))?;
        let bias = match bias_name {
            Some(name) => {
                let qualified_bias = qualified_key(&vb, name);
                Some(vb.get(out_channels, name).with_context(|| {
                    format!("load transposed-convolution bias `{qualified_bias}`")
                })?)
            }
            None => None,
        };
        Ok(Self {
            weight,
            bias,
            out_channels,
            padding,
            output_padding,
            stride,
            dilation,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut output = xs.conv_transpose2d(
            &self.weight,
            self.padding,
            self.output_padding,
            self.stride,
            self.dilation,
        )?;
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(&bias.reshape((1, self.out_channels, 1, 1))?)?;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct ConvNormAct {
    convolution: Conv2dLayer,
    normalization: BatchNorm,
    activation: Activation,
}

impl ConvNormAct {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        activation: Activation,
        convolution_bias: bool,
        convolution_weight_name: &str,
        convolution_bias_name: &str,
        normalization_name: &str,
    ) -> Result<Self> {
        let convolution = Conv2dLayer::load(
            vb.clone(),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            groups,
            convolution_weight_name,
            convolution_bias.then_some(convolution_bias_name),
        )?;
        let normalization = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp(normalization_name),
        )
        .with_context(|| {
            format!(
                "load batch normalization `{}`",
                qualified_key(&vb, normalization_name)
            )
        })?;
        Ok(Self {
            convolution,
            normalization,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.convolution.forward(xs)?;
        let xs = self.normalization.forward_t(&xs, false)?;
        self.activation.apply(&xs)
    }
}

#[derive(Clone, Debug)]
struct ConvTransposeNormAct {
    convolution: ConvTranspose2dLayer,
    normalization: BatchNorm,
    activation: Activation,
}

impl ConvTransposeNormAct {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        activation: Activation,
        convolution_bias: bool,
    ) -> Result<Self> {
        let convolution = ConvTranspose2dLayer::load(
            vb.clone(),
            in_channels,
            out_channels,
            kernel,
            stride,
            0,
            0,
            1,
            "convolution.weight",
            convolution_bias.then_some("convolution.bias"),
        )?;
        let normalization = batch_norm(
            out_channels,
            BatchNormConfig {
                eps: 1e-5,
                ..Default::default()
            },
            vb.pp("norm"),
        )
        .with_context(|| {
            format!(
                "load transposed-convolution batch normalization `{}`",
                qualified_key(&vb, "norm")
            )
        })?;
        Ok(Self {
            convolution,
            normalization,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.convolution.forward(xs)?;
        let xs = self.normalization.forward_t(&xs, false)?;
        self.activation.apply(&xs)
    }
}

fn qualified_key(vb: &VarBuilder, name: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}

fn load_mmaped_weights(assets: &PpOcrV5Assets, device: &Device) -> Result<VarBuilder<'static>> {
    anyhow::ensure!(
        assets.weights.is_file(),
        "{} weights disappeared before native load: {}",
        assets.role.as_str(),
        assets.weights.display()
    );
    let paths = [assets.weights.clone()];
    // SAFETY: Candle's mmap loader owns the mapping and only reads the immutable
    // safetensors file.  Asset discovery and the file check above establish the
    // required local path checks before this call.
    unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, device) }
        .with_context(|| format!("mmap {} safetensors", assets.role.as_str()))
}

fn image_batch(input: &Tensor) -> Result<Tensor> {
    let input = input
        .to_dtype(DType::F32)
        .context("convert PP-OCRv5 input tensor to F32")?;
    let input = match input.rank() {
        3 => input.unsqueeze(0),
        4 => Ok(input),
        rank => anyhow::bail!(
            "PP-OCRv5 expects a CHW or NCHW tensor, received rank {rank} with shape {:?}",
            input.dims()
        ),
    }?;
    anyhow::ensure!(
        input.dim(1)? == 3,
        "PP-OCRv5 expects three input channels, received shape {:?}",
        input.dims()
    );
    Ok(input)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    #[test]
    fn hg_embedding_pads_before_same_max_pool() -> anyhow::Result<()> {
        let input = Tensor::from_vec(
            vec![-4.0_f32, -3.0, -2.0, -1.0, -5.0, -6.0],
            (1, 1, 2, 3),
            &Device::Cpu,
        )?;
        let padded = input.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        let output = padded.max_pool2d_with_stride((2, 2), (1, 1))?;
        assert_eq!(output.dims(), &[1, 1, 2, 3]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            vec![-1.0, -2.0, 0.0, 0.0, 0.0, 0.0]
        );
        Ok(())
    }
}

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
struct HgBackbone {
    embedder: HgEmbeddings,
    stages: Vec<HgStage>,
}

impl HgBackbone {
    fn load(vb: VarBuilder, recognizer: bool) -> Result<Self> {
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

    fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Vec<Tensor>> {
        let mut hidden_state = self.embedder.forward(pixel_values)?;
        let mut feature_maps = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            hidden_state = stage.forward(&hidden_state)?;
            feature_maps.push(hidden_state.clone());
        }
        Ok(feature_maps)
    }
}

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
struct DetNeck {
    input_channel_adjustment: Vec<Conv2dLayer>,
    input_feature_projection: Vec<Conv2dLayer>,
    path_aggregation_head: Vec<Conv2dLayer>,
    path_aggregation_lateral: Vec<Conv2dLayer>,
    intraclass_blocks: Vec<DetIntraclassBlock>,
}

impl DetNeck {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut input_channel_adjustment = Vec::with_capacity(4);
        let mut input_feature_projection = Vec::with_capacity(4);
        let mut path_aggregation_lateral = Vec::with_capacity(4);
        for (index, channels) in [128usize, 512, 1024, 2048].into_iter().enumerate() {
            input_channel_adjustment.push(
                Conv2dLayer::load(
                    vb.pp("input_channel_adjustment_convolution")
                        .pp(index.to_string()),
                    channels,
                    256,
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
                    256,
                    64,
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
                    64,
                    64,
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
                    64,
                    64,
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

    fn forward(&self, feature_maps: &[Tensor]) -> candle_core::Result<Tensor> {
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
struct DetHead {
    binarize_head: DetSegmentationHead,
    local_refinement_module: DetLocalModule,
}

impl DetHead {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            binarize_head: DetSegmentationHead::load(vb.pp("binarize_head"))?,
            local_refinement_module: DetLocalModule::load(vb.pp("local_refinement_module"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
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

/// Native PP-OCRv5 server detector.
///
/// `load` maps the exact local Transformers safetensors file into F32 Candle
/// tensors.  `forward` only computes the graph and returns the raw sigmoid map;
/// thresholding, contour extraction, and coordinate restoration remain outside
/// this module.  Numerical parity is intentionally not asserted here.
#[derive(Clone, Debug)]
pub struct PpOcrV5Detector {
    backbone: HgBackbone,
    neck: DetNeck,
    head: DetHead,
}

impl PpOcrV5Detector {
    /// Load the detector graph from a discovered local model tree.
    pub(crate) fn load(assets: &PpOcrV5Assets, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            assets.role == GraphRole::Detector,
            "detector loader received {} assets",
            assets.role.as_str()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = HgBackbone::load(vb.pp("model").pp("backbone"), false)
            .context("load native PP-OCRv5 detector HGNetV2 backbone")?;
        let neck = DetNeck::load(vb.pp("model").pp("neck"))
            .context("load native PP-OCRv5 detector neck")?;
        let head = DetHead::load(vb.pp("head")).context("load native PP-OCRv5 detector head")?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    /// Run the native detector on a normalized CHW or NCHW image tensor.
    pub fn forward(&self, input: &Tensor) -> Result<DetectorOutput> {
        let input = image_batch(input)?;
        let feature_maps = self
            .backbone
            .forward(&input)
            .context("forward native PP-OCRv5 detector backbone")?;
        let neck = self
            .neck
            .forward(&feature_maps)
            .context("forward native PP-OCRv5 detector neck")?;
        let probabilities = self
            .head
            .forward(&neck)
            .context("forward native PP-OCRv5 detector head")?;
        Ok(DetectorOutput::new(probabilities))
    }
}

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
struct RecHead {
    conv_block: [ConvNormAct; 5],
    svtr_block: [RecSvtrBlock; 2],
    norm: LayerNorm,
    head: Linear,
}

impl RecHead {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv_block = [
            ConvNormAct::load(
                vb.pp("encoder").pp("conv_block").pp("0"),
                2048,
                256,
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
                256,
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
                2048,
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
                4096,
                256,
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
                256,
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

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
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

/// Native PP-OCRv5 server recognizer.
///
/// `load` maps the exact local Transformers safetensors file into F32 Candle
/// tensors. `forward` returns per-time-step raw CTC logits and deliberately
/// leaves softmax, CTC collapse, and character decoding to the caller.
/// Numerical parity is still an explicit follow-up measurement, not an implicit claim.
#[derive(Clone, Debug)]
pub struct PpOcrV5Recognizer {
    backbone: HgBackbone,
    head: RecHead,
}

impl PpOcrV5Recognizer {
    /// Load the recognizer graph from a discovered local model tree.
    pub(crate) fn load(assets: &PpOcrV5Assets, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            assets.role == GraphRole::Recognizer,
            "recognizer loader received {} assets",
            assets.role.as_str()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = HgBackbone::load(vb.pp("model").pp("backbone"), true)
            .context("load native PP-OCRv5 recognizer HGNetV2 backbone")?;
        let head =
            RecHead::load(vb.pp("head")).context("load native PP-OCRv5 recognizer SVTR head")?;
        Ok(Self { backbone, head })
    }

    /// Run the native recognizer on a normalized CHW or NCHW crop tensor.
    pub fn forward(&self, input: &Tensor) -> Result<RecognizerOutput> {
        let input = image_batch(input)?;
        let feature_maps = self
            .backbone
            .forward(&input)
            .context("forward native PP-OCRv5 recognizer backbone")?;
        let last_feature = feature_maps
            .last()
            .context("recognizer HGNetV2 did not produce a final feature map")?;
        let pooled = last_feature
            .avg_pool2d((3, 2))
            .context("average-pool native PP-OCRv5 recognizer backbone feature")?;
        let logits = self
            .head
            .forward(&pooled)
            .context("forward native PP-OCRv5 recognizer SVTR head")?;
        Ok(RecognizerOutput::new(logits))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SharedPpOcrV5Core {
    detector: PpOcrV5Assets,
    recognizer: PpOcrV5Assets,
}

impl SharedPpOcrV5Core {
    /// Discover both native PP-OCRv5 model trees for one OCR run.
    pub(crate) fn from_local_model_dirs(
        detector_dir: impl AsRef<Path>,
        recognizer_dir: impl AsRef<Path>,
    ) -> Result<Self> {
        let detector = PpOcrV5Assets::preflight(GraphRole::Detector, detector_dir)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let recognizer = PpOcrV5Assets::preflight(GraphRole::Recognizer, recognizer_dir)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(Self {
            detector,
            recognizer,
        })
    }

    pub(crate) fn load_detector(&self, device: &Device) -> Result<PpOcrV5Detector> {
        PpOcrV5Detector::load(&self.detector, device)
    }

    pub(crate) fn load_recognizer(&self, device: &Device) -> Result<PpOcrV5Recognizer> {
        PpOcrV5Recognizer::load(&self.recognizer, device)
    }

    pub(crate) fn validate_assets(&self) -> Result<()> {
        self.detector
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        self.recognizer
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod conv_tests {
    use super::*;

    fn compatibility_forward(layer: &Conv2dLayer, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut padded = xs.clone();
        if layer.padding.0 != 0 {
            padded = padded.pad_with_zeros(2, layer.padding.0, layer.padding.0)?;
        }
        if layer.padding.1 != 0 {
            padded = padded.pad_with_zeros(3, layer.padding.1, layer.padding.1)?;
        }
        let mut output = padded.conv2d(&layer.weight, 0, 1, 1, layer.groups)?;
        if let Some(bias) = &layer.bias {
            output = output.broadcast_add(&bias.reshape((1, layer.out_channels, 1, 1))?)?;
        }
        if layer.stride.0 != 1 {
            output = stride_select(&output, 2, layer.stride.0)?;
        }
        if layer.stride.1 != 1 {
            output = stride_select(&output, 3, layer.stride.1)?;
        }
        Ok(output)
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor) -> candle_core::Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "value {index} differs: actual={actual}, expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn native_symmetric_convolution_matches_compatibility_path() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            (0..50)
                .map(|value| ((value % 11) as f32 - 5.0) / 7.0)
                .collect::<Vec<_>>(),
            (1, 2, 5, 5),
            &device,
        )?;
        let layer = Conv2dLayer {
            weight: Tensor::from_vec(
                (0..18)
                    .map(|value| ((value % 7) as f32 - 3.0) / 5.0)
                    .collect::<Vec<_>>(),
                (2, 1, 3, 3),
                &device,
            )?,
            bias: Some(Tensor::from_vec(vec![0.25_f32, -0.5], 2, &device)?),
            out_channels: 2,
            stride: (2, 2),
            padding: (1, 1),
            groups: 2,
        };

        let actual = layer.forward(&input)?;
        let expected = compatibility_forward(&layer, &input)?;
        assert_tensor_close(&actual, &expected)
    }

    #[test]
    fn asymmetric_convolution_keeps_compatibility_geometry() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            (0..30)
                .map(|value| ((value % 13) as f32 - 6.0) / 9.0)
                .collect::<Vec<_>>(),
            (1, 1, 5, 6),
            &device,
        )?;
        let layer = Conv2dLayer {
            weight: Tensor::from_vec(
                (0..9)
                    .map(|value| (value as f32 - 4.0) / 6.0)
                    .collect::<Vec<_>>(),
                (1, 1, 3, 3),
                &device,
            )?,
            bias: Some(Tensor::from_vec(vec![0.125_f32], 1, &device)?),
            out_channels: 1,
            stride: (2, 1),
            padding: (1, 0),
            groups: 1,
        };

        let actual = layer.forward(&input)?;
        let expected = compatibility_forward(&layer, &input)?;
        assert_tensor_close(&actual, &expected)
    }
}
