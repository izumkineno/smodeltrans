//! Shared convolution, normalization, and weight-loading primitives.
//!
//! Both the server (HGNetV2) and mobile (PPLCNetV3) adapters build on these.

use super::super::assets::PpOcrAssets;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{BatchNorm, BatchNormConfig, Module, ModuleT, VarBuilder, batch_norm};
#[derive(Clone, Copy, Debug)]
pub(crate) enum Activation {
    None,
    Relu,
    Silu,
}

impl Activation {
    pub(crate) fn apply(self, xs: &Tensor) -> candle_core::Result<Tensor> {
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
pub(crate) struct Conv2dLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    out_channels: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
}

impl Conv2dLayer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
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

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
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
pub(crate) struct ConvTranspose2dLayer {
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
    pub(crate) fn load(
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

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
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
pub(crate) struct ConvNormAct {
    convolution: Conv2dLayer,
    normalization: BatchNorm,
    activation: Activation,
}

impl ConvNormAct {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
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

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.convolution.forward(xs)?;
        let xs = self.normalization.forward_t(&xs, false)?;
        self.activation.apply(&xs)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConvTransposeNormAct {
    convolution: ConvTranspose2dLayer,
    normalization: BatchNorm,
    activation: Activation,
}

impl ConvTransposeNormAct {
    pub(crate) fn load(
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

    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
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

pub(crate) fn load_mmaped_weights(
    assets: &PpOcrAssets,
    device: &Device,
) -> Result<VarBuilder<'static>> {
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
pub(crate) fn image_batch(input: &Tensor) -> Result<Tensor> {
    let input = input
        .to_dtype(DType::F32)
        .context("convert PP-OCR input tensor to F32")?;
    let input = match input.rank() {
        3 => input.unsqueeze(0),
        4 => Ok(input),
        rank => anyhow::bail!(
            "PP-OCR expects a CHW or NCHW tensor, received rank {rank} with shape {:?}",
            input.dims()
        ),
    }?;
    anyhow::ensure!(
        input.dim(1)? == 3,
        "PP-OCR expects three input channels, received shape {:?}",
        input.dims()
    );
    Ok(input)
}

/// Paddle hard-sigmoid: `clip(x / 6.0 + 0.5, 0.0, 1.0)`.
pub(crate) fn hardsigmoid(xs: &Tensor) -> Result<Tensor> {
    let six = Tensor::new(&[6.0f32], xs.device())?;
    let half = Tensor::new(&[0.5f32], xs.device())?;
    xs.broadcast_div(&six)?
        .broadcast_add(&half)?
        .clamp(0.0f32, 1.0f32)
        .context("apply hardsigmoid")
}

/// Shared SE dataflow: global average pool → squeeze → relu → excite →
/// hardsigmoid → elementwise gate on the identity.
pub(crate) fn se_gate(
    xs: &Tensor,
    conv1: &Conv2dLayer,
    conv2: &Conv2dLayer,
    context: &str,
) -> Result<Tensor> {
    let identity = xs.clone();
    let pooled = xs.mean((2, 3))?.unsqueeze(2)?.unsqueeze(3)?;
    let x = conv1.forward(&pooled)?;
    let x = x.relu()?;
    let x = conv2.forward(&x)?;
    let x = hardsigmoid(&x)?;
    identity
        .broadcast_mul(&x)
        .with_context(|| context.to_owned())
}
