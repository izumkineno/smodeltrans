//! Native server-family detector and recognizer graphs.

use super::super::super::assets::{GraphRole, PpOcrAssets};
use super::super::super::common::{
    DetectorOutput, RecHead, RecognizerOutput, image_batch, load_mmaped_weights,
};
use super::{backbone::HgBackbone, head::DetHead, neck::DetNeck};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::Module;
/// Native PP-OCRv5 server detector.
///
/// `load` maps the exact local Transformers safetensors file into F32 Candle
/// tensors.  `forward` only computes the graph and returns the raw sigmoid map;
/// thresholding, contour extraction, and coordinate restoration remain outside
/// this module.  Numerical parity is intentionally not asserted here.
#[derive(Clone, Debug)]
pub struct PpOcrServerDetector {
    backbone: HgBackbone,
    neck: DetNeck,
    head: DetHead,
}

impl PpOcrServerDetector {
    /// Load the detector graph from a discovered local model tree.
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            assets.role == GraphRole::Detector,
            "detector loader received {} assets",
            assets.role.as_str()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = HgBackbone::load(vb.pp("model").pp("backbone"), false)
            .context("load native PP-OCRv5 detector HGNetV2 backbone")?;
        let neck = DetNeck::load(vb.pp("model").pp("neck"), [128, 512, 1024, 2048], 256)
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
/// Native PP-OCRv5 server recognizer.
///
/// `load` maps the exact local Transformers safetensors file into F32 Candle
/// tensors. `forward` returns per-time-step raw CTC logits and deliberately
/// leaves softmax, CTC collapse, and character decoding to the caller.
/// Numerical parity is still an explicit follow-up measurement, not an implicit claim.
#[derive(Clone, Debug)]
pub struct PpOcrServerRecognizer {
    backbone: HgBackbone,
    head: RecHead,
}

impl PpOcrServerRecognizer {
    /// Load the recognizer graph from a discovered local model tree.
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        anyhow::ensure!(
            assets.role == GraphRole::Recognizer,
            "recognizer loader received {} assets",
            assets.role.as_str()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = HgBackbone::load(vb.pp("model").pp("backbone"), true)
            .context("load native PP-OCRv5 recognizer HGNetV2 backbone")?;
        let head = RecHead::load(vb.pp("head"), 2048)
            .context("load native PP-OCRv5 recognizer SVTR head")?;
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
