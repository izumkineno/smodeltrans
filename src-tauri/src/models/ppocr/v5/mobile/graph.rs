//! Native mobile-family detector and recognizer graphs.

use super::super::super::assets::{GraphRole, PpOcrAssets, PpOcrVariant};
use super::super::super::common::{
    DbHead, DetectorOutput, RecHead, RecognizerOutput, image_batch, load_mmaped_weights,
};
use super::{backbone::PpLcNetV3, neck::RseFpn};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use candle_nn::Module;
/// Native PP-OCRv5 mobile detector.
#[derive(Clone, Debug)]
pub(crate) struct PpOcrMobileDetector {
    backbone: PpLcNetV3,
    neck: RseFpn,
    head: DbHead,
}

impl PpOcrMobileDetector {
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        ensure!(
            assets.role == GraphRole::Detector,
            "mobile detector loader received {} assets",
            assets.role.as_str()
        );
        ensure!(
            assets.variant == PpOcrVariant::V5Mobile,
            "mobile detector loader received {} assets",
            assets.variant.label()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = PpLcNetV3::load(vb.pp("model"), true, 0.75)
            .context("load native PP-OCRv5 mobile detector PPLCNetV3 backbone")?;
        // The scale-0.75 detector projects its last four stages to these
        // channels via the backbone's own layer list.
        let neck = RseFpn::load(vb.pp("model").pp("neck"), [12, 18, 42, 360], 96)
            .context("load native PP-OCRv5 mobile detector RSEFPN neck")?;
        let head = DbHead::load(vb.pp("head"), 96)
            .context("load native PP-OCRv5 mobile detector DB head")?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<DetectorOutput> {
        let input = image_batch(input)?;
        let features = self
            .backbone
            .forward_det(&input)
            .context("forward mobile detector backbone")?;
        let fuse = self
            .neck
            .forward(&features)
            .context("forward mobile detector neck")?;
        let probabilities = self
            .head
            .forward(&fuse)
            .context("forward mobile detector head")?;
        Ok(DetectorOutput::new(probabilities))
    }
}
/// Native PP-OCRv5 mobile recognizer: PPLCNetV3 backbone feeding the shared
/// SVTR/CTC head.
#[derive(Clone, Debug)]
pub(crate) struct PpOcrMobileRecognizer {
    backbone: PpLcNetV3,
    head: RecHead,
}

impl PpOcrMobileRecognizer {
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        ensure!(
            assets.role == GraphRole::Recognizer,
            "mobile recognizer loader received {} assets",
            assets.role.as_str()
        );
        ensure!(
            assets.variant == PpOcrVariant::V5Mobile,
            "mobile recognizer loader received {} assets",
            assets.variant.label()
        );
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = PpLcNetV3::load(vb.pp("model"), false, 0.95)
            .context("load native PP-OCRv5 mobile recognizer PPLCNetV3 backbone")?;
        let head = RecHead::load(vb.pp("head"), 480)
            .context("load native PP-OCRv5 mobile recognizer SVTR head")?;
        Ok(Self { backbone, head })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<RecognizerOutput> {
        let input = image_batch(input)?;
        let feature = self
            .backbone
            .forward_rec(&input)
            .context("forward mobile recognizer backbone")?;
        // Both PP-OCRv5 recognizers keep the raw final backbone feature and
        // apply the exported `avg_pool2d([3, 2])` before the shared CTC head.
        let pooled = feature
            .avg_pool2d((3, 2))
            .context("average-pool mobile recognizer backbone feature")?;
        let logits = self
            .head
            .forward(&pooled)
            .context("forward mobile recognizer SVTR head")?;
        Ok(RecognizerOutput::new(logits))
    }
}
