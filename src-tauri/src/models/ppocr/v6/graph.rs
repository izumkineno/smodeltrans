//! Native PP-OCRv6 detector and recognizer graphs.
//!
//! Tier dispatch: tiny/small detectors use PPLCNetV4 + RepLKFPN + DB head;
//! the medium detector uses PPLCNetV4 + RepLKPAN (the shared DB++ neck) + DB
//! head; tiny recognizers use the reshape-CTC projection head; small/medium
//! recognizers use PPLCNetV4 + LightSVTR + CTC.  The medium detector's weight
//! manifest (350 tensors) is validated before graph load so a mislabeled or
//! mismatched asset fails with an explicit error instead of silently loading
//! as a different tier.

use super::super::assets::{GraphRole, PpOcrAssets, PpOcrVariant};
use super::super::common::{
    DbHead, DetectorOutput, RecognizerOutput, image_batch, load_mmaped_weights,
};
use super::super::v5::server::DetNeck;
use super::{backbone::PpLcNetV4, head::LightSvtrHead, head::TinyCtcHead, neck::RepLkFpn};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use serde_json::Value;

/// Expected weight tensor count for each v6 detector tier.  These are hard
/// gates: a mismatched manifest is an asset error, never a silent fallback.
const V6_DET_TENSOR_COUNTS: [(PpOcrVariant, usize); 3] = [
    (PpOcrVariant::V6Tiny, 271),
    (PpOcrVariant::V6Small, 271),
    (PpOcrVariant::V6Medium, 350),
];

fn count_safetensors_tensors(path: &std::path::Path) -> Result<usize> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {} for manifest validation", path.display()))?;
    ensure!(
        bytes.len() >= 8,
        "safetensors file is too short: {}",
        path.display()
    );
    let header_len: usize = u64::from_le_bytes(bytes[0..8].try_into().expect("8-byte header length"))
        .try_into()
        .context("safetensors header length overflows usize")?;
    ensure!(
        header_len < bytes.len(),
        "safetensors header length exceeds file size"
    );
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .context("parse safetensors header")?;
    let map = header
        .as_object()
        .context("safetensors header must be a JSON object")?;
    Ok(map.keys().count())
}

/// Native PP-OCRv6 detector.
#[derive(Clone, Debug)]
pub(crate) struct PpOcrV6Detector {
    backbone: PpLcNetV4,
    neck: V6DetNeck,
    head: DbHead,
}

impl PpOcrV6Detector {
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        ensure!(
            assets.role == GraphRole::Detector,
            "v6 detector loader received {} assets",
            assets.role.as_str()
        );
        ensure!(
            assets.variant.is_v6(),
            "v6 detector loader received {} assets",
            assets.variant.label()
        );
        if let Some((_, expected)) = V6_DET_TENSOR_COUNTS
            .iter()
            .find(|(variant, _)| *variant == assets.variant)
        {
            let actual = count_safetensors_tensors(&assets.weights).with_context(|| {
                format!(
                    "validate {} detector weight manifest",
                    assets.variant.label()
                )
            })?;
            ensure!(
                actual == *expected,
                "{} detector weight manifest has {actual} tensors, expected {expected}; \
                 refusing to load a mismatched PP-OCRv6 asset",
                assets.variant.label()
            );
        }
        let config_value: Value = serde_json::from_slice(
            &std::fs::read(&assets.config)
                .with_context(|| format!("read {}", assets.config.display()))?,
        )
        .context("parse v6 detector config.json")?;
        let backbone_config = config_value
            .get("backbone_config")
            .context("v6 detector config has no backbone_config")?;
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = PpLcNetV4::load(
            vb.pp("model").pp("backbone"),
            backbone_config,
            true,
        )
        .context("load native PP-OCRv6 detector PPLCNetV4 backbone")?;
        let neck = match assets.variant {
            PpOcrVariant::V6Medium => {
                let out_channels = config_value
                    .get("neck_out_channels")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("v6 medium detector config has no neck_out_channels")?;
                let in_channels = backbone_stage_out_channels(backbone_config);
                V6DetNeck::Pan(DetNeck::load(vb.pp("model").pp("neck"), in_channels, out_channels)?)
            }
            _ => {
                let out_channels = config_value
                    .get("neck_out_channels")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("v6 detector config has no neck_out_channels")?;
                let dilated_kernel_size = config_value
                    .get("dilated_kernel_size")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("v6 detector config has no dilated_kernel_size")?;
                let in_channels = backbone_stage_out_channels(backbone_config);
                V6DetNeck::Fpn(RepLkFpn::load(
                    vb.pp("model").pp("neck"),
                    in_channels,
                    out_channels,
                    dilated_kernel_size,
                )?)
            }
        };
        let head_channels = config_value
            .get("neck_out_channels")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .context("v6 detector config has no neck_out_channels")?;
        let head = DbHead::load(vb.pp("head"), head_channels)
            .context("load native PP-OCRv6 detector DB head")?;
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
            .context("forward native PP-OCRv6 detector backbone")?;
        let fuse = self
            .neck
            .forward(&features)
            .context("forward native PP-OCRv6 detector neck")?;
        let probabilities = self
            .head
            .forward(&fuse)
            .context("forward native PP-OCRv6 detector head")?;
        Ok(DetectorOutput::new(probabilities))
    }
}

#[derive(Clone, Debug)]
enum V6DetNeck {
    Fpn(RepLkFpn),
    Pan(DetNeck),
}

impl V6DetNeck {
    fn forward(&self, features: &[Tensor]) -> Result<Tensor> {
        match self {
            Self::Fpn(neck) => neck.forward(features),
            Self::Pan(neck) => Ok(neck.forward(features)?),
        }
    }
}

/// The four backbone stage output channels, in export order.
fn backbone_stage_out_channels(backbone_config: &Value) -> [usize; 4] {
    let mut out = [0usize; 4];
    if let Some(stages) = backbone_config
        .get("block_configs")
        .and_then(Value::as_array)
    {
        for (index, stage) in stages.iter().enumerate().take(4) {
            if let Some(blocks) = stage.as_array() {
                if let Some(last) = blocks.last().and_then(Value::as_array) {
                    if let Some(channels) = last.get(2).and_then(Value::as_u64) {
                        out[index] = channels as usize;
                    }
                }
            }
        }
    }
    out
}

/// Native PP-OCRv6 recognizer.
#[derive(Clone, Debug)]
pub(crate) struct PpOcrV6Recognizer {
    backbone: PpLcNetV4,
    head: V6RecHead,
}

impl PpOcrV6Recognizer {
    pub(crate) fn load(assets: &PpOcrAssets, device: &Device) -> Result<Self> {
        ensure!(
            assets.role == GraphRole::Recognizer,
            "v6 recognizer loader received {} assets",
            assets.role.as_str()
        );
        ensure!(
            assets.variant.is_v6(),
            "v6 recognizer loader received {} assets",
            assets.variant.label()
        );
        let config_value: Value = serde_json::from_slice(
            &std::fs::read(&assets.config)
                .with_context(|| format!("read {}", assets.config.display()))?,
        )
        .context("parse v6 recognizer config.json")?;
        let backbone_config = config_value
            .get("backbone_config")
            .context("v6 recognizer config has no backbone_config")?;
        let vb = load_mmaped_weights(assets, device)?;
        let backbone = PpLcNetV4::load(
            vb.pp("model").pp("backbone"),
            backbone_config,
            false,
        )
        .context("load native PP-OCRv6 recognizer PPLCNetV4 backbone")?;
        let head = match assets.variant {
            PpOcrVariant::V6Tiny => {
                let hidden_size = config_value
                    .get("hidden_size")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("tiny recognizer config has no hidden_size")?;
                let out_channels = config_value
                    .get("head_out_channels")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("tiny recognizer config has no head_out_channels")?;
                V6RecHead::Tiny(TinyCtcHead::load(
                    vb.pp("head"),
                    backbone_stage_out_channels(backbone_config)[3],
                    hidden_size,
                    out_channels,
                )?)
            }
            _ => {
                let hidden_size = config_value
                    .get("hidden_size")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("v6 recognizer config has no hidden_size")?;
                let out_channels = config_value
                    .get("head_out_channels")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .context("v6 recognizer config has no head_out_channels")?;
                let mlp_ratio = config_value
                    .get("mlp_ratio")
                    .and_then(Value::as_f64)
                    .map(|value| value as f32)
                    .unwrap_or(4.0);
                let depth = config_value
                    .get("depth")
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .unwrap_or(2);
                let local_kernel = config_value
                    .get("conv_kernel_size")
                    .and_then(Value::as_array)
                    .and_then(|pair| pair.last())
                    .and_then(Value::as_u64)
                    .map(|value| value as usize)
                    .unwrap_or(7);
                V6RecHead::Light(LightSvtrHead::load(
                    vb.pp("head"),
                    backbone_stage_out_channels(backbone_config)[3],
                    hidden_size,
                    mlp_ratio,
                    depth,
                    local_kernel,
                    out_channels,
                )?)
            }
        };
        Ok(Self { backbone, head })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<RecognizerOutput> {
        let input = image_batch(input)?;
        let feature = self
            .backbone
            .forward_rec(&input)
            .context("forward native PP-OCRv6 recognizer backbone")?;
        let logits = self
            .head
            .forward(&feature)
            .context("forward native PP-OCRv6 recognizer head")?;
        Ok(RecognizerOutput::new(logits))
    }
}

#[derive(Clone, Debug)]
enum V6RecHead {
    Tiny(TinyCtcHead),
    Light(LightSvtrHead),
}

impl V6RecHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Tiny(head) => head.forward(xs),
            Self::Light(head) => head.forward(xs),
        }
    }
}


