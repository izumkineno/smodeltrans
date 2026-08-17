//! PP-OCR asset discovery: identity, variant, and postprocess parameters.
//!
//! Identity is authoritative from each asset tree's `inference.yml`
//! `Global.model_name` (e.g. `PP-OCRv6_tiny_det`).  `config.json` is not
//! trusted for identity or pairing: upstream PP-OCRv6 exports mislabel some
//! tiers (the tiny detector declares `pp_ocrv6_small_det` and the medium
//! recognizer declares `pp_ocrv6_small_rec`), so pairing and status logic
//! must never consult `config.json.model_type`.

use crate::backend::failure::BackendFailure;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphRole {
    Detector,
    Recognizer,
}

impl GraphRole {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "detector",
            Self::Recognizer => "recognizer",
        }
    }
}

/// Every supported PP-OCR model pair tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PpOcrVariant {
    V5Server,
    V5Mobile,
    V6Tiny,
    V6Small,
    V6Medium,
}

impl PpOcrVariant {
    /// Parse an `inference.yml` `Global.model_name` for a graph role.
    pub(crate) fn from_model_name(role: GraphRole, model_name: &str) -> Option<Self> {
        match (role, model_name) {
            (GraphRole::Detector, "PP-OCRv5_server_det")
            | (GraphRole::Recognizer, "PP-OCRv5_server_rec") => Some(Self::V5Server),
            (GraphRole::Detector, "PP-OCRv5_mobile_det")
            | (GraphRole::Recognizer, "PP-OCRv5_mobile_rec") => Some(Self::V5Mobile),
            (GraphRole::Detector, "PP-OCRv6_tiny_det")
            | (GraphRole::Recognizer, "PP-OCRv6_tiny_rec") => Some(Self::V6Tiny),
            (GraphRole::Detector, "PP-OCRv6_small_det")
            | (GraphRole::Recognizer, "PP-OCRv6_small_rec") => Some(Self::V6Small),
            (GraphRole::Detector, "PP-OCRv6_medium_det")
            | (GraphRole::Recognizer, "PP-OCRv6_medium_rec") => Some(Self::V6Medium),
            _ => None,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::V5Server => "v5-server",
            Self::V5Mobile => "v5-mobile",
            Self::V6Tiny => "v6-tiny",
            Self::V6Small => "v6-small",
            Self::V6Medium => "v6-medium",
        }
    }

    pub(crate) const fn is_v5(self) -> bool {
        matches!(self, Self::V5Server | Self::V5Mobile)
    }

    pub(crate) const fn is_v6(self) -> bool {
        !self.is_v5()
    }

    /// Cheap variant probe for status display and settings validation:
    /// reads `inference.yml` only, without hashing the weight manifest.
    pub(crate) fn probe(role: GraphRole, directory: &Path) -> Option<Self> {
        let inference = read_inference_yaml(&directory.join("inference.yml"))?;
        let model_name = inference.get("Global")?.get("model_name")?.as_scalar()?;
        Self::from_model_name(role, model_name)
    }
}

/// DB postprocess parameters shipped with each detector asset tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DetectorPostProcess {
    pub(crate) binary_thresh: f32,
    pub(crate) box_thresh: f32,
    pub(crate) max_candidates: usize,
    pub(crate) unclip_ratio: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct PpOcrAssets {
    pub(crate) role: GraphRole,
    pub(crate) variant: PpOcrVariant,
    pub(crate) directory: PathBuf,
    pub(crate) weights: PathBuf,
    pub(crate) config: PathBuf,
    pub(crate) preprocessor: PathBuf,
    pub(crate) inference: PathBuf,
    pub(crate) character_list: Vec<String>,
    pub(crate) postprocess: Option<DetectorPostProcess>,
    pub(crate) manifest_digest: String,
}

impl PpOcrAssets {
    pub(crate) fn preflight(
        role: GraphRole,
        directory: impl AsRef<Path>,
    ) -> Result<Self, BackendFailure> {
        let directory = directory.as_ref().to_path_buf();
        if !directory.is_dir() {
            return Err(BackendFailure::asset(format!(
                "{} model directory does not exist: {}",
                role.as_str(),
                directory.display()
            )));
        }
        let weights = directory.join("model.safetensors");
        let config = directory.join("config.json");
        let preprocessor = directory.join("preprocessor_config.json");
        let inference = directory.join("inference.yml");
        for path in [&weights, &config, &preprocessor, &inference] {
            let metadata = fs::metadata(path).map_err(|_| {
                BackendFailure::asset(format!(
                    "{} asset is missing: {}",
                    role.as_str(),
                    path.display()
                ))
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(BackendFailure::asset(format!(
                    "{} asset is empty or not a file: {}",
                    role.as_str(),
                    path.display()
                )));
            }
        }
        let inference_value = read_inference_yaml(&inference).ok_or_else(|| {
            BackendFailure::asset(format!(
                "{} inference.yml has no Global.model_name",
                role.as_str()
            ))
        })?;
        let model_name = inference_value
            .get("Global")
            .and_then(|node| node.get("model_name"))
            .and_then(|node| node.as_scalar())
            .ok_or_else(|| {
                BackendFailure::asset(format!(
                    "{} inference.yml has no Global.model_name",
                    role.as_str()
                ))
            })?;
        let variant = PpOcrVariant::from_model_name(role, model_name).ok_or_else(|| {
            BackendFailure::asset(format!(
                "{} inference.yml has unexpected model_name {model_name}",
                role.as_str()
            ))
        })?;
        let processor_value: Value =
            serde_json::from_slice(&fs::read(&preprocessor).map_err(|error| {
                BackendFailure::asset(format!("read OCR preprocessor: {error}"))
            })?)
            .map_err(|error| BackendFailure::asset(format!("parse OCR preprocessor: {error}")))?;
        let mut character_list = Vec::new();
        if role == GraphRole::Recognizer {
            let values = processor_value
                .get("character_list")
                .and_then(Value::as_array)
                .ok_or_else(|| BackendFailure::asset("recognizer character_list is missing"))?;
            for value in values {
                let character = value.as_str().ok_or_else(|| {
                    BackendFailure::asset("recognizer character_list contains non-string data")
                })?;
                character_list.push(character.to_owned());
            }
            if character_list.is_empty() {
                return Err(BackendFailure::asset("recognizer character_list is empty"));
            }
        } else if processor_value.get("image_mode").and_then(Value::as_str) != Some("BGR") {
            return Err(BackendFailure::asset(
                "detector preprocessor must use BGR image mode",
            ));
        }
        let postprocess = if role == GraphRole::Detector {
            Some(parse_db_postprocess(&inference_value).ok_or_else(|| {
                BackendFailure::asset("detector inference.yml has no DBPostProcess values")
            })?)
        } else {
            None
        };
        let mut digest = Sha256::new();
        for path in [&weights, &config, &preprocessor, &inference] {
            let bytes = fs::read(path)
                .map_err(|error| BackendFailure::asset(format!("hash OCR asset: {error}")))?;
            digest.update(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .as_bytes(),
            );
            digest.update(bytes);
        }
        Ok(Self {
            role,
            variant,
            directory,
            weights,
            config,
            preprocessor,
            inference,
            character_list,
            postprocess,
            manifest_digest: format_digest(digest.finalize()),
        })
    }

    pub(crate) fn validate(&self) -> Result<(), BackendFailure> {
        Self::preflight(self.role, &self.directory).map(|_| ())
    }
}

fn parse_db_postprocess(inference: &YamlNode) -> Option<DetectorPostProcess> {
    let postprocess = inference.get("PostProcess")?;
    if postprocess.get("name")?.as_scalar()? != "DBPostProcess" {
        return None;
    }
    let binary_thresh = postprocess.get("thresh")?.as_scalar()?.parse().ok()?;
    let box_thresh = postprocess.get("box_thresh")?.as_scalar()?.parse().ok()?;
    let max_candidates = postprocess
        .get("max_candidates")?
        .as_scalar()?
        .parse()
        .ok()?;
    let unclip_ratio = postprocess.get("unclip_ratio")?.as_scalar()?.parse().ok()?;
    Some(DetectorPostProcess {
        binary_thresh,
        box_thresh,
        max_candidates,
        unclip_ratio,
    })
}

fn format_digest(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal YAML-subset reader.
//
// The shipped `inference.yml` files are generated by the same export pipeline
// and only need nested mapping support (`Key:`, indented `  subkey: value`,
// scalar values, and `- item` list lines we deliberately skip).  This keeps
// asset parsing dependency-free instead of pulling in a full YAML crate.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum YamlNode {
    Map(Vec<(String, YamlNode)>),
    Scalar(String),
}

impl YamlNode {
    pub(crate) fn as_map(&self) -> &[(String, YamlNode)] {
        match self {
            Self::Map(entries) => entries,
            Self::Scalar(_) => &[],
        }
    }

    pub(crate) fn as_scalar(&self) -> Option<&str> {
        match self {
            Self::Scalar(value) => Some(value.as_str()),
            Self::Map(_) => None,
        }
    }

    pub(crate) fn get<'a>(&'a self, key: &str) -> Option<&'a YamlNode> {
        self.as_map()
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, node)| node)
    }
}

/// Parse a mapping root: `Key:` lines whose indentation defines nesting.
/// Sequence items (`- ...`) are skipped; their scalar values are never needed
/// for identity or DB postprocess resolution.
pub(crate) fn read_inference_yaml(path: &Path) -> Option<YamlNode> {
    let content = fs::read_to_string(path).ok()?;
    parse_yaml_str(&content)
}

pub(crate) fn parse_yaml_str(content: &str) -> Option<YamlNode> {
    let mut stack: Vec<(usize, String, YamlNode)> = Vec::new();
    let mut root = YamlNode::Map(Vec::new());
    for raw_line in content.lines() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if line.trim_start().starts_with('-') {
            continue;
        }
        let (key, value) = line.trim().split_once(':')?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() {
            continue;
        }
        // Close maps whose indentation is at or below this line's, attaching
        // each completed map to its parent (or the root).
        while stack
            .last()
            .is_some_and(|(parent_indent, _, _)| *parent_indent >= indent)
        {
            let (_, closed_key, closed) = stack.pop().expect("stack entry checked above");
            attach_yaml_node(&mut root, &mut stack, closed_key, closed);
        }
        if value.is_empty() {
            stack.push((indent, key.to_owned(), YamlNode::Map(Vec::new())));
        } else {
            attach_yaml_node(
                &mut root,
                &mut stack,
                key.to_owned(),
                YamlNode::Scalar(value.to_owned()),
            );
        }
    }
    while let Some((_, closed_key, closed)) = stack.pop() {
        attach_yaml_node(&mut root, &mut stack, closed_key, closed);
    }
    if let YamlNode::Map(entries) = &root {
        if entries.is_empty() {
            return None;
        }
    }
    Some(root)
}

fn attach_yaml_node(
    root: &mut YamlNode,
    stack: &mut Vec<(usize, String, YamlNode)>,
    key: String,
    node: YamlNode,
) {
    match stack.last_mut() {
        Some((_, _, parent)) => {
            if let YamlNode::Map(entries) = parent {
                entries.push((key, node));
            }
        }
        None => {
            if let YamlNode::Map(entries) = root {
                entries.push((key, node));
            }
        }
    }
}
