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

    fn expected_model_type(self) -> &'static str {
        match self {
            Self::Detector => "pp_ocrv5_server_det",
            Self::Recognizer => "pp_ocrv5_server_rec",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PpOcrV5Assets {
    pub(crate) role: GraphRole,
    pub(crate) directory: PathBuf,
    pub(crate) weights: PathBuf,
    pub(crate) config: PathBuf,
    pub(crate) preprocessor: PathBuf,
    pub(crate) inference: PathBuf,
    pub(crate) character_list: Vec<String>,
    pub(crate) manifest_digest: String,
}

impl PpOcrV5Assets {
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
        for (name, path) in [
            ("model.safetensors", &weights),
            ("config.json", &config),
            ("preprocessor_config.json", &preprocessor),
            ("inference.yml", &inference),
        ] {
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
            let _ = name;
        }
        let config_value: Value = serde_json::from_slice(
            &fs::read(&config)
                .map_err(|error| BackendFailure::asset(format!("read OCR config: {error}")))?,
        )
        .map_err(|error| BackendFailure::asset(format!("parse OCR config: {error}")))?;
        let model_type = config_value
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| BackendFailure::asset("OCR config has no model_type"))?;
        if model_type != role.expected_model_type() {
            return Err(BackendFailure::asset(format!(
                "{} config has unexpected model_type {model_type}",
                role.as_str()
            )));
        }
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
            directory,
            weights,
            config,
            preprocessor,
            inference,
            character_list,
            manifest_digest: format_digest(digest.finalize()),
        })
    }

    pub(crate) fn validate(&self) -> Result<(), BackendFailure> {
        Self::preflight(self.role, &self.directory).map(|_| ())
    }
}

fn format_digest(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
