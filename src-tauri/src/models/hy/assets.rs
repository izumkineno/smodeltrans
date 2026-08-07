//! Local Hy-MT2 GGUF asset ownership.
//!
//! Hy obtains its tokenizer vocabulary and special-token metadata from the
//! validated GGUF itself; no remote tokenizer or sidecar runtime is accepted.

use crate::backend::failure::BackendFailure;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HyAssets {
    pub(crate) model: PathBuf,
}

impl HyAssets {
    pub(crate) fn preflight(path: impl AsRef<Path>) -> Result<Self, BackendFailure> {
        let model = path.as_ref().to_path_buf();
        let metadata = fs::metadata(&model).map_err(|error| {
            BackendFailure::asset(format!(
                "read Hy GGUF metadata {}: {error}",
                model.display()
            ))
        })?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(BackendFailure::asset(format!(
                "Hy GGUF is empty or not a file: {}",
                model.display()
            )));
        }
        if model.extension().and_then(|value| value.to_str()) != Some("gguf") {
            return Err(BackendFailure::asset(format!(
                "Hy model must be a GGUF file: {}",
                model.display()
            )));
        }
        Ok(Self { model })
    }

    pub(crate) fn validate(&self) -> Result<(), BackendFailure> {
        Self::preflight(&self.model).map(|_| ())
    }
}
