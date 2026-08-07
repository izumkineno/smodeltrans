use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
};

const DEFAULT_HY_FILE: &str = "Hy-MT2-1.8B-Q4_K_M.gguf";
const DEFAULT_IDLE_UNLOAD_MINUTES: u32 = 30;
const MAX_IDLE_UNLOAD_MINUTES: u32 = 24 * 60;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeviceKind {
    Cpu,
    Cuda,
}

impl DeviceKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

impl std::str::FromStr for DeviceKind {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BackendSettings {
    pub(crate) detector_model_dir: PathBuf,
    pub(crate) recognizer_model_dir: PathBuf,
    pub(crate) hy_model: PathBuf,
    pub(crate) font_path: Option<PathBuf>,
    pub(crate) target_language: String,
    pub(crate) region_parallelism: usize,
    pub(crate) translation_batch_size: usize,
    pub(crate) device_kind: DeviceKind,
    pub(crate) idle_unload_minutes: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedBackendSettings {
    pub(crate) detector_model_dir: Option<PathBuf>,
    pub(crate) recognizer_model_dir: Option<PathBuf>,
    pub(crate) hy_model: Option<PathBuf>,
    pub(crate) idle_unload_minutes: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendStatus {
    pub(crate) ready: bool,
    pub(crate) device: String,
    pub(crate) detector_model_dir: String,
    pub(crate) recognizer_model_dir: String,
    pub(crate) hy_model: String,
    pub(crate) font_path: Option<String>,
    pub(crate) translator_loaded: bool,
    pub(crate) idle_unload_minutes: u32,
    pub(crate) message: String,
}

impl BackendSettings {
    pub(crate) fn from_environment() -> Result<Self, String> {
        Self::from_environment_with_resource_root_and_config(None, None)
    }

    pub(crate) fn from_environment_with_resource_root(
        resource_root: Option<PathBuf>,
    ) -> Result<Self, String> {
        Self::from_environment_with_resource_root_and_config(resource_root, None)
    }

    pub(crate) fn from_environment_with_resource_root_and_config(
        resource_root: Option<PathBuf>,
        config_path: Option<&Path>,
    ) -> Result<Self, String> {
        let persisted = config_path.and_then(read_persisted_settings);
        let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = env::var_os("SMODELTRANS_WORKSPACE_ROOT")
            .map(|path| resolve_path(Some(PathBuf::from(path)), Path::new(".")))
            .unwrap_or_else(|| manifest_root.join(".."));
        let packaged_model_root = if cfg!(debug_assertions) {
            None
        } else {
            resource_root.map(|root| root.join("models"))
        };
        let model_root = env::var_os("SMODELTRANS_MODEL_ROOT")
            .map(|path| resolve_path(Some(PathBuf::from(path)), &workspace_root))
            .or(packaged_model_root)
            .unwrap_or_else(|| workspace_root.join("models"));
        let detector_default = model_root.join("ppocrv5").join("detector");
        let recognizer_default = model_root.join("ppocrv5").join("recognizer");
        let hy_default = model_root.join("hy").join(DEFAULT_HY_FILE);
        let device_text = env::var("SMODELTRANS_DEVICE").unwrap_or_else(|_| "cuda".to_owned());
        let device_kind = device_text
            .parse()
            .map_err(|_| "SMODELTRANS_DEVICE must be cpu or cuda".to_owned())?;
        let target_language = env::var("SMODELTRANS_TARGET_LANGUAGE")
            .unwrap_or_else(|_| "Chinese".to_owned())
            .trim()
            .to_owned();
        let target_length = target_language.chars().count();
        if !(1..=64).contains(&target_length) {
            return Err("SMODELTRANS_TARGET_LANGUAGE must contain 1..=64 characters".to_owned());
        }
        let font_path = env::var_os("SMODELTRANS_FONT_PATH")
            .map(|path| resolve_path(Some(path), &workspace_root));
        let detector_model_dir = persisted
            .as_ref()
            .and_then(|settings| settings.detector_model_dir.clone())
            .unwrap_or_else(|| {
                env_path(
                    "SMODELTRANS_DET_MODEL_DIR",
                    detector_default,
                    &workspace_root,
                )
            });
        let recognizer_model_dir = persisted
            .as_ref()
            .and_then(|settings| settings.recognizer_model_dir.clone())
            .unwrap_or_else(|| {
                env_path(
                    "SMODELTRANS_REC_MODEL_DIR",
                    recognizer_default,
                    &workspace_root,
                )
            });
        let hy_model = persisted
            .as_ref()
            .and_then(|settings| settings.hy_model.clone())
            .unwrap_or_else(|| env_path("SMODELTRANS_HY_MODEL", hy_default, &workspace_root));
        let idle_unload_minutes = persisted
            .and_then(|settings| settings.idle_unload_minutes)
            .unwrap_or(bounded_env_u32(
                "SMODELTRANS_IDLE_UNLOAD_MINUTES",
                DEFAULT_IDLE_UNLOAD_MINUTES,
                0,
                MAX_IDLE_UNLOAD_MINUTES,
            )?);
        Ok(Self {
            detector_model_dir,
            recognizer_model_dir,
            hy_model,
            font_path,
            target_language,
            region_parallelism: bounded_env("SMODELTRANS_REGION_PARALLELISM", 16, 1, 16)?,
            translation_batch_size: bounded_env("SMODELTRANS_TRANSLATION_BATCH_SIZE", 4, 1, 4)?,
            device_kind,
            idle_unload_minutes,
        })
    }

    pub(crate) fn update_model_paths(
        &self,
        detector_model_dir: &str,
        recognizer_model_dir: &str,
        hy_model: &str,
        idle_unload_minutes: u32,
    ) -> Result<Self, String> {
        if idle_unload_minutes > MAX_IDLE_UNLOAD_MINUTES {
            return Err(format!(
                "idle_unload_minutes must be in 0..={MAX_IDLE_UNLOAD_MINUTES}"
            ));
        }
        let mut next = self.clone();
        next.detector_model_dir = selected_path("detector_model_dir", detector_model_dir)?;
        next.recognizer_model_dir = selected_path("recognizer_model_dir", recognizer_model_dir)?;
        next.hy_model = selected_path("hy_model", hy_model)?;
        next.idle_unload_minutes = idle_unload_minutes;
        Ok(next)
    }

    pub(crate) fn persisted(&self) -> PersistedBackendSettings {
        PersistedBackendSettings {
            detector_model_dir: Some(self.detector_model_dir.clone()),
            recognizer_model_dir: Some(self.recognizer_model_dir.clone()),
            hy_model: Some(self.hy_model.clone()),
            idle_unload_minutes: Some(self.idle_unload_minutes),
        }
    }

    pub(crate) fn status(&self, translator_loaded: bool) -> BackendStatus {
        let ocr_assets_ready = self.detector_model_dir.is_dir()
            && self.recognizer_model_dir.is_dir()
            && self.font_path.as_ref().is_none_or(|path| path.is_file());
        let hy_asset_ready = self.hy_model.is_file();
        let ready = ocr_assets_ready && hy_asset_ready && self.device_kind == DeviceKind::Cuda;
        let message = if !ocr_assets_ready {
            "PP-OCRv5 模型资产未就绪，请检查 models/ppocrv5 文件夹。"
        } else if self.device_kind == DeviceKind::Cpu {
            "图片翻译需要 CUDA；CPU 配置不会执行 OCR-only 成功路径。"
        } else if !hy_asset_ready {
            "Hy-MT2 模型资产未就绪，请检查 models/hy 文件夹。"
        } else {
            "PP-OCRv5 与 Hy-MT2 本地模型资产已就绪。"
        };
        BackendStatus {
            ready,
            device: self.device_kind.as_str().to_owned(),
            detector_model_dir: self.detector_model_dir.display().to_string(),
            recognizer_model_dir: self.recognizer_model_dir.display().to_string(),
            hy_model: self.hy_model.display().to_string(),
            font_path: self
                .font_path
                .as_ref()
                .map(|path| path.display().to_string()),
            translator_loaded,
            idle_unload_minutes: self.idle_unload_minutes,
            message: message.to_owned(),
        }
    }
}

impl BackendStatus {
    pub(crate) fn configuration_error(message: &str) -> Self {
        Self {
            ready: false,
            device: "invalid".to_owned(),
            detector_model_dir: String::new(),
            recognizer_model_dir: String::new(),
            hy_model: String::new(),
            font_path: None,
            translator_loaded: false,
            idle_unload_minutes: DEFAULT_IDLE_UNLOAD_MINUTES,
            message: format!("后端配置无效：{message}"),
        }
    }
}

fn read_persisted_settings(path: &Path) -> Option<PersistedBackendSettings> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn selected_path(name: &str, value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value.trim());
    if path.as_os_str().is_empty() {
        return Err(format!("{name} 不能为空"));
    }
    if !path.is_absolute() {
        return Err(format!("{name} 必须是绝对路径"));
    }
    Ok(path)
}

fn bounded_env_u32(name: &str, default: u32, min: u32, max: u32) -> Result<u32, String> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{name} must be an integer"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{name} must be in {min}..={max}"));
    }
    Ok(parsed)
}

fn env_path(name: &str, default: PathBuf, base: &Path) -> PathBuf {
    env::var_os(name)
        .map(|path| resolve_path(Some(path), base))
        .unwrap_or(default)
}

fn resolve_path(value: Option<impl Into<PathBuf>>, base: impl AsRef<Path>) -> PathBuf {
    let path = value
        .map(Into::into)
        .unwrap_or_else(|| base.as_ref().to_path_buf());
    if path.is_absolute() {
        path
    } else {
        base.as_ref().join(path)
    }
}

fn bounded_env(name: &str, default: usize, min: usize, max: usize) -> Result<usize, String> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be an integer"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{name} must be in {min}..={max}"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::{BackendSettings, DeviceKind};
    use std::path::PathBuf;

    fn settings() -> BackendSettings {
        BackendSettings {
            detector_model_dir: PathBuf::from("C:\\models\\detector"),
            recognizer_model_dir: PathBuf::from("C:\\models\\recognizer"),
            hy_model: PathBuf::from("C:\\models\\hy.gguf"),
            font_path: None,
            target_language: "Chinese".to_owned(),
            region_parallelism: 16,
            translation_batch_size: 4,
            device_kind: DeviceKind::Cuda,
            idle_unload_minutes: 30,
        }
    }

    #[test]
    fn selected_model_paths_and_idle_timeout_are_updated_together() {
        let updated = settings()
            .update_model_paths(
                "D:\\models\\detector",
                "D:\\models\\recognizer",
                "D:\\models\\hy.gguf",
                0,
            )
            .expect("valid model settings");

        assert_eq!(
            updated.detector_model_dir,
            PathBuf::from("D:\\models\\detector")
        );
        assert_eq!(
            updated.recognizer_model_dir,
            PathBuf::from("D:\\models\\recognizer")
        );
        assert_eq!(updated.hy_model, PathBuf::from("D:\\models\\hy.gguf"));
        assert_eq!(updated.idle_unload_minutes, 0);
    }

    #[test]
    fn selected_model_paths_reject_relative_paths_and_large_idle_timeout() {
        let current = settings();

        assert!(
            current
                .update_model_paths(
                    "models/detector",
                    "D:\\models\\recognizer",
                    "D:\\models\\hy.gguf",
                    30
                )
                .is_err()
        );
        assert!(
            current
                .update_model_paths(
                    "D:\\models\\detector",
                    "D:\\models\\recognizer",
                    "D:\\models\\hy.gguf",
                    1441,
                )
                .is_err()
        );
    }
}
