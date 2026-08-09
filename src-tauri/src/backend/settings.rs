use crate::model_config::{GenerationConfig, MemoryConfig, PromptConfig};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
};

const DEFAULT_HY_FILE: &str = "Hy-MT2-1.8B-Q4_K_M.gguf";
const DEFAULT_IDLE_UNLOAD_MINUTES: u32 = 30;
const MAX_IDLE_UNLOAD_MINUTES: u32 = 24 * 60;
const DEFAULT_TARGET_LANGUAGE: &str = "Chinese";
const DEFAULT_REGION_PARALLELISM: usize = 16;
const DEFAULT_TRANSLATION_BATCH_SIZE: usize = 4;
const MAX_TARGET_LANGUAGE_CHARS: usize = 64;

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
    pub(crate) prompt: PromptConfig,
    pub(crate) generation: GenerationConfig,
    pub(crate) memory: MemoryConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendGenerationSettings {
    pub(crate) max_new_tokens: usize,
    pub(crate) sampling: bool,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) seed: Option<String>,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) stop_tokens: Vec<u32>,
    pub(crate) stop_strings: Vec<String>,
}

impl BackendGenerationSettings {
    fn from_config(config: &GenerationConfig) -> Self {
        Self {
            max_new_tokens: config.max_new_tokens,
            sampling: config.sampling,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            seed: config.seed.map(|seed| seed.to_string()),
            repetition_penalty: config.repetition_penalty,
            frequency_penalty: config.frequency_penalty,
            stop_tokens: config.stop_tokens.clone(),
            stop_strings: config.stop_strings.clone(),
        }
    }

    fn into_config(self) -> Result<GenerationConfig, String> {
        let config = GenerationConfig {
            max_new_tokens: self.max_new_tokens,
            sampling: self.sampling,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: parse_seed(self.seed)?,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            stop_tokens: unique_tokens(self.stop_tokens),
            stop_strings: unique_trimmed_strings(self.stop_strings),
        };
        validate_generation(config)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendMemorySettings {
    pub(crate) enabled: bool,
    pub(crate) max_tokens: usize,
    pub(crate) max_turns: usize,
}

impl BackendMemorySettings {
    fn from_config(config: &MemoryConfig) -> Self {
        Self {
            enabled: config.enabled,
            max_tokens: config.max_tokens,
            max_turns: config.max_turns,
        }
    }

    fn into_config(self) -> Result<MemoryConfig, String> {
        let config = MemoryConfig {
            enabled: self.enabled,
            max_tokens: self.max_tokens,
            max_turns: self.max_turns,
        };
        validate_memory(config)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendPromptSettings {
    pub(crate) system: String,
    #[serde(default)]
    pub(crate) user: String,
}

impl BackendPromptSettings {
    fn from_config(config: &PromptConfig) -> Self {
        Self {
            system: config.system.clone(),
            user: config.user.clone(),
        }
    }

    fn into_config(self) -> Result<PromptConfig, String> {
        let config = PromptConfig {
            system: self.system.trim().to_owned(),
            user: self.user.trim().to_owned(),
        };
        validate_prompt(config)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendSettingsUpdate {
    pub(crate) detector_model_dir: String,
    pub(crate) recognizer_model_dir: String,
    pub(crate) hy_model: String,
    pub(crate) font_path: Option<String>,
    pub(crate) target_language: String,
    pub(crate) device: String,
    pub(crate) region_parallelism: usize,
    pub(crate) translation_batch_size: usize,
    pub(crate) idle_unload_minutes: u32,
    pub(crate) generation: BackendGenerationSettings,
    pub(crate) memory: BackendMemorySettings,
    pub(crate) prompt: BackendPromptSettings,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedBackendSettings {
    pub(crate) detector_model_dir: Option<PathBuf>,
    pub(crate) recognizer_model_dir: Option<PathBuf>,
    pub(crate) hy_model: Option<PathBuf>,
    #[serde(
        default,
        deserialize_with = "deserialize_optional_option",
        serialize_with = "serialize_optional_option"
    )]
    pub(crate) font_path: Option<Option<PathBuf>>,
    pub(crate) target_language: Option<String>,
    pub(crate) device: Option<String>,
    pub(crate) region_parallelism: Option<usize>,
    pub(crate) translation_batch_size: Option<usize>,
    pub(crate) idle_unload_minutes: Option<u32>,
    pub(crate) generation: Option<PersistedGenerationSettings>,
    pub(crate) memory: Option<PersistedMemorySettings>,
    pub(crate) prompt: Option<PersistedPromptSettings>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedGenerationSettings {
    pub(crate) max_new_tokens: Option<usize>,
    pub(crate) sampling: Option<bool>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<String>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) stop_tokens: Option<Vec<u32>>,
    pub(crate) stop_strings: Option<Vec<String>>,
}

impl PersistedGenerationSettings {
    fn into_config(self, mut config: GenerationConfig) -> Result<GenerationConfig, String> {
        if let Some(value) = self.max_new_tokens {
            config.max_new_tokens = value;
        }
        if let Some(value) = self.sampling {
            config.sampling = value;
        }
        if let Some(value) = self.temperature {
            config.temperature = value;
        }
        if let Some(value) = self.top_k {
            config.top_k = value;
        }
        if let Some(value) = self.top_p {
            config.top_p = value;
        }
        if self.seed.is_some() {
            config.seed = parse_seed(self.seed)?;
        }
        if let Some(value) = self.repetition_penalty {
            config.repetition_penalty = value;
        }
        if let Some(value) = self.frequency_penalty {
            config.frequency_penalty = value;
        }
        if let Some(value) = self.stop_tokens {
            config.stop_tokens = unique_tokens(value);
        }
        if let Some(value) = self.stop_strings {
            config.stop_strings = unique_trimmed_strings(value);
        }
        validate_generation(config)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedMemorySettings {
    pub(crate) enabled: Option<bool>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) max_turns: Option<usize>,
}

impl PersistedMemorySettings {
    fn into_config(self, mut config: MemoryConfig) -> Result<MemoryConfig, String> {
        if let Some(value) = self.enabled {
            config.enabled = value;
        }
        if let Some(value) = self.max_tokens {
            config.max_tokens = value;
        }
        if let Some(value) = self.max_turns {
            config.max_turns = value;
        }
        validate_memory(config)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PersistedPromptSettings {
    pub(crate) system: Option<String>,
    pub(crate) user: Option<String>,
}

impl PersistedPromptSettings {
    fn into_config(self, mut config: PromptConfig) -> Result<PromptConfig, String> {
        if let Some(value) = self.system {
            config.system = value.trim().to_owned();
        }
        if let Some(value) = self.user {
            config.user = value.trim().to_owned();
        }
        validate_prompt(config)
    }
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
    pub(crate) target_language: String,
    pub(crate) region_parallelism: usize,
    pub(crate) translation_batch_size: usize,
    pub(crate) translator_loaded: bool,
    pub(crate) idle_unload_minutes: u32,
    pub(crate) generation: BackendGenerationSettings,
    pub(crate) memory: BackendMemorySettings,
    pub(crate) prompt: BackendPromptSettings,
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

        let device_text = persisted
            .as_ref()
            .and_then(|settings| settings.device.clone())
            .unwrap_or_else(|| {
                env::var("SMODELTRANS_DEVICE").unwrap_or_else(|_| "cuda".to_owned())
            });
        let device_kind = parse_device(&device_text)?;

        let target_language = persisted
            .as_ref()
            .and_then(|settings| settings.target_language.clone())
            .unwrap_or_else(|| {
                env::var("SMODELTRANS_TARGET_LANGUAGE")
                    .unwrap_or_else(|_| DEFAULT_TARGET_LANGUAGE.to_owned())
            });
        let target_language = validate_target_language(&target_language)?;

        let font_path_from_env = env::var_os("SMODELTRANS_FONT_PATH")
            .map(|path| resolve_path(Some(path), &workspace_root));
        let font_path = persisted
            .as_ref()
            .and_then(|settings| settings.font_path.clone())
            .unwrap_or(font_path_from_env);

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
            .as_ref()
            .and_then(|settings| settings.idle_unload_minutes)
            .unwrap_or(bounded_env_u32(
                "SMODELTRANS_IDLE_UNLOAD_MINUTES",
                DEFAULT_IDLE_UNLOAD_MINUTES,
                0,
                MAX_IDLE_UNLOAD_MINUTES,
            )?);
        if idle_unload_minutes > MAX_IDLE_UNLOAD_MINUTES {
            return Err(format!(
                "idle_unload_minutes must be in 0..={MAX_IDLE_UNLOAD_MINUTES}"
            ));
        }
        let region_parallelism = persisted
            .as_ref()
            .and_then(|settings| settings.region_parallelism)
            .unwrap_or(bounded_env(
                "SMODELTRANS_REGION_PARALLELISM",
                DEFAULT_REGION_PARALLELISM,
                1,
                DEFAULT_REGION_PARALLELISM,
            )?);
        validate_region_parallelism(region_parallelism)?;
        let translation_batch_size = persisted
            .as_ref()
            .and_then(|settings| settings.translation_batch_size)
            .unwrap_or(bounded_env(
                "SMODELTRANS_TRANSLATION_BATCH_SIZE",
                DEFAULT_TRANSLATION_BATCH_SIZE,
                1,
                DEFAULT_TRANSLATION_BATCH_SIZE,
            )?);
        validate_translation_batch_size(translation_batch_size)?;
        let generation = generation_from_persisted(
            persisted
                .as_ref()
                .and_then(|settings| settings.generation.clone()),
        )?;
        let memory = memory_from_persisted(
            persisted
                .as_ref()
                .and_then(|settings| settings.memory.clone()),
        )?;
        let prompt = prompt_from_persisted(
            persisted
                .as_ref()
                .and_then(|settings| settings.prompt.clone()),
        )?;

        Ok(Self {
            detector_model_dir,
            recognizer_model_dir,
            hy_model,
            font_path,
            target_language,
            region_parallelism,
            translation_batch_size,
            device_kind,
            idle_unload_minutes,
            prompt,
            generation,
            memory,
        })
    }

    pub(crate) fn update_from_request(
        &self,
        request: BackendSettingsUpdate,
    ) -> Result<Self, String> {
        let mut next = self.clone();
        next.detector_model_dir = selected_path("detector_model_dir", &request.detector_model_dir)?;
        next.recognizer_model_dir =
            selected_path("recognizer_model_dir", &request.recognizer_model_dir)?;
        next.hy_model = selected_path("hy_model", &request.hy_model)?;
        next.font_path = selected_optional_path("font_path", request.font_path.as_deref())?;
        next.target_language = validate_target_language(&request.target_language)?;
        next.device_kind = parse_device(&request.device)?;
        validate_region_parallelism(request.region_parallelism)?;
        next.region_parallelism = request.region_parallelism;
        validate_translation_batch_size(request.translation_batch_size)?;
        next.translation_batch_size = request.translation_batch_size;
        validate_idle_unload_minutes(request.idle_unload_minutes)?;
        next.idle_unload_minutes = request.idle_unload_minutes;
        next.generation = request.generation.into_config()?;
        next.memory = request.memory.into_config()?;
        next.prompt = request.prompt.into_config()?;
        Ok(next)
    }

    pub(crate) fn persisted(&self) -> PersistedBackendSettings {
        PersistedBackendSettings {
            detector_model_dir: Some(self.detector_model_dir.clone()),
            recognizer_model_dir: Some(self.recognizer_model_dir.clone()),
            hy_model: Some(self.hy_model.clone()),
            font_path: Some(self.font_path.clone()),
            target_language: Some(self.target_language.clone()),
            device: Some(self.device_kind.as_str().to_owned()),
            region_parallelism: Some(self.region_parallelism),
            translation_batch_size: Some(self.translation_batch_size),
            idle_unload_minutes: Some(self.idle_unload_minutes),
            generation: Some(PersistedGenerationSettings::from(&self.generation)),
            memory: Some(PersistedMemorySettings::from(&self.memory)),
            prompt: Some(PersistedPromptSettings::from(&self.prompt)),
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
            target_language: self.target_language.clone(),
            region_parallelism: self.region_parallelism,
            translation_batch_size: self.translation_batch_size,
            translator_loaded,
            idle_unload_minutes: self.idle_unload_minutes,
            generation: BackendGenerationSettings::from_config(&self.generation),
            memory: BackendMemorySettings::from_config(&self.memory),
            prompt: BackendPromptSettings::from_config(&self.prompt),
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
            target_language: DEFAULT_TARGET_LANGUAGE.to_owned(),
            region_parallelism: DEFAULT_REGION_PARALLELISM,
            translation_batch_size: DEFAULT_TRANSLATION_BATCH_SIZE,
            translator_loaded: false,
            idle_unload_minutes: DEFAULT_IDLE_UNLOAD_MINUTES,
            generation: BackendGenerationSettings::from_config(&GenerationConfig::default()),
            memory: BackendMemorySettings::from_config(&MemoryConfig::default()),
            prompt: BackendPromptSettings::from_config(&PromptConfig::default()),
            message: format!("后端配置无效：{message}"),
        }
    }
}

impl From<&GenerationConfig> for PersistedGenerationSettings {
    fn from(config: &GenerationConfig) -> Self {
        Self {
            max_new_tokens: Some(config.max_new_tokens),
            sampling: Some(config.sampling),
            temperature: Some(config.temperature),
            top_k: Some(config.top_k),
            top_p: Some(config.top_p),
            seed: config.seed.map(|seed| seed.to_string()),
            repetition_penalty: Some(config.repetition_penalty),
            frequency_penalty: Some(config.frequency_penalty),
            stop_tokens: Some(config.stop_tokens.clone()),
            stop_strings: Some(config.stop_strings.clone()),
        }
    }
}

impl From<&MemoryConfig> for PersistedMemorySettings {
    fn from(config: &MemoryConfig) -> Self {
        Self {
            enabled: Some(config.enabled),
            max_tokens: Some(config.max_tokens),
            max_turns: Some(config.max_turns),
        }
    }
}

impl From<&PromptConfig> for PersistedPromptSettings {
    fn from(config: &PromptConfig) -> Self {
        Self {
            system: Some(config.system.clone()),
            user: Some(config.user.clone()),
        }
    }
}

fn generation_from_persisted(
    persisted: Option<PersistedGenerationSettings>,
) -> Result<GenerationConfig, String> {
    match persisted {
        Some(persisted) => persisted.into_config(GenerationConfig::default()),
        None => validate_generation(GenerationConfig::default()),
    }
}

fn memory_from_persisted(
    persisted: Option<PersistedMemorySettings>,
) -> Result<MemoryConfig, String> {
    match persisted {
        Some(persisted) => persisted.into_config(MemoryConfig::default()),
        None => validate_memory(MemoryConfig::default()),
    }
}

fn prompt_from_persisted(
    persisted: Option<PersistedPromptSettings>,
) -> Result<PromptConfig, String> {
    match persisted {
        Some(persisted) => persisted.into_config(PromptConfig::default()),
        None => validate_prompt(PromptConfig::default()),
    }
}

fn validate_generation(config: GenerationConfig) -> Result<GenerationConfig, String> {
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
}

fn validate_memory(config: MemoryConfig) -> Result<MemoryConfig, String> {
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
}

fn validate_prompt(config: PromptConfig) -> Result<PromptConfig, String> {
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
}

fn parse_seed(value: Option<String>) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.starts_with('0') || !value.chars().all(|character| character.is_ascii_digit()) {
        return Err("generation.seed must be empty or a positive u64 decimal string".to_owned());
    }
    value
        .parse::<u64>()
        .map(Some)
        .map_err(|_| "generation.seed must be empty or a positive u64 decimal string".to_owned())
}

fn unique_tokens(values: Vec<u32>) -> Vec<u32> {
    let mut seen = HashSet::with_capacity(values.len());
    values
        .into_iter()
        .filter(|value| seen.insert(*value))
        .collect()
}

fn unique_trimmed_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(values.len());
    values
        .into_iter()
        .map(|value| value.trim().to_owned())
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn validate_target_language(value: &str) -> Result<String, String> {
    let value = value.trim();
    let target_length = value.chars().count();
    if !(1..=MAX_TARGET_LANGUAGE_CHARS).contains(&target_length) {
        return Err(format!(
            "targetLanguage must contain 1..={MAX_TARGET_LANGUAGE_CHARS} characters"
        ));
    }
    Ok(value.to_owned())
}

fn parse_device(value: &str) -> Result<DeviceKind, String> {
    value
        .parse()
        .map_err(|_| "device must be cpu or cuda".to_owned())
}

fn validate_region_parallelism(value: usize) -> Result<(), String> {
    if !(1..=DEFAULT_REGION_PARALLELISM).contains(&value) {
        return Err(format!(
            "regionParallelism must be in 1..={DEFAULT_REGION_PARALLELISM}"
        ));
    }
    Ok(())
}

fn validate_translation_batch_size(value: usize) -> Result<(), String> {
    if !(1..=DEFAULT_TRANSLATION_BATCH_SIZE).contains(&value) {
        return Err(format!(
            "translationBatchSize must be in 1..={DEFAULT_TRANSLATION_BATCH_SIZE}"
        ));
    }
    Ok(())
}

fn validate_idle_unload_minutes(value: u32) -> Result<(), String> {
    if value > MAX_IDLE_UNLOAD_MINUTES {
        return Err(format!(
            "idle_unload_minutes must be in 0..={MAX_IDLE_UNLOAD_MINUTES}"
        ));
    }
    Ok(())
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

fn selected_optional_path(name: &str, value: Option<&str>) -> Result<Option<PathBuf>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    selected_path(name, value).map(Some)
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

fn deserialize_optional_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

fn serialize_optional_option<S, T>(
    value: &Option<Option<T>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    match value {
        Some(value) => value.serialize(serializer),
        None => serializer.serialize_none(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackendGenerationSettings, BackendMemorySettings, BackendPromptSettings, BackendSettings,
        BackendSettingsUpdate, DeviceKind, PersistedBackendSettings, PersistedGenerationSettings,
        generation_from_persisted,
    };
    use crate::model_config::{
        GenerationConfig, MAX_SYSTEM_PROMPT_CHARS, MAX_USER_PROMPT_CHARS, MemoryConfig,
        PromptConfig,
    };
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
            prompt: PromptConfig::default(),
            generation: GenerationConfig::default(),
            memory: MemoryConfig::default(),
        }
    }

    fn full_update() -> BackendSettingsUpdate {
        BackendSettingsUpdate {
            detector_model_dir: "D:\\models\\detector".to_owned(),
            recognizer_model_dir: "D:\\models\\recognizer".to_owned(),
            hy_model: "D:\\models\\hy.gguf".to_owned(),
            font_path: None,
            target_language: "Japanese".to_owned(),
            device: "cuda".to_owned(),
            region_parallelism: 8,
            translation_batch_size: 2,
            idle_unload_minutes: 0,
            generation: BackendGenerationSettings {
                max_new_tokens: 64,
                sampling: true,
                temperature: 0.7,
                top_k: 32,
                top_p: 0.9,
                seed: Some("42".to_owned()),
                repetition_penalty: 1.1,
                frequency_penalty: 0.2,
                stop_tokens: vec![120020, 120020],
                stop_strings: vec!["</s>".to_owned(), "</s>".to_owned()],
            },
            memory: BackendMemorySettings {
                enabled: true,
                max_tokens: 1024,
                max_turns: 4,
            },
            prompt: BackendPromptSettings {
                system: "Return JSON.".to_owned(),
                user: "Preserve product names.".to_owned(),
            },
        }
    }

    #[test]
    fn selected_model_paths_and_full_model_settings_are_updated_together() {
        let updated = settings()
            .update_from_request(full_update())
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
        assert_eq!(updated.font_path, None);
        assert_eq!(updated.target_language, "Japanese");
        assert_eq!(updated.device_kind, DeviceKind::Cuda);
        assert_eq!(updated.region_parallelism, 8);
        assert_eq!(updated.translation_batch_size, 2);
        assert_eq!(updated.idle_unload_minutes, 0);
        assert_eq!(updated.generation.max_new_tokens, 64);
        assert!(updated.generation.sampling);
        assert_eq!(updated.generation.top_k, 32);
        assert_eq!(updated.generation.seed, Some(42));
        assert_eq!(updated.generation.stop_tokens, vec![120020]);
        assert_eq!(updated.generation.stop_strings, vec!["</s>".to_owned()]);
        assert!(updated.memory.enabled);
        assert_eq!(updated.memory.max_tokens, 1024);
        assert_eq!(updated.memory.max_turns, 4);
        assert_eq!(updated.prompt.system, "Return JSON.");
        assert_eq!(updated.prompt.user, "Preserve product names.");
    }

    #[test]
    fn persisted_generation_defaults_are_legacy_safe() {
        let default = generation_from_persisted(None).expect("default generation");
        assert_eq!(default, GenerationConfig::default());

        let legacy: PersistedBackendSettings = serde_json::from_value(serde_json::json!({
            "detectorModelDir": "D:\\models\\detector",
            "recognizerModelDir": "D:\\models\\recognizer",
            "hyModel": "D:\\models\\hy.gguf",
            "idleUnloadMinutes": 30
        }))
        .expect("legacy persisted settings");
        assert!(legacy.generation.is_none());
        assert!(legacy.memory.is_none());
        assert!(legacy.prompt.is_none());
    }

    #[test]
    fn selected_model_settings_reject_invalid_values() {
        let current = settings();

        let mut invalid = full_update();
        invalid.font_path = Some("fonts/font.ttf".to_owned());
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.device = "metal".to_owned();
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.region_parallelism = 0;
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.translation_batch_size = 5;
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.generation.top_k = 0;
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.generation.top_k = 1025;
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.generation.seed = Some("0".to_owned());
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.generation.stop_strings = vec![" ".to_owned()];
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.memory.max_tokens = 0;
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.prompt.system = "x".repeat(MAX_SYSTEM_PROMPT_CHARS + 1);
        assert!(current.update_from_request(invalid).is_err());

        let mut invalid = full_update();
        invalid.prompt.user = "x".repeat(MAX_USER_PROMPT_CHARS + 1);
        assert!(current.update_from_request(invalid).is_err());
    }

    #[test]
    fn persisted_generation_rejects_invalid_sampling_shape() {
        let invalid = PersistedGenerationSettings {
            sampling: Some(true),
            top_k: Some(0),
            ..PersistedGenerationSettings::default()
        };

        assert!(
            generation_from_persisted(Some(invalid)).is_err(),
            "sampling requires top_k"
        );
    }
}
