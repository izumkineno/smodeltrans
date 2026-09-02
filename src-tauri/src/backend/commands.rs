use super::{
    contracts::RegionRecord,
    engine::BackendEngine,
    failure::BackendFailure,
    input::{
        DecodedImage, decode_image, decode_ocr_image, validate_target_language, validate_text,
    },
    settings::{
        BackendSettings, BackendSettingsUpdate, BackendStatus, ModelCatalogOptions,
        ModelCatalogUpdate,
    },
};

use crate::model_support::{RunId, RunRegistry, lock_with_cancellation};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tauri::{Emitter, State};
#[derive(Clone)]
pub(crate) struct BackendState {
    pub(crate) settings: Arc<Mutex<Result<BackendSettings, String>>>,
    pub(crate) engine: Arc<Mutex<Option<BackendEngine>>>,
    pub(crate) last_activity: Arc<Mutex<Instant>>,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) runs: RunRegistry,
    pub(crate) live_active: Arc<AtomicBool>,
}

impl BackendState {
    pub(crate) fn new_with_resource_root_and_config(
        resource_root: Option<PathBuf>,
        config_path: Option<PathBuf>,
    ) -> Self {
        Self {
            settings: Arc::new(Mutex::new(
                BackendSettings::from_environment_with_resource_root_and_config(
                    resource_root,
                    config_path.as_deref(),
                ),
            )),
            engine: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            config_path,
            runs: RunRegistry::default(),
            live_active: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn start_idle_monitor(&self) {
        let settings = Arc::clone(&self.settings);
        let engine = Arc::clone(&self.engine);
        let last_activity = Arc::clone(&self.last_activity);
        let live_active = Arc::clone(&self.live_active);
        thread::spawn(move || {
            tracing::info!(target: "backend::commands", "idle monitor started");
            loop {
                thread::sleep(Duration::from_secs(1));
                if live_active.load(Ordering::SeqCst) {
                    continue;
                }
                let idle_seconds = settings
                    .lock()
                    .ok()
                    .and_then(|settings| {
                        settings
                            .as_ref()
                            .ok()
                            .map(|settings| settings.idle_unload_seconds)
                    })
                    .unwrap_or(0);
                if idle_seconds == 0 {
                    continue;
                }
                let idle_for = Duration::from_secs(u64::from(idle_seconds));
                let last_used = last_activity
                    .lock()
                    .map(|last_activity| *last_activity)
                    .unwrap_or_else(|_| Instant::now());
                if last_used.elapsed() < idle_for {
                    continue;
                }
                let Ok(mut engine) = engine.try_lock() else {
                    tracing::debug!(
                        target: "backend::commands",
                        idle_seconds,
                        "idle check skipped: engine lock busy"
                    );
                    continue;
                };
                if last_used.elapsed() >= idle_for
                    && let Some(engine) = engine.as_mut()
                {
                    let (ocr_loaded, translator_loaded) = engine.model_states();
                    if ocr_loaded || translator_loaded {
                        let elapsed_secs = last_used.elapsed().as_secs();
                        tracing::info!(
                            target: "backend::commands",
                            idle_seconds,
                            elapsed_secs,
                            ocr_loaded,
                            translator_loaded,
                            "idle timeout reached, unloading models"
                        );
                        engine.unload_models();
                        tracing::info!(
                            target: "backend::commands",
                            idle_seconds,
                            ocr_loaded,
                            translator_loaded,
                            "idle models unloaded"
                        );
                    } else {
                        tracing::trace!(
                            target: "backend::commands",
                            idle_seconds,
                            "idle timeout reached but no models loaded"
                        );
                    }
                }
            }
        });
    }

    pub(crate) fn touch_activity(&self) {
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = Instant::now();
        }
    }

    pub(crate) fn model_states(&self) -> Result<(bool, bool), BackendFailure> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| BackendFailure::internal("后端状态锁已损坏"))?;
        Ok(engine
            .as_ref()
            .map(BackendEngine::model_states)
            .unwrap_or((false, false)))
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TranslateImageRequest {
    pub(crate) image_base64: String,
    pub(crate) file_name: String,
    pub(crate) target_language: String,
    #[serde(default)]
    pub(crate) request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TranslateTextRequest {
    pub(crate) text: String,
    pub(crate) target_language: String,
    #[serde(default)]
    pub(crate) request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OcrImageRequest {
    pub(crate) image_base64: String,
    pub(crate) file_name: String,
    #[serde(default)]
    pub(crate) request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CancelTranslationRequest {
    pub(crate) request_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelControlRequest {
    pub(crate) model: String,
    pub(crate) action: String,
}

#[derive(Clone, Copy, Debug)]
enum ModelTarget {
    Ocr,
    Translator,
}

impl ModelTarget {
    fn parse(value: &str) -> Result<Self, BackendFailure> {
        match value {
            "ocr" => Ok(Self::Ocr),
            "translator" => Ok(Self::Translator),
            _ => Err(BackendFailure::arguments(
                "model must be 'ocr' or 'translator'",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ModelAction {
    Load,
    Unload,
}

impl ModelAction {
    fn parse(value: &str) -> Result<Self, BackendFailure> {
        match value {
            "load" => Ok(Self::Load),
            "unload" => Ok(Self::Unload),
            _ => Err(BackendFailure::arguments(
                "action must be 'load' or 'unload'",
            )),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TranslationResponse {
    pub(crate) text: String,
    pub(crate) markdown: String,
    pub(crate) annotated_image_data_url: String,
    pub(crate) provider_label: String,
    pub(crate) is_translated: bool,
    pub(crate) duration_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TextTranslationResponse {
    pub(crate) text: String,
    pub(crate) provider_label: String,
    pub(crate) duration_ms: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OcrCharacterResponse {
    pub(crate) order: u32,
    pub(crate) quad: [[i32; 2]; 4],
    pub(crate) recognized_text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OcrRegionResponse {
    pub(crate) order: u32,
    pub(crate) quad: [[i32; 2]; 4],
    pub(crate) recognized_text: String,
    pub(crate) char_boxes: Vec<OcrCharacterResponse>,
}

impl From<RegionRecord> for OcrRegionResponse {
    fn from(region: RegionRecord) -> Self {
        Self {
            order: region.order,
            quad: region.quad_points,
            recognized_text: region.source_text,
            char_boxes: region
                .characters
                .into_iter()
                .map(|character| OcrCharacterResponse {
                    order: character.order,
                    quad: character.quad_points,
                    recognized_text: character.source_text,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OcrResponse {
    pub(crate) text: String,
    pub(crate) markdown: String,
    pub(crate) annotated_image_data_url: String,
    pub(crate) provider_label: String,
    pub(crate) duration_ms: u64,
    pub(crate) image_width: u32,
    pub(crate) image_height: u32,
    pub(crate) regions: Vec<OcrRegionResponse>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TranslationProgressEvent {
    request_id: String,
    progress: u8,
    stage: String,
}

fn emit_translation_progress(
    app: &tauri::AppHandle,
    run_id: &RunId,
    progress: u8,
    stage: &'static str,
) {
    tracing::debug!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        progress,
        stage,
        "translation progress"
    );
    let _ = app.emit(
        "translation-progress",
        TranslationProgressEvent {
            request_id: run_id.as_str().to_owned(),
            progress,
            stage: stage.to_owned(),
        },
    );
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BackendError {
    pub(crate) code: String,
    pub(crate) message: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelRuntimeStatus {
    pub(crate) backend: BackendStatus,
    pub(crate) ocr_loaded: bool,
    pub(crate) translator_loaded: bool,
    pub(crate) busy: bool,
}

impl From<BackendFailure> for BackendError {
    fn from(failure: BackendFailure) -> Self {
        Self {
            code: failure.code().as_str().to_owned(),
            message: failure.message().to_owned(),
        }
    }
}

fn current_backend_status(state: &BackendState) -> Result<BackendStatus, BackendError> {
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone();
    let settings = match settings {
        Ok(settings) => settings,
        Err(message) => return Ok(BackendStatus::configuration_error(&message)),
    };
    let translator_loaded = state.model_states()?.1;
    Ok(settings.status(translator_loaded))
}

fn current_model_runtime_status(state: &BackendState) -> Result<ModelRuntimeStatus, BackendError> {
    let (ocr_loaded, translator_loaded) = state.model_states()?;
    Ok(ModelRuntimeStatus {
        backend: current_backend_status(state)?,
        ocr_loaded,
        translator_loaded,
        busy: state.runs.is_busy()? || state.live_active.load(Ordering::SeqCst),
    })
}

fn ensure_live_inactive(state: &BackendState) -> Result<(), BackendFailure> {
    if state.live_active.load(Ordering::SeqCst) {
        return Err(BackendFailure::arguments(
            "实时翻译正在运行，请先停止实时会话",
        ));
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_backend_status(
    state: State<'_, BackendState>,
) -> Result<BackendStatus, BackendError> {
    let span = tracing::info_span!(target: "backend::commands", "get_backend_status");
    let _enter = span.enter();
    tracing::debug!(target: "backend::commands", "get_backend_status called");
    let result = current_backend_status(state.inner());
    match &result {
        Ok(status) => tracing::info!(
            target: "backend::commands",
            ready = status.ready,
            has_message = !status.message.is_empty(),
            "get_backend_status succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            code = %err.code,
            message = %err.message,
            "get_backend_status failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) fn get_model_runtime_status(
    state: State<'_, BackendState>,
) -> Result<ModelRuntimeStatus, BackendError> {
    let span = tracing::info_span!(target: "backend::commands", "get_model_runtime_status");
    let _enter = span.enter();
    tracing::debug!(target: "backend::commands", "get_model_runtime_status called");
    let result = current_model_runtime_status(state.inner());
    match &result {
        Ok(status) => tracing::debug!(
            target: "backend::commands",
            ocr_loaded = status.ocr_loaded,
            translator_loaded = status.translator_loaded,
            busy = status.busy,
            ready = status.backend.ready,
            "get_model_runtime_status succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            code = %err.code,
            message = %err.message,
            "get_model_runtime_status failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) fn update_backend_settings(
    request: BackendSettingsUpdate,
    state: State<'_, BackendState>,
) -> Result<BackendStatus, BackendError> {
    let span = tracing::info_span!(target: "backend::commands", "update_backend_settings");
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        detector_model_dir = %request.detector_model_dir,
        recognizer_model_dir = %request.recognizer_model_dir,
        hy_model = %request.hy_model,
        target_language = %request.target_language,
        device = %request.device,
        region_parallelism = request.region_parallelism,
        translation_batch_size = request.translation_batch_size,
        idle_unload_seconds = request.idle_unload_seconds,
        "update_backend_settings requested"
    );
    tracing::debug!(
        target: "backend::commands",
        font_path = ?request.font_path,
        generation = ?request.generation,
        memory = ?request.memory,
        prompt_template_len = request.prompt.template.chars().count(),
        "update_backend_settings params"
    );
    let result: Result<BackendStatus, BackendError> = (|| {
        ensure_live_inactive(state.inner())?;
        let current = state
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        let updated = current
            .update_from_request(request)
            .map_err(BackendFailure::arguments)?;
        if let Some(config_path) = state.config_path.as_deref() {
            tracing::debug!(
                target: "backend::commands",
                config_path = %config_path.display(),
                "persisting backend settings"
            );
            persist_backend_settings(config_path, &updated)?;
        }
        {
            let mut settings = state
                .settings
                .lock()
                .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?;
            *settings = Ok(updated.clone());
        }
        let mut engine = state
            .engine
            .lock()
            .map_err(|_| BackendFailure::internal("后端状态锁已损坏"))?;
        *engine = None;
        drop(engine);
        state.touch_activity();
        Ok(updated.status(false))
    })();
    match &result {
        Ok(status) => tracing::info!(
            target: "backend::commands",
            ready = status.ready,
            "update_backend_settings succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            code = %err.code,
            message = %err.message,
            "update_backend_settings failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) fn list_model_catalog(
    state: State<'_, BackendState>,
) -> Result<ModelCatalogOptions, BackendError> {
    let span = tracing::info_span!(target: "backend::commands", "list_model_catalog");
    let _enter = span.enter();
    tracing::debug!(target: "backend::commands", "list_model_catalog called");
    let result: Result<ModelCatalogOptions, BackendError> = (|| {
        let settings = state
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        Ok(settings.catalog_options())
    })();
    match &result {
        Ok(options) => tracing::info!(
            target: "backend::commands",
            translation_count = options.translation.len(),
            ocr_count = options.ocr.len(),
            font_count = options.fonts.len(),
            "list_model_catalog succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            code = %err.code,
            message = %err.message,
            "list_model_catalog failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) fn save_model_catalog(
    request: ModelCatalogUpdate,
    state: State<'_, BackendState>,
) -> Result<BackendStatus, BackendError> {
    let span = tracing::info_span!(target: "backend::commands", "save_model_catalog");
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        translation_entries = request.translation.len(),
        ocr_entries = request.ocr.len(),
        font_entries = request.fonts.len(),
        "save_model_catalog requested"
    );
    tracing::debug!(
        target: "backend::commands",
        request = ?request,
        "save_model_catalog params"
    );
    let result: Result<BackendStatus, BackendError> = (|| {
        let mut settings = state
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?;
        let mut current = settings.clone().map_err(BackendFailure::arguments)?;
        current
            .save_catalog(request)
            .map_err(BackendFailure::arguments)?;
        if let Some(config_path) = state.config_path.as_deref() {
            tracing::debug!(
                target: "backend::commands",
                config_path = %config_path.display(),
                "persisting catalog settings"
            );
            persist_backend_settings(config_path, &current)?;
        }
        *settings = Ok(current);
        drop(settings);
        state.touch_activity();
        current_backend_status(state.inner())
    })();
    match &result {
        Ok(status) => tracing::info!(
            target: "backend::commands",
            ready = status.ready,
            "save_model_catalog succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            code = %err.code,
            message = %err.message,
            "save_model_catalog failed"
        ),
    }
    result
}

fn persist_backend_settings(
    config_path: &Path,
    settings: &BackendSettings,
) -> Result<(), BackendFailure> {
    let parent = config_path
        .parent()
        .ok_or_else(|| BackendFailure::internal("模型设置路径无效"))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| BackendFailure::internal(format!("创建模型设置目录失败：{error}")))?;
    let content = serde_json::to_vec_pretty(&settings.persisted())
        .map_err(|error| BackendFailure::internal(format!("序列化模型设置失败：{error}")))?;
    std::fs::write(config_path, content)
        .map_err(|error| BackendFailure::internal(format!("保存模型设置失败：{error}")))?;
    Ok(())
}

#[tauri::command]
pub(crate) async fn control_model(
    request: ModelControlRequest,
    state: State<'_, BackendState>,
) -> Result<ModelRuntimeStatus, BackendError> {
    let span = tracing::info_span!(
        target: "backend::commands",
        "control_model",
        model = %request.model,
        action = %request.action
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        model = %request.model,
        action = %request.action,
        "control_model requested"
    );
    let started_at = Instant::now();
    let model_raw = request.model.clone();
    let action_raw = request.action.clone();
    let result: Result<ModelRuntimeStatus, BackendError> = async move {
        ensure_live_inactive(state.inner())?;
        let target = ModelTarget::parse(request.model.trim())?;
        let action = ModelAction::parse(request.action.trim())?;
        tracing::debug!(
            target: "backend::commands",
            model = ?target,
            action = ?action,
            "control_model parsed"
        );
        let worker_state = state.inner().clone();
        let result = tauri::async_runtime::spawn_blocking(move || -> Result<(), BackendFailure> {
            tracing::info!(
                target: "backend::commands",
                model = ?target,
                action = ?action,
                "control_model worker started"
            );
            if worker_state.runs.is_busy()? {
                tracing::warn!(
                    target: "backend::commands",
                    model = ?target,
                    action = ?action,
                    "control_model rejected: busy"
                );
                return Err(BackendFailure::arguments(
                    "模型正在处理请求，请在当前任务完成后重试",
                ));
            }
            let settings = worker_state
                .settings
                .lock()
                .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
                .clone()
                .map_err(BackendFailure::arguments)?;
            let mut engine = worker_state
                .engine
                .lock()
                .map_err(|_| BackendFailure::internal("后端状态锁已损坏"))?;
            if matches!(action, ModelAction::Load) && engine.is_none() {
                tracing::info!(target: "backend::commands", "control_model initializing backend engine");
                *engine = Some(BackendEngine::new(settings)?);
            }
            let result = match (engine.as_mut(), target, action) {
                (Some(engine), ModelTarget::Ocr, ModelAction::Load) => engine.load_ocr(),
                (Some(engine), ModelTarget::Translator, ModelAction::Load) => {
                    let target_language = engine.settings.target_language.clone();
                    tracing::debug!(
                        target: "backend::commands",
                        target_language = %target_language,
                        "control_model loading translator"
                    );
                    engine.load_translator(&target_language)
                }
                (Some(engine), ModelTarget::Ocr, ModelAction::Unload) => {
                    engine.unload_ocr();
                    Ok(())
                }
                (Some(engine), ModelTarget::Translator, ModelAction::Unload) => {
                    engine.unload_translator();
                    Ok(())
                }
                (None, _, ModelAction::Unload) => Ok(()),
                (None, _, ModelAction::Load) => Err(BackendFailure::internal("Candle 后端未初始化")),
            };
            drop(engine);
            worker_state.touch_activity();
            match &result {
                Ok(()) => tracing::info!(
                    target: "backend::commands",
                    model = ?target,
                    action = ?action,
                    "control_model worker succeeded"
                ),
                Err(err) => tracing::error!(
                    target: "backend::commands",
                    model = ?target,
                    action = ?action,
                    error = %err,
                    code = %err.code().as_str(),
                    "control_model worker failed"
                ),
            }
            result
        })
        .await
        .map_err(|error| {
            BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
        })
        .and_then(|result| result);
        result?;
        current_model_runtime_status(state.inner())
    }
    .await;
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(status) => tracing::info!(
            target: "backend::commands",
            model = %model_raw,
            action = %action_raw,
            duration_ms,
            ocr_loaded = status.ocr_loaded,
            translator_loaded = status.translator_loaded,
            busy = status.busy,
            "control_model succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            model = %model_raw,
            action = %action_raw,
            duration_ms,
            code = %err.code,
            message = %err.message,
            "control_model failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) fn cancel_translation(
    request: CancelTranslationRequest,
    state: State<'_, BackendState>,
) -> Result<(), BackendError> {
    let span = tracing::info_span!(
        target: "backend::commands",
        "cancel_translation",
        request_id = %request.request_id
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %request.request_id,
        "cancel_translation requested"
    );
    let result: Result<(), BackendError> = (|| {
        let run_id = RunId::parse(&request.request_id)?;
        state.runs.cancel(&run_id)?;
        Ok(())
    })();
    match &result {
        Ok(()) => tracing::info!(
            target: "backend::commands",
            request_id = %request.request_id,
            "cancel_translation succeeded"
        ),
        Err(err) => tracing::warn!(
            target: "backend::commands",
            request_id = %request.request_id,
            code = %err.code,
            message = %err.message,
            "cancel_translation failed"
        ),
    }
    result
}
#[tauri::command]
pub(crate) async fn translate_image(
    app: tauri::AppHandle,
    request: TranslateImageRequest,
    state: State<'_, BackendState>,
) -> Result<TranslationResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let file_name = request.file_name.clone();
    let target_language = request.target_language.clone();
    let image_base64_len = request.image_base64.len();
    let span = tracing::info_span!(
        target: "backend::commands",
        "translate_image",
        request_id = %run_id.as_str(),
        file_name = %file_name,
        target_language = %target_language
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        file_name = %file_name,
        target_language = %target_language,
        image_base64_len,
        "translate_image started"
    );
    let run_id_for_task = run_id.clone();
    let result: Result<TranslationResponse, BackendError> = async move {
        let run_id = run_id_for_task;
        ensure_live_inactive(state.inner())?;
        let lease = state.runs.register(run_id.clone())?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            "translate_image lease registered"
        );
        emit_translation_progress(&app, &run_id, 5, "正在准备翻译请求");
        let decoded = decode_request(request).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                code = %err.code().as_str(),
                "translate_image decode failed"
            );
            err
        })?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            file_name = %decoded.file_name(),
            target_language = %decoded.target_language(),
            width = decoded.canvas().width(),
            height = decoded.canvas().height(),
            "translate_image decoded"
        );
        emit_translation_progress(&app, &run_id, 15, "图片已解码");
        let worker_state = state.inner().clone();
        let app_clone = app.clone();
        let run_id_clone = run_id.clone();
        let result = tauri::async_runtime::spawn_blocking(
            move || -> Result<TranslationResponse, BackendFailure> {
                tracing::debug!(
                    target: "backend::commands",
                    request_id = %run_id_clone.as_str(),
                    "translate_image worker started"
                );
                lease.token().check()?;
                let response = translate_image_blocking(
                    decoded,
                    &worker_state,
                    lease.token(),
                    &app_clone,
                    &run_id_clone,
                    started_at,
                )?;
                lease.finalize_success()?;
                Ok(response)
            },
        )
        .await
        .map_err(|error| {
            tracing::error!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %error,
                "translate_image worker panicked"
            );
            BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
        })
        .and_then(|result| result);
        result.map_err(BackendError::from)
    }
    .await;
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(response) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            is_translated = response.is_translated,
            text_len = response.text.chars().count(),
            provider_label = %response.provider_label,
            "translate_image succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            code = %err.code,
            message = %err.message,
            "translate_image failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) async fn translate_text(
    app: tauri::AppHandle,
    request: TranslateTextRequest,
    state: State<'_, BackendState>,
) -> Result<TextTranslationResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let target_language_raw = request.target_language.clone();
    let text_len = request.text.chars().count();
    let span = tracing::info_span!(
        target: "backend::commands",
        "translate_text",
        request_id = %run_id.as_str(),
        target_language = %target_language_raw
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        target_language = %target_language_raw,
        text_len,
        "translate_text started"
    );
    let run_id_for_task = run_id.clone();
    let result: Result<TextTranslationResponse, BackendError> = async move {
        let run_id = run_id_for_task;
        ensure_live_inactive(state.inner())?;
        let lease = state.runs.register(run_id.clone())?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            "translate_text lease registered"
        );
        emit_translation_progress(&app, &run_id, 5, "正在准备翻译请求");
        let (text, target_language) = decode_text_request(request).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                code = %err.code().as_str(),
                "translate_text decode failed"
            );
            err
        })?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            target_language = %target_language,
            text_len = text.chars().count(),
            "translate_text decoded"
        );
        emit_translation_progress(&app, &run_id, 15, "文本请求已验证");
        let worker_state = state.inner().clone();
        let app_clone = app.clone();
        let run_id_clone = run_id.clone();
        let result = tauri::async_runtime::spawn_blocking(
            move || -> Result<TextTranslationResponse, BackendFailure> {
                tracing::debug!(
                    target: "backend::commands",
                    request_id = %run_id_clone.as_str(),
                    "translate_text worker started"
                );
                lease.token().check()?;
                let response = translate_text_blocking(
                    text,
                    target_language,
                    &worker_state,
                    lease.token(),
                    &app_clone,
                    &run_id_clone,
                    started_at,
                )?;
                lease.finalize_success()?;
                Ok(response)
            },
        )
        .await
        .map_err(|error| {
            tracing::error!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %error,
                "translate_text worker panicked"
            );
            BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
        })
        .and_then(|result| result);
        result.map_err(BackendError::from)
    }
    .await;
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(response) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            text_len = response.text.chars().count(),
            provider_label = %response.provider_label,
            "translate_text succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            code = %err.code,
            message = %err.message,
            "translate_text failed"
        ),
    }
    result
}

#[tauri::command]
pub(crate) async fn ocr_image(
    app: tauri::AppHandle,
    request: OcrImageRequest,
    state: State<'_, BackendState>,
) -> Result<OcrResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let file_name = request.file_name.clone();
    let image_base64_len = request.image_base64.len();
    let span = tracing::info_span!(
        target: "backend::commands",
        "ocr_image",
        request_id = %run_id.as_str(),
        file_name = %file_name
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        file_name = %file_name,
        image_base64_len,
        "ocr_image started"
    );
    let run_id_for_task = run_id.clone();
    let result: Result<OcrResponse, BackendError> = async move {
        let run_id = run_id_for_task;
        ensure_live_inactive(state.inner())?;
        let lease = state.runs.register(run_id.clone())?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            "ocr_image lease registered"
        );
        emit_translation_progress(&app, &run_id, 5, "正在准备 OCR 请求");
        let decoded = decode_ocr_request(request).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                code = %err.code().as_str(),
                "ocr_image decode failed"
            );
            err
        })?;
        tracing::debug!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            file_name = %decoded.file_name(),
            width = decoded.canvas().width(),
            height = decoded.canvas().height(),
            "ocr_image decoded"
        );
        emit_translation_progress(&app, &run_id, 15, "图片已解码");
        let worker_state = state.inner().clone();
        let app_clone = app.clone();
        let run_id_clone = run_id.clone();
        let result =
            tauri::async_runtime::spawn_blocking(move || -> Result<OcrResponse, BackendFailure> {
                tracing::debug!(
                    target: "backend::commands",
                    request_id = %run_id_clone.as_str(),
                    "ocr_image worker started"
                );
                lease.token().check()?;
                let response = ocr_image_blocking(
                    decoded,
                    &worker_state,
                    lease.token(),
                    &app_clone,
                    &run_id_clone,
                    started_at,
                )?;
                lease.finalize_success()?;
                Ok(response)
            })
            .await
            .map_err(|error| {
                tracing::error!(
                    target: "backend::commands",
                    request_id = %run_id.as_str(),
                    error = %error,
                    "ocr_image worker panicked"
                );
                BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
            })
            .and_then(|result| result);
        result.map_err(BackendError::from)
    }
    .await;
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(response) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            text_len = response.text.chars().count(),
            region_count = response.regions.len(),
            provider_label = %response.provider_label,
            "ocr_image succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            code = %err.code,
            message = %err.message,
            "ocr_image failed"
        ),
    }
    result
}

fn decode_request(request: TranslateImageRequest) -> Result<DecodedImage, BackendFailure> {
    decode_image(
        &request.image_base64,
        request.file_name,
        request.target_language,
    )
}

fn decode_text_request(request: TranslateTextRequest) -> Result<(String, String), BackendFailure> {
    let text = validate_text(&request.text)?;
    let target_language = validate_target_language(&request.target_language)?;
    Ok((text, target_language))
}

fn decode_ocr_request(request: OcrImageRequest) -> Result<DecodedImage, BackendFailure> {
    decode_ocr_image(&request.image_base64, request.file_name)
}

fn translate_image_blocking(
    request: DecodedImage,
    state: &BackendState,
    cancellation: &crate::model_support::CancellationToken,
    app: &tauri::AppHandle,
    run_id: &RunId,
    started_at: Instant,
) -> Result<TranslationResponse, BackendFailure> {
    let span = tracing::info_span!(
        target: "backend::commands",
        "translate_image_blocking",
        request_id = %run_id.as_str(),
        file_name = %request.file_name(),
        target_language = %request.target_language()
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        file_name = %request.file_name(),
        target_language = %request.target_language(),
        width = request.canvas().width(),
        height = request.canvas().height(),
        "translate_image_blocking started"
    );
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    tracing::debug!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        target_language = %settings.target_language,
        device = %settings.device_kind.as_str(),
        region_parallelism = settings.region_parallelism,
        translation_batch_size = settings.translation_batch_size,
        "translate_image_blocking settings loaded"
    );
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                "translate_image_blocking engine lock failed / cancelled"
            );
            err
        })?;
        if engine.is_none() {
            tracing::info!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                "translate_image_blocking initializing engine"
            );
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.translate(&request, cancellation, |progress, stage| {
            tracing::trace!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                progress,
                stage,
                "translate_image progress callback"
            );
            emit_translation_progress(app, run_id, progress, stage);
        })
    };
    state.touch_activity();
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(res) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            is_translated = res.is_translated,
            provider_label = %res.provider_label,
            text_len = res.text.chars().count(),
            "translate_image_blocking engine succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            error = %err,
            code = %err.code().as_str(),
            "translate_image_blocking engine failed"
        ),
    }
    let result = result?;
    let image_base64 = BASE64.encode(result.annotated_png);
    let response = TranslationResponse {
        text: result.text,
        markdown: result.markdown,
        annotated_image_data_url: format!("data:image/png;base64,{image_base64}"),
        provider_label: result.provider_label,
        is_translated: result.is_translated,
        duration_ms: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
    };
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        duration_ms = response.duration_ms,
        "translate_image_blocking completed"
    );
    Ok(response)
}

fn translate_text_blocking(
    text: String,
    target_language: String,
    state: &BackendState,
    cancellation: &crate::model_support::CancellationToken,
    app: &tauri::AppHandle,
    run_id: &RunId,
    started_at: Instant,
) -> Result<TextTranslationResponse, BackendFailure> {
    let span = tracing::info_span!(
        target: "backend::commands",
        "translate_text_blocking",
        request_id = %run_id.as_str(),
        target_language = %target_language
    );
    let _enter = span.enter();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        target_language = %target_language,
        text_len = text.chars().count(),
        "translate_text_blocking started"
    );
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    tracing::debug!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        settings_target_language = %settings.target_language,
        device = %settings.device_kind.as_str(),
        "translate_text_blocking settings loaded"
    );
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                "translate_text_blocking engine lock failed / cancelled"
            );
            err
        })?;
        if engine.is_none() {
            tracing::info!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                "translate_text_blocking initializing engine"
            );
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.translate_text(
            &text,
            &target_language,
            "",
            cancellation,
            |progress, stage| {
                tracing::trace!(
                    target: "backend::commands",
                    request_id = %run_id.as_str(),
                    progress,
                    stage,
                    "translate_text progress callback"
                );
                emit_translation_progress(app, run_id, progress, stage);
            },
            |_| {},
        )
    };
    state.touch_activity();
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(translated) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            text_len = translated.chars().count(),
            "translate_text_blocking engine succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            error = %err,
            code = %err.code().as_str(),
            "translate_text_blocking engine failed"
        ),
    }
    let translated = result?;
    let response = TextTranslationResponse {
        text: translated,
        provider_label: "Hy-MT2 / Candle".to_owned(),
        duration_ms: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
    };
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        duration_ms = response.duration_ms,
        "translate_text_blocking completed"
    );
    Ok(response)
}
fn ocr_image_blocking(
    request: DecodedImage,
    state: &BackendState,
    cancellation: &crate::model_support::CancellationToken,
    app: &tauri::AppHandle,
    run_id: &RunId,
    started_at: Instant,
) -> Result<OcrResponse, BackendFailure> {
    let span = tracing::info_span!(
        target: "backend::commands",
        "ocr_image_blocking",
        request_id = %run_id.as_str(),
        file_name = %request.file_name()
    );
    let _enter = span.enter();
    let (image_width, image_height) = request.canvas().dimensions();
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        file_name = %request.file_name(),
        image_width,
        image_height,
        "ocr_image_blocking started"
    );
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    tracing::debug!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        device = %settings.device_kind.as_str(),
        region_parallelism = settings.region_parallelism,
        "ocr_image_blocking settings loaded"
    );
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation).map_err(|err| {
            tracing::warn!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                error = %err,
                "ocr_image_blocking engine lock failed / cancelled"
            );
            err
        })?;
        if engine.is_none() {
            tracing::info!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                "ocr_image_blocking initializing engine"
            );
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.ocr(&request, cancellation, |progress, stage| {
            tracing::trace!(
                target: "backend::commands",
                request_id = %run_id.as_str(),
                progress,
                stage,
                "ocr_image progress callback"
            );
            emit_translation_progress(app, run_id, progress, stage);
        })
    };
    state.touch_activity();
    let duration_ms = started_at.elapsed().as_millis() as u64;
    match &result {
        Ok(res) => tracing::info!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            region_count = res.regions.len(),
            provider_label = %res.provider_label,
            "ocr_image_blocking engine succeeded"
        ),
        Err(err) => tracing::error!(
            target: "backend::commands",
            request_id = %run_id.as_str(),
            duration_ms,
            error = %err,
            code = %err.code().as_str(),
            "ocr_image_blocking engine failed"
        ),
    }
    let result = result?;
    let image_base64 = BASE64.encode(result.annotated_png);
    let response = OcrResponse {
        text: result.text,
        markdown: result.markdown,
        annotated_image_data_url: format!("data:image/png;base64,{image_base64}"),
        provider_label: result.provider_label,
        duration_ms: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        image_width,
        image_height,
        regions: result
            .regions
            .into_iter()
            .map(OcrRegionResponse::from)
            .collect(),
    };
    tracing::info!(
        target: "backend::commands",
        request_id = %run_id.as_str(),
        duration_ms = response.duration_ms,
        region_count = response.regions.len(),
        "ocr_image_blocking completed"
    );
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::{
        BackendSettingsUpdate, OcrImageRequest, OcrRegionResponse, TranslateImageRequest,
        TranslateTextRequest,
    };
    use crate::backend::contracts::{CharacterRecord, RegionRecord};
    use std::path::PathBuf;
    #[test]
    fn old_translate_request_without_request_id_still_deserializes() {
        let request: TranslateImageRequest = serde_json::from_value(serde_json::json!({
            "imageBase64": "AAAA",
            "fileName": "sample.png",
            "targetLanguage": "Chinese"
        }))
        .expect("old request shape");
        assert!(request.request_id.is_none());
    }

    #[test]
    fn text_request_uses_camel_case_and_defaults_request_id() {
        let request: TranslateTextRequest = serde_json::from_value(serde_json::json!({
            "text": "你好",
            "targetLanguage": "English"
        }))
        .expect("text request shape");
        assert_eq!(request.text, "你好");
        assert_eq!(request.target_language, "English");
        assert!(request.request_id.is_none());
    }

    #[test]
    fn ocr_request_uses_camel_case_and_defaults_request_id() {
        let request: OcrImageRequest = serde_json::from_value(serde_json::json!({
            "imageBase64": "AAAA",
            "fileName": "sample.png"
        }))
        .expect("OCR request shape");
        assert_eq!(request.image_base64, "AAAA");
        assert_eq!(request.file_name, "sample.png");
        assert!(request.request_id.is_none());
    }

    #[test]
    fn update_backend_settings_request_deserializes_full_model_parameters() {
        let request: BackendSettingsUpdate = serde_json::from_value(serde_json::json!({
            "detectorModelDir": "D:\\models\\detector",
            "recognizerModelDir": "D:\\models\\recognizer",
            "hyModel": "D:\\models\\hy.gguf",
            "fontPath": null,
            "targetLanguage": "Japanese",
            "device": "cuda",
            "regionParallelism": 8,
            "translationBatchSize": 2,
            "idleUnloadSeconds": 0,
            "generation": {
                "maxNewTokens": 64,
                "sampling": true,
                "temperature": 0.7,
                "topK": 32,
                "topP": 0.9,
                "seed": "42",
                "repetitionPenalty": 1.1,
                "frequencyPenalty": 0.2,
                "stopTokens": [120020],
                "stopStrings": ["</s>"]
            },
            "memory": {
                "enabled": true,
                "maxTokens": 1024,
                "maxTurns": 4
            },
            "prompt": {
                "template": "Preserve product names."
            }
        }))
        .expect("full settings request");

        assert_eq!(
            request.detector_model_dir,
            PathBuf::from("D:\\models\\detector").display().to_string()
        );
        assert_eq!(request.font_path, None);
        assert_eq!(request.generation.seed, Some("42".to_owned()));
        assert_eq!(request.generation.stop_tokens, vec![120020]);
        assert!(request.memory.enabled);
        assert_eq!(request.prompt.template, "Preserve product names.");
    }
    #[test]
    fn ocr_region_response_serializes_selectable_character_boxes() {
        let response = OcrRegionResponse::from(RegionRecord {
            order: 1,
            quad_points: [[0, 0], [20, 0], [20, 10], [0, 10]],
            source_text: "AB".to_owned(),
            confidence_milli: 900,
            translated_text: String::new(),
            characters: vec![CharacterRecord {
                order: 1,
                quad_points: [[0, 0], [10, 0], [10, 10], [0, 10]],
                source_text: "A".to_owned(),
            }],
        });

        let value = serde_json::to_value(response).unwrap();

        assert_eq!(value["recognizedText"], "AB");
        assert_eq!(value["charBoxes"][0]["recognizedText"], "A");
        assert_eq!(
            value["charBoxes"][0]["quad"][2],
            serde_json::json!([10, 10])
        );
    }
}
