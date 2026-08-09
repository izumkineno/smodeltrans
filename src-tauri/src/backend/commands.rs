use super::{
    contracts::RegionRecord,
    engine::BackendEngine,
    failure::BackendFailure,
    input::{
        DecodedImage, decode_image, decode_ocr_image, validate_target_language, validate_text,
    },
    settings::{BackendSettings, BackendSettingsUpdate, BackendStatus},
};

use crate::model_support::{RunId, RunRegistry, lock_with_cancellation};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
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
}

impl BackendState {
    pub(crate) fn new() -> Self {
        Self::new_with_resource_root_and_config(None, None)
    }

    pub(crate) fn new_with_resource_root(resource_root: Option<PathBuf>) -> Self {
        Self::new_with_resource_root_and_config(resource_root, None)
    }

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
        }
    }

    pub(crate) fn start_idle_monitor(&self) {
        let settings = Arc::clone(&self.settings);
        let engine = Arc::clone(&self.engine);
        let last_activity = Arc::clone(&self.last_activity);
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(5));
                let idle_minutes = settings
                    .lock()
                    .ok()
                    .and_then(|settings| {
                        settings
                            .as_ref()
                            .ok()
                            .map(|settings| settings.idle_unload_minutes)
                    })
                    .unwrap_or(0);
                if idle_minutes == 0 {
                    continue;
                }
                let idle_for = Duration::from_secs(u64::from(idle_minutes) * 60);
                let last_used = last_activity
                    .lock()
                    .map(|last_activity| *last_activity)
                    .unwrap_or_else(|_| Instant::now());
                if last_used.elapsed() < idle_for {
                    continue;
                }
                let Ok(mut engine) = engine.try_lock() else {
                    continue;
                };
                if last_used.elapsed() >= idle_for {
                    if let Some(engine) = engine.as_mut() {
                        engine.unload_models();
                    }
                }
            }
        });
    }

    fn touch_activity(&self) {
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = Instant::now();
        }
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

impl From<BackendFailure> for BackendError {
    fn from(failure: BackendFailure) -> Self {
        Self {
            code: failure.code().as_str().to_owned(),
            message: failure.message().to_owned(),
        }
    }
}

#[tauri::command]
pub(crate) fn get_backend_status(
    state: State<'_, BackendState>,
) -> Result<BackendStatus, BackendError> {
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone();
    let settings = match settings {
        Ok(settings) => settings,
        Err(message) => return Ok(BackendStatus::configuration_error(&message)),
    };
    let engine = state
        .engine
        .lock()
        .map_err(|_| BackendFailure::internal("后端状态锁已损坏"))?;
    Ok(settings.status(
        engine
            .as_ref()
            .is_some_and(BackendEngine::translator_loaded),
    ))
}

#[tauri::command]
pub(crate) fn update_backend_settings(
    request: BackendSettingsUpdate,
    state: State<'_, BackendState>,
) -> Result<BackendStatus, BackendError> {
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
    state.touch_activity();
    Ok(updated.status(false))
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
pub(crate) fn cancel_translation(
    request: CancelTranslationRequest,
    state: State<'_, BackendState>,
) -> Result<(), BackendError> {
    let run_id = RunId::parse(&request.request_id)?;
    state.runs.cancel(&run_id)?;
    Ok(())
}

#[tauri::command]
pub(crate) async fn translate_image(
    app: tauri::AppHandle,
    request: TranslateImageRequest,
    state: State<'_, BackendState>,
) -> Result<TranslationResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let lease = state.runs.register(run_id.clone())?;
    emit_translation_progress(&app, &run_id, 5, "正在准备翻译请求");
    let decoded = decode_request(request)?;
    emit_translation_progress(&app, &run_id, 15, "图片已解码");
    let state = state.inner().clone();
    let app = app.clone();
    let run_id = run_id.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<TranslationResponse, BackendFailure> {
        lease.token().check()?;
        let response =
            translate_image_blocking(decoded, &state, lease.token(), &app, &run_id, started_at)?;
        lease.finalize_success()?;
        Ok(response)
    })
    .await
    .map_err(|error| {
        BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
    })?
    .map_err(BackendError::from)
}

#[tauri::command]
pub(crate) async fn translate_text(
    app: tauri::AppHandle,
    request: TranslateTextRequest,
    state: State<'_, BackendState>,
) -> Result<TextTranslationResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let lease = state.runs.register(run_id.clone())?;
    emit_translation_progress(&app, &run_id, 5, "正在准备翻译请求");
    let (text, target_language) = decode_text_request(request)?;
    emit_translation_progress(&app, &run_id, 15, "文本请求已验证");
    let state = state.inner().clone();
    let app = app.clone();
    let run_id = run_id.clone();
    tauri::async_runtime::spawn_blocking(
        move || -> Result<TextTranslationResponse, BackendFailure> {
            lease.token().check()?;
            let response = translate_text_blocking(
                text,
                target_language,
                &state,
                lease.token(),
                &app,
                &run_id,
                started_at,
            )?;
            lease.finalize_success()?;
            Ok(response)
        },
    )
    .await
    .map_err(|error| {
        BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
    })?
    .map_err(BackendError::from)
}

#[tauri::command]
pub(crate) async fn ocr_image(
    app: tauri::AppHandle,
    request: OcrImageRequest,
    state: State<'_, BackendState>,
) -> Result<OcrResponse, BackendError> {
    let started_at = Instant::now();
    let run_id = RunId::from_optional(request.request_id.as_deref())?;
    let lease = state.runs.register(run_id.clone())?;
    emit_translation_progress(&app, &run_id, 5, "正在准备 OCR 请求");
    let decoded = decode_ocr_request(request)?;
    emit_translation_progress(&app, &run_id, 15, "图片已解码");
    let state = state.inner().clone();
    let app = app.clone();
    let run_id = run_id.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<OcrResponse, BackendFailure> {
        lease.token().check()?;
        let response =
            ocr_image_blocking(decoded, &state, lease.token(), &app, &run_id, started_at)?;
        lease.finalize_success()?;
        Ok(response)
    })
    .await
    .map_err(|error| {
        BackendFailure::internal(format!("Candle worker exited unexpectedly: {error}"))
    })?
    .map_err(BackendError::from)
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
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation)?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.translate(&request, cancellation, |progress, stage| {
            emit_translation_progress(app, run_id, progress, stage);
        })
    };
    state.touch_activity();
    let result = result?;
    let image_base64 = BASE64.encode(result.annotated_png);
    Ok(TranslationResponse {
        text: result.text,
        markdown: result.markdown,
        annotated_image_data_url: format!("data:image/png;base64,{image_base64}"),
        provider_label: result.provider_label,
        is_translated: result.is_translated,
        duration_ms: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
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
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation)?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.translate_text(&text, &target_language, cancellation, |progress, stage| {
            emit_translation_progress(app, run_id, progress, stage);
        })
    };
    state.touch_activity();
    Ok(TextTranslationResponse {
        text: result?,
        provider_label: "Hy-MT2 / Candle".to_owned(),
        duration_ms: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
    })
}

fn ocr_image_blocking(
    request: DecodedImage,
    state: &BackendState,
    cancellation: &crate::model_support::CancellationToken,
    app: &tauri::AppHandle,
    run_id: &RunId,
    started_at: Instant,
) -> Result<OcrResponse, BackendFailure> {
    state.touch_activity();
    let settings = state
        .settings
        .lock()
        .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
        .clone()
        .map_err(BackendFailure::arguments)?;
    let (image_width, image_height) = request.canvas().dimensions();
    let result = {
        let mut engine = lock_with_cancellation(&state.engine, cancellation)?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        engine.ocr(&request, cancellation, |progress, stage| {
            emit_translation_progress(app, run_id, progress, stage);
        })
    };
    state.touch_activity();
    let result = result?;
    let image_base64 = BASE64.encode(result.annotated_png);
    Ok(OcrResponse {
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
    })
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
            "idleUnloadMinutes": 0,
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
                "system": "Return concise JSON.",
                "user": "Preserve product names."
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
        assert_eq!(request.prompt.system, "Return concise JSON.");
        assert_eq!(request.prompt.user, "Preserve product names.");
    }
    #[test]
    fn ocr_region_response_serializes_selectable_character_boxes() {
        let response = OcrRegionResponse::from(RegionRecord {
            order: 1,
            quad_points: [[0, 0], [20, 0], [20, 10], [0, 10]],
            source_text: "AB".to_owned(),
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
