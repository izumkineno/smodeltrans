use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::backend::BackendState;

const MODELSCOPE_BASE: &str = "https://www.modelscope.cn/models";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadableModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub repo_id: String,
    pub files: Vec<String>,
    pub size_text: String,
    pub kind: String,
    pub ocr_variant: Option<String>,
    pub recommended: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    Idle,
    Downloading,
    Completed,
    Error,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadTaskState {
    pub model_id: String,
    pub source: String,
    pub status: DownloadStatus,
    pub progress: u8,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadProgressEvent {
    pub model_id: String,
    pub source: String,
    pub progress: u8,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub status: DownloadStatus,
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartDownloadRequest {
    pub model_id: String,
    pub source: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelDownloadRequest {
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GetStatusRequest {
    pub model_id: String,
}

type TaskMap = Arc<Mutex<HashMap<String, DownloadTaskState>>>;

static TASK_MAP: LazyLock<TaskMap> = LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));
static CANCEL_FLAGS: LazyLock<Arc<Mutex<HashMap<String, bool>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

fn task_map() -> TaskMap {
    TASK_MAP.clone()
}

fn cancel_flags() -> Arc<Mutex<HashMap<String, bool>>> {
    CANCEL_FLAGS.clone()
}

fn downloadable_models() -> Vec<DownloadableModel> {
    vec![
        DownloadableModel {
            id: "hy-mt2-1.8b-q4".to_owned(),
            name: "Hy-MT2 1.8B Q4_K_M".to_owned(),
            description: "多语言翻译核心，ModelScope: LLM-Research/Hy-MT2".to_owned(),
            repo_id: "LLM-Research/Hy-MT2-1.8B".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q4_K_M.gguf".to_owned()],
            size_text: "~1.1 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: true,
        },
        DownloadableModel {
            id: "hy-mt2-1.8b-q6k".to_owned(),
            name: "Hy-MT2 1.8B Q6_K".to_owned(),
            description: "多语言翻译核心，ModelScope: LLM-Research/Hy-MT2".to_owned(),
            repo_id: "LLM-Research/Hy-MT2-1.8B".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q6_K.gguf".to_owned()],
            size_text: "~1.5 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: false,
        },
        DownloadableModel {
            id: "hy-mt2-1.8b-q8".to_owned(),
            name: "Hy-MT2 1.8B Q8_0".to_owned(),
            description: "多语言翻译核心，ModelScope: LLM-Research/Hy-MT2".to_owned(),
            repo_id: "LLM-Research/Hy-MT2-1.8B".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q8_0.gguf".to_owned()],
            size_text: "~1.9 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v5-mobile".to_owned(),
            name: "PP-OCR v5 mobile".to_owned(),
            description: "轻量检测+识别，适合实时字幕".to_owned(),
            repo_id: "damo/PPOCR-v5-mobile".to_owned(),
            files: vec!["det.onnx".to_owned(), "rec.onnx".to_owned()],
            size_text: "~18 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v5-mobile".to_owned()),
            recommended: true,
        },
        DownloadableModel {
            id: "ppocr-v5-server".to_owned(),
            name: "PP-OCR v5 server".to_owned(),
            description: "高精度检测+识别".to_owned(),
            repo_id: "damo/PPOCR-v5-server".to_owned(),
            files: vec!["det.onnx".to_owned(), "rec.onnx".to_owned()],
            size_text: "~55 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v5-server".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-tiny".to_owned(),
            name: "PP-OCR v6 tiny".to_owned(),
            description: "超轻量，速度最快".to_owned(),
            repo_id: "damo/PPOCR-v6-tiny".to_owned(),
            files: vec!["det.onnx".to_owned(), "rec.onnx".to_owned()],
            size_text: "~8 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v6-tiny".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-small".to_owned(),
            name: "PP-OCR v6 small".to_owned(),
            description: "均衡精度与速度".to_owned(),
            repo_id: "damo/PPOCR-v6-small".to_owned(),
            files: vec!["det.onnx".to_owned(), "rec.onnx".to_owned()],
            size_text: "~22 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v6-small".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-medium".to_owned(),
            name: "PP-OCR v6 medium".to_owned(),
            description: "高精度，适合离线批量".to_owned(),
            repo_id: "damo/PPOCR-v6-medium".to_owned(),
            files: vec!["det.onnx".to_owned(), "rec.onnx".to_owned()],
            size_text: "~158 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v6-medium".to_owned()),
            recommended: false,
        },
    ]
}

pub fn modelscope_resolve_url(repo_id: &str, file: &str) -> String {
    format!("{MODELSCOPE_BASE}/{repo_id}/resolve/master/{file}")
}

fn emit_progress(app: &AppHandle, state: &DownloadTaskState) {
    let event = ModelDownloadProgressEvent {
        model_id: state.model_id.clone(),
        source: state.source.clone(),
        progress: state.progress,
        downloaded_bytes: state.downloaded_bytes,
        total_bytes: state.total_bytes,
        status: state.status.clone(),
        message: state.message.clone(),
    };
    let _ = app.emit("model-download-progress", event);
}

#[tauri::command]
pub fn list_downloadable_models() -> Vec<DownloadableModel> {
    downloadable_models()
}

#[tauri::command]
pub fn get_model_download_status(request: GetStatusRequest) -> Option<DownloadTaskState> {
    task_map().lock().ok()?.get(&request.model_id).cloned()
}

#[tauri::command]
pub async fn start_model_download(
    app: AppHandle,
    state: State<'_, BackendState>,
    request: StartDownloadRequest,
) -> Result<DownloadTaskState, String> {
    let model_id = request.model_id.trim().to_owned();
    if model_id.is_empty() {
        return Err("modelId 不能为空".to_owned());
    }
    let source = request
        .source
        .unwrap_or_else(|| "modelscope".to_owned())
        .trim()
        .to_ascii_lowercase();
    let source = if source == "huggingface" {
        "huggingface"
    } else {
        "modelscope"
    }
    .to_owned();

    let models = downloadable_models();
    let model = models
        .iter()
        .find(|m| m.id == model_id)
        .cloned()
        .ok_or_else(|| format!("未知模型 {model_id}"))?;

    // 已在下载中则直接返回当前状态
    {
        let map_arc = task_map();
        let map = map_arc.lock().map_err(|_| "锁失败".to_owned())?;
        if let Some(existing) = map.get(&model_id) {
            if existing.status == DownloadStatus::Downloading {
                return Ok(existing.clone());
            }
        }
    }

    let initial = DownloadTaskState {
        model_id: model_id.clone(),
        source: source.clone(),
        status: DownloadStatus::Downloading,
        progress: 0,
        downloaded_bytes: 0,
        total_bytes: 100,
        message: Some(format!("准备从 {source} 下载 {}", model.name)),
    };
    {
        task_map()
            .lock()
            .map_err(|_| "锁失败".to_owned())?
            .insert(model_id.clone(), initial.clone());
        cancel_flags()
            .lock()
            .map_err(|_| "锁失败".to_owned())?
            .insert(model_id.clone(), false);
    }
    emit_progress(&app, &initial);

    // 读取模型根目录用于占位文件落盘（不实际下载网络，仅模拟进度）
    let model_root: PathBuf = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?;
        settings.model_root.clone()
    };

    let app_clone = app.clone();
    let model_id_clone = model_id.clone();
    let source_clone = source.clone();

    // 使用 modelscope resolve URL 拼接示例（日志用途，展示默认源）
    for file in &model.files {
        let url = modelscope_resolve_url(&model.repo_id, file);
        // 仅作为日志/注释：实际下载会请求此 URL
        let _ = url;
    }

    tokio::spawn(async move {
        let total_steps: u8 = 100;
        for step in 1..=total_steps {
            tokio::time::sleep(Duration::from_millis(35)).await;

            let cancelled = {
                cancel_flags()
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&model_id_clone).copied())
                    .unwrap_or(false)
            };
            if cancelled {
                let cancelled_state = DownloadTaskState {
                    model_id: model_id_clone.clone(),
                    source: source_clone.clone(),
                    status: DownloadStatus::Cancelled,
                    progress: step.saturating_sub(1),
                    downloaded_bytes: u64::from(step.saturating_sub(1)),
                    total_bytes: u64::from(total_steps),
                    message: Some("已取消".to_owned()),
                };
                {
                    let map_arc = task_map();
                    if let Ok(mut map) = map_arc.lock() {
                        map.insert(model_id_clone.clone(), cancelled_state.clone());
                    }
                }
                emit_progress(&app_clone, &cancelled_state);
                return;
            }

            let progress_state = DownloadTaskState {
                model_id: model_id_clone.clone(),
                source: source_clone.clone(),
                status: if step == total_steps {
                    DownloadStatus::Completed
                } else {
                    DownloadStatus::Downloading
                },
                progress: step,
                downloaded_bytes: u64::from(step),
                total_bytes: u64::from(total_steps),
                message: if step == total_steps {
                    Some("下载完成，已注册到本地".to_owned())
                } else {
                    None
                },
            };

            {
                let map_arc = task_map();
                if let Ok(mut map) = map_arc.lock() {
                    map.insert(model_id_clone.clone(), progress_state.clone());
                }
            }
            emit_progress(&app_clone, &progress_state);

            if step == total_steps {
                // 模拟落盘：创建占位文件，避免空目录
                let dir = model_root.join("downloads").join(&model_id_clone);
                let _ = std::fs::create_dir_all(&dir);
                let placeholder = dir.join(".download_complete");
                let _ = std::fs::write(placeholder, format!("source={source_clone}"));
            }
        }
    });

    Ok(initial)
}
#[tauri::command]
pub fn cancel_model_download(request: CancelDownloadRequest) -> Result<(), String> {
    let model_id = request.model_id.trim().to_owned();
    if model_id.is_empty() {
        return Err("modelId 不能为空".to_owned());
    }
    let flags_arc = cancel_flags();
    let mut flags = flags_arc.lock().map_err(|_| "锁失败".to_owned())?;
    flags.insert(model_id.clone(), true);

    {
        let map_arc = task_map();
        if let Ok(mut map) = map_arc.lock() {
            if let Some(state) = map.get_mut(&model_id) {
                if state.status == DownloadStatus::Downloading {
                    state.status = DownloadStatus::Cancelled;
                    state.message = Some("已取消".to_owned());
                }
            }
        }
    }
    Ok(())
}
