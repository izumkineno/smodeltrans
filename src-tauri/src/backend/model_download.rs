use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use simple_downloader::{DownloadInfo, Downloader};
use tauri::{AppHandle, Emitter, State};
use tokio::task::JoinHandle;

use crate::backend::BackendState;

const MODELSCOPE_BASE: &str = "https://www.modelscope.cn/models";
const HUGGINGFACE_BASE: &str = "https://huggingface.co";
/// ModelScope CDN 会拦截空或异常 UA，显式使用浏览器 UA 避免 403 `denied by UA ACL = blacklist`
const BROWSER_UA: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 smodeltrans/0.1.0";
const MAX_RETRIES: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DownloadFileSpec {
    pub repo_id: String,
    pub file: String,
    pub dest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadableModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub repo_id: String,
    pub files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_specs: Vec<DownloadFileSpec>,
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
static DOWNLOAD_HANDLES: LazyLock<Arc<Mutex<HashMap<String, JoinHandle<()>>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

fn task_map() -> TaskMap {
    TASK_MAP.clone()
}

fn cancel_flags() -> Arc<Mutex<HashMap<String, bool>>> {
    CANCEL_FLAGS.clone()
}

fn download_handles() -> Arc<Mutex<HashMap<String, JoinHandle<()>>>> {
    DOWNLOAD_HANDLES.clone()
}

fn downloadable_models() -> Vec<DownloadableModel> {
    vec![
        DownloadableModel {
            id: "hy-mt2-1.8b-q4".to_owned(),
            name: "Hy-MT2 1.8B Q4_K_M".to_owned(),
            description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q4_K_M.gguf".to_owned()],
            file_specs: vec![DownloadFileSpec {
                repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
                file: "Hy-MT2-1.8B-Q4_K_M.gguf".to_owned(),
                dest: "Hy-MT2-1.8B-Q4_K_M.gguf".to_owned(),
            }],
            size_text: "~1.1 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: true,
        },
        DownloadableModel {
            id: "hy-mt2-1.8b-q6k".to_owned(),
            name: "Hy-MT2 1.8B Q6_K".to_owned(),
            description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q6_K.gguf".to_owned()],
            file_specs: vec![DownloadFileSpec {
                repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
                file: "Hy-MT2-1.8B-Q6_K.gguf".to_owned(),
                dest: "Hy-MT2-1.8B-Q6_K.gguf".to_owned(),
            }],
            size_text: "~1.5 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: false,
        },
        DownloadableModel {
            id: "hy-mt2-1.8b-q8".to_owned(),
            name: "Hy-MT2 1.8B Q8_0".to_owned(),
            description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
            files: vec!["Hy-MT2-1.8B-Q8_0.gguf".to_owned()],
            file_specs: vec![DownloadFileSpec {
                repo_id: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF".to_owned(),
                file: "Hy-MT2-1.8B-Q8_0.gguf".to_owned(),
                dest: "Hy-MT2-1.8B-Q8_0.gguf".to_owned(),
            }],
            size_text: "~1.9 GB".to_owned(),
            kind: "translation".to_owned(),
            ocr_variant: None,
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v5-mobile".to_owned(),
            name: "PP-OCR v5 mobile".to_owned(),
            description: "轻量检测+识别，适合实时字幕，ModelScope: PaddlePaddle/PP-OCRv5".to_owned(),
            repo_id: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors".to_owned(),
            files: vec![
                "mobile_det/model.safetensors".to_owned(),
                "mobile_det/config.json".to_owned(),
                "mobile_det/preprocessor_config.json".to_owned(),
                "mobile_det/inference.yml".to_owned(),
                "mobile_rec/model.safetensors".to_owned(),
                "mobile_rec/config.json".to_owned(),
                "mobile_rec/preprocessor_config.json".to_owned(),
                "mobile_rec/inference.yml".to_owned(),
            ],
            file_specs: vec![
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "mobile_det/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors".to_owned(), file: "config.json".to_owned(), dest: "mobile_det/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "mobile_det/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "mobile_det/inference.yml".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "mobile_rec/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors".to_owned(), file: "config.json".to_owned(), dest: "mobile_rec/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "mobile_rec/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "mobile_rec/inference.yml".to_owned() },
            ],
            size_text: "~18 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v5-mobile".to_owned()),
            recommended: true,
        },
        DownloadableModel {
            id: "ppocr-v5-server".to_owned(),
            name: "PP-OCR v5 server".to_owned(),
            description: "高精度检测+识别，ModelScope: PaddlePaddle/PP-OCRv5".to_owned(),
            repo_id: "PaddlePaddle/PP-OCRv5_server_det_safetensors".to_owned(),
            files: vec![
                "server_det/model.safetensors".to_owned(),
                "server_det/config.json".to_owned(),
                "server_det/preprocessor_config.json".to_owned(),
                "server_det/inference.yml".to_owned(),
                "server_rec/model.safetensors".to_owned(),
                "server_rec/config.json".to_owned(),
                "server_rec/preprocessor_config.json".to_owned(),
                "server_rec/inference.yml".to_owned(),
            ],
            file_specs: vec![
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_det_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "server_det/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_det_safetensors".to_owned(), file: "config.json".to_owned(), dest: "server_det/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_det_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "server_det/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_det_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "server_det/inference.yml".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_rec_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "server_rec/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_rec_safetensors".to_owned(), file: "config.json".to_owned(), dest: "server_rec/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_rec_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "server_rec/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv5_server_rec_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "server_rec/inference.yml".to_owned() },
            ],
            size_text: "~55 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v5-server".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-tiny".to_owned(),
            name: "PP-OCR v6 tiny".to_owned(),
            description: "超轻量，速度最快，ModelScope: PaddlePaddle/PP-OCRv6".to_owned(),
            repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(),
            files: vec![
                "tiny_det/model.safetensors".to_owned(),
                "tiny_det/config.json".to_owned(),
                "tiny_det/preprocessor_config.json".to_owned(),
                "tiny_det/inference.yml".to_owned(),
                "tiny_det/configuration.json".to_owned(),
                "tiny_rec/model.safetensors".to_owned(),
                "tiny_rec/config.json".to_owned(),
                "tiny_rec/preprocessor_config.json".to_owned(),
                "tiny_rec/inference.yml".to_owned(),
            ],
            file_specs: vec![
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "tiny_det/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(), file: "config.json".to_owned(), dest: "tiny_det/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "tiny_det/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "tiny_det/inference.yml".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors".to_owned(), file: "configuration.json".to_owned(), dest: "tiny_det/configuration.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "tiny_rec/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors".to_owned(), file: "config.json".to_owned(), dest: "tiny_rec/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "tiny_rec/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "tiny_rec/inference.yml".to_owned() },
            ],
            size_text: "~8 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v6-tiny".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-small".to_owned(),
            name: "PP-OCR v6 small".to_owned(),
            description: "均衡精度与速度，ModelScope: PaddlePaddle/PP-OCRv6".to_owned(),
            repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(),
            files: vec![
                "small_det/model.safetensors".to_owned(),
                "small_det/config.json".to_owned(),
                "small_det/preprocessor_config.json".to_owned(),
                "small_det/inference.yml".to_owned(),
                "small_det/configuration.json".to_owned(),
                "small_rec/model.safetensors".to_owned(),
                "small_rec/config.json".to_owned(),
                "small_rec/preprocessor_config.json".to_owned(),
                "small_rec/inference.yml".to_owned(),
            ],
            file_specs: vec![
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "small_det/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(), file: "config.json".to_owned(), dest: "small_det/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "small_det/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "small_det/inference.yml".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors".to_owned(), file: "configuration.json".to_owned(), dest: "small_det/configuration.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_rec_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "small_rec/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_rec_safetensors".to_owned(), file: "config.json".to_owned(), dest: "small_rec/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_rec_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "small_rec/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_small_rec_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "small_rec/inference.yml".to_owned() },
            ],
            size_text: "~22 MB".to_owned(),
            kind: "ocr".to_owned(),
            ocr_variant: Some("v6-small".to_owned()),
            recommended: false,
        },
        DownloadableModel {
            id: "ppocr-v6-medium".to_owned(),
            name: "PP-OCR v6 medium".to_owned(),
            description: "高精度，适合离线批量，ModelScope: PaddlePaddle/PP-OCRv6".to_owned(),
            repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(),
            files: vec![
                "medium_det/model.safetensors".to_owned(),
                "medium_det/config.json".to_owned(),
                "medium_det/preprocessor_config.json".to_owned(),
                "medium_det/inference.yml".to_owned(),
                "medium_det/configuration.json".to_owned(),
                "medium_rec/model.safetensors".to_owned(),
                "medium_rec/config.json".to_owned(),
                "medium_rec/preprocessor_config.json".to_owned(),
                "medium_rec/inference.yml".to_owned(),
            ],
            file_specs: vec![
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "medium_det/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(), file: "config.json".to_owned(), dest: "medium_det/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "medium_det/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "medium_det/inference.yml".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors".to_owned(), file: "configuration.json".to_owned(), dest: "medium_det/configuration.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors".to_owned(), file: "model.safetensors".to_owned(), dest: "medium_rec/model.safetensors".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors".to_owned(), file: "config.json".to_owned(), dest: "medium_rec/config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors".to_owned(), file: "preprocessor_config.json".to_owned(), dest: "medium_rec/preprocessor_config.json".to_owned() },
                DownloadFileSpec { repo_id: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors".to_owned(), file: "inference.yml".to_owned(), dest: "medium_rec/inference.yml".to_owned() },
            ],
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

pub fn huggingface_resolve_url(repo_id: &str, file: &str) -> String {
    format!("{HUGGINGFACE_BASE}/{repo_id}/resolve/main/{file}")
}

fn resolve_download_url(repo_id: &str, file: &str, source: &str) -> String {
    if source == "huggingface" {
        huggingface_resolve_url(repo_id, file)
    } else {
        modelscope_resolve_url(repo_id, file)
    }
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

fn is_retryable_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    // 网络抖动 / 超时 / 连接重置 / 403 CDN 限流 / 5xx 均可重试
    lower.contains("403")
        || lower.contains("forbidden")
        || lower.contains("429")
        || lower.contains("too many requests")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("connection")
        || lower.contains("reset")
        || lower.contains("broken pipe")
        || lower.contains("50")
        || lower.contains("denied by ua")
        || lower.contains("blacklist")
}

fn friendly_download_error(spec: &DownloadFileSpec, url: &str, err: &str, source: &str, dest: &Path) -> String {
    let lower = err.to_ascii_lowercase();
    let base = format!("下载失败 {}: {err}", spec.dest);
    if lower.contains("403") || lower.contains("forbidden") || lower.contains("denied by ua") || lower.contains("blacklist") {
        format!(
            "{base}；ModelScope CDN 拒绝访问 (403)，通常为网络连接问题或 UA 被拦截。已自动使用浏览器标识重试仍失败，请：1) 点击“重试” 2) 检查网络/系统代理后重试 3) 手动在浏览器打开 {url} 下载后放到 {}。源: {source}",
            dest.display()
        )
    } else if lower.contains("timed out") || lower.contains("timeout") || lower.contains("connection") {
        format!(
            "{base}；网络连接超时或中断，请检查网络/代理后重试，或手动下载 {url} 到 {}。源: {source}",
            dest.display()
        )
    } else if lower.contains("404") || lower.contains("not found") {
        format!("{base}；文件在 {source} 未找到 (404)，请确认链接 {url} 是否有效")
    } else {
        format!("{base}；{err}；可尝试重试或手动下载 {url} 到 {}", dest.display())
    }
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

    // 读取模型根目录
    let model_root: PathBuf = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?;
        settings.model_root.clone()
    };

    let app_clone = app.clone();
    let model_id_clone = model_id.clone();
    let source_clone = source.clone();
    let model_clone = model.clone();

    // 使用 simple_downloader 进行真实下载
    let handle = tokio::spawn(async move {
        let file_specs: Vec<DownloadFileSpec> = if !model_clone.file_specs.is_empty() {
            model_clone.file_specs.clone()
        } else {
            model_clone
                .files
                .iter()
                .map(|f| DownloadFileSpec {
                    repo_id: model_clone.repo_id.clone(),
                    file: f.clone(),
                    dest: f.clone(),
                })
                .collect()
        };
        let total_files = file_specs.len() as u64;
        if total_files == 0 {
            let error_state = DownloadTaskState {
                model_id: model_id_clone.clone(),
                source: source_clone.clone(),
                status: DownloadStatus::Error,
                progress: 0,
                downloaded_bytes: 0,
                total_bytes: 100,
                message: Some("模型文件列表为空".to_owned()),
            };
            if let Ok(mut map) = task_map().lock() {
                map.insert(model_id_clone.clone(), error_state.clone());
            }
            emit_progress(&app_clone, &error_state);
            return;
        }

        let base_dir = model_root.join("downloads").join(&model_id_clone);
        if let Err(err) = tokio::fs::create_dir_all(&base_dir).await {
            let error_state = DownloadTaskState {
                model_id: model_id_clone.clone(),
                source: source_clone.clone(),
                status: DownloadStatus::Error,
                progress: 0,
                downloaded_bytes: 0,
                total_bytes: 100,
                message: Some(format!("创建目录失败: {err}")),
            };
            if let Ok(mut map) = task_map().lock() {
                map.insert(model_id_clone.clone(), error_state.clone());
            }
            emit_progress(&app_clone, &error_state);
            return;
        }

        for (idx, spec) in file_specs.iter().enumerate() {
            // 检查取消
            let cancelled = cancel_flags()
                .lock()
                .ok()
                .and_then(|m| m.get(&model_id_clone).copied())
                .unwrap_or(false);
            if cancelled {
                let cancelled_state = DownloadTaskState {
                    model_id: model_id_clone.clone(),
                    source: source_clone.clone(),
                    status: DownloadStatus::Cancelled,
                    progress: ((idx as f64 / total_files as f64) * 100.0) as u8,
                    downloaded_bytes: idx as u64,
                    total_bytes: total_files,
                    message: Some("已取消".to_owned()),
                };
                if let Ok(mut map) = task_map().lock() {
                    map.insert(model_id_clone.clone(), cancelled_state.clone());
                }
                emit_progress(&app_clone, &cancelled_state);
                return;
            }

            let url = resolve_download_url(&spec.repo_id, &spec.file, &source_clone);
            let dest_path = base_dir.join(&spec.dest);
            if let Some(parent) = dest_path.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }

            // 带重试的下载：首试 8 线程，失败自动降级为单线程，并使用浏览器 UA 避免 403 `denied by UA ACL = blacklist`
            let mut last_err: Option<String> = None;
            let mut succeeded = false;
            let mut was_cancelled = false;
            for attempt in 0..MAX_RETRIES {
                // 重试前检查取消
                let cancelled_before = cancel_flags()
                    .lock()
                    .ok()
                    .and_then(|m| m.get(&model_id_clone).copied())
                    .unwrap_or(false);
                if cancelled_before {
                    was_cancelled = true;
                    break;
                }
                if attempt > 0 {
                    let delay_ms = 500u64 * (1u64 << (attempt - 1));
                    let retry_msg = format!(
                        "下载 {} 遇到网络问题，正在重试 {}/{}（{}）...",
                        spec.dest,
                        attempt + 1,
                        MAX_RETRIES,
                        last_err.clone().unwrap_or_default()
                    );
                    let retry_state = DownloadTaskState {
                        model_id: model_id_clone.clone(),
                        source: source_clone.clone(),
                        status: DownloadStatus::Downloading,
                        progress: ((idx as f64 / total_files as f64) * 100.0) as u8,
                        downloaded_bytes: idx as u64,
                        total_bytes: total_files,
                        message: Some(retry_msg),
                    };
                    if let Ok(mut map) = task_map().lock() {
                        map.insert(model_id_clone.clone(), retry_state.clone());
                    }
                    emit_progress(&app_clone, &retry_state);
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    // 再次检查取消
                    let cancelled_during_wait = cancel_flags()
                        .lock()
                        .ok()
                        .and_then(|m| m.get(&model_id_clone).copied())
                        .unwrap_or(false);
                    if cancelled_during_wait {
                        was_cancelled = true;
                        break;
                    }
                }
                let workers = if attempt == 0 { 8 } else { 1 };
                let url_clone = url.clone();
                let dest_str = dest_path.to_string_lossy().to_string();
                let app_for_file = app_clone.clone();
                let model_id_for_file = model_id_clone.clone();
                let source_for_file = source_clone.clone();
                let file_idx = idx as u64;
                let file_name = spec.dest.clone();

                let download_result = {
                    let cancel_flags_for_select = cancel_flags();
                    let model_id_for_select = model_id_clone.clone();
                    let download_fut = async move {
                        let builder = Downloader::builder(url_clone.clone(), dest_str.clone())
                            .workers(workers)
                            .update_interval(0.5)
                            .client_builder(|| {
                                simple_downloader::reqwest::ClientBuilder::new()
                                    .user_agent(BROWSER_UA)
                                    .timeout(Duration::from_secs(180))
                                    .connect_timeout(Duration::from_secs(15))
                                    .pool_max_idle_per_host(8)
                            });
                        #[cfg(feature = "resume")]
                        let builder = builder.resume(true);
                        let res = builder
                            .run(move |total_size, mut info_rx| {
                                let app_inner = app_for_file.clone();
                                let model_id_inner = model_id_for_file.clone();
                                let file_name_inner = file_name.clone();
                                let source_inner = source_for_file.clone();
                                async move {
                                    while let Ok(info) = info_rx.recv().await {
                                        if let DownloadInfo::MonitorUpdate {
                                            total_downloaded,
                                            total_size: _,
                                            ..
                                        } = &info
                                        {
                                            let file_progress = if total_size > 0 {
                                                *total_downloaded as f64 / total_size as f64
                                            } else {
                                                0.0
                                            };
                                            let overall_progress = ((file_idx as f64 + file_progress)
                                                / total_files as f64
                                                * 100.0) as u8;
                                            let overall_progress = overall_progress.min(100);
                                            let speed_msg = format!("{:.2} MB/s", info.speed_mbps());
                                            let msg = if file_progress < 1.0 {
                                                Some(format!(
                                                    "下载 {file_name_inner} {overall_progress}% {speed_msg}"
                                                ))
                                            } else {
                                                None
                                            };
                                            let state = DownloadTaskState {
                                                model_id: model_id_inner.clone(),
                                                source: source_inner.clone(),
                                                status: DownloadStatus::Downloading,
                                                progress: overall_progress,
                                                downloaded_bytes: *total_downloaded,
                                                total_bytes: total_size,
                                                message: msg,
                                            };
                                            if let Ok(mut map) = task_map().lock() {
                                                map.insert(model_id_inner.clone(), state.clone());
                                            }
                                            emit_progress(&app_inner, &state);
                                            if info.is_complete() {
                                                break;
                                            }
                                        }
                                    }
                                }
                            })
                            .await;
                        res
                    };
                    let model_id_for_cancel = model_id_for_select.clone();
                    let cancel_fut = async move {
                        loop {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            let is_cancelled = cancel_flags_for_select
                                .lock()
                                .ok()
                                .and_then(|m| m.get(&model_id_for_cancel).copied())
                                .unwrap_or(false);
                            if is_cancelled {
                                break;
                            }
                        }
                    };
                    tokio::select! {
                        res = download_fut => Some(res),
                        _ = cancel_fut => None,
                    }
                };

                match download_result {
                    Some(Ok(())) => {
                        succeeded = true;
                        break;
                    }
                    Some(Err(err)) => {
                        let err_str = format!("{err}");
                        last_err = Some(err_str.clone());
                        // 非重试错误或已达最大重试直接退出
                        if !is_retryable_error(&err_str) || attempt + 1 == MAX_RETRIES {
                            break;
                        }
                        // 可重试则进入下一轮
                    }
                    None => {
                        was_cancelled = true;
                        break;
                    }
                }
            }

            if was_cancelled {
                let cancelled_state = DownloadTaskState {
                    model_id: model_id_clone.clone(),
                    source: source_clone.clone(),
                    status: DownloadStatus::Cancelled,
                    progress: ((idx as f64 / total_files as f64) * 100.0) as u8,
                    downloaded_bytes: idx as u64,
                    total_bytes: total_files,
                    message: Some("已取消".to_owned()),
                };
                if let Ok(mut map) = task_map().lock() {
                    map.insert(model_id_clone.clone(), cancelled_state.clone());
                }
                emit_progress(&app_clone, &cancelled_state);
                if let Ok(mut handles) = download_handles().lock() {
                    handles.remove(&model_id_clone);
                }
                return;
            }
            if succeeded {
                let overall_progress = (((idx + 1) as f64 / total_files as f64) * 100.0) as u8;
                let state = DownloadTaskState {
                    model_id: model_id_clone.clone(),
                    source: source_clone.clone(),
                    status: if (idx + 1) as u64 == total_files {
                        DownloadStatus::Completed
                    } else {
                        DownloadStatus::Downloading
                    },
                    progress: overall_progress,
                    downloaded_bytes: (idx + 1) as u64,
                    total_bytes: total_files,
                    message: if (idx + 1) as u64 == total_files {
                        Some("下载完成，已落盘".to_owned())
                    } else {
                        Some(format!("已完成 {}/{} 文件", idx + 1, total_files))
                    },
                };
                if let Ok(mut map) = task_map().lock() {
                    map.insert(model_id_clone.clone(), state.clone());
                }
                emit_progress(&app_clone, &state);
                if (idx + 1) as u64 == total_files {
                    let placeholder = base_dir.join(".download_complete");
                    let _ = tokio::fs::write(&placeholder, format!("source={source_clone}")).await;
                    if let Ok(mut handles) = download_handles().lock() {
                        handles.remove(&model_id_clone);
                    }
                    return;
                }
            } else {
                let err_str = last_err.unwrap_or_else(|| "未知错误".to_owned());
                let friendly = friendly_download_error(&spec, &url, &err_str, &source_clone, &dest_path);
                let error_state = DownloadTaskState {
                    model_id: model_id_clone.clone(),
                    source: source_clone.clone(),
                    status: DownloadStatus::Error,
                    progress: ((idx as f64 / total_files as f64) * 100.0) as u8,
                    downloaded_bytes: idx as u64,
                    total_bytes: total_files,
                    message: Some(friendly),
                };
                if let Ok(mut map) = task_map().lock() {
                    map.insert(model_id_clone.clone(), error_state.clone());
                }
                emit_progress(&app_clone, &error_state);
                if let Ok(mut handles) = download_handles().lock() {
                    handles.remove(&model_id_clone);
                }
                return;
            }
        }

        // 清理句柄（正常完成分支已清理，此处兜底）
        if let Ok(mut handles) = download_handles().lock() {
            handles.remove(&model_id_clone);
        }
    });

    // 保存句柄以支持取消
    if let Ok(mut handles) = download_handles().lock() {
        handles.insert(model_id.clone(), handle);
    }

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

    // 中止对应的下载任务
    if let Ok(mut handles) = download_handles().lock() {
        if let Some(handle) = handles.remove(&model_id) {
            handle.abort();
        }
    }

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

fn download_base_dir(model_root: &Path, model_id: &str) -> PathBuf {
    model_root.join("downloads").join(model_id)
}

fn is_model_downloaded_on_disk(model_root: &Path, model: &DownloadableModel) -> bool {
    let base = download_base_dir(model_root, &model.id);
    // 优先检查完成标记
    let marker = base.join(".download_complete");
    let has_marker = marker.is_file();
    // 检查所有 file_specs 是否存在
    let specs: Vec<&DownloadFileSpec> = if !model.file_specs.is_empty() {
        model.file_specs.iter().collect()
    } else {
        return has_marker && base.is_dir();
    };
    if specs.is_empty() {
        return has_marker;
    }
    let all_exist = specs.iter().all(|spec| base.join(&spec.dest).is_file());
    if has_marker {
        return all_exist;
    }
    // 无标记但文件齐全也视为已下载（兼容旧版本）
    all_exist
}

fn downloaded_paths_for_model(
    model_root: &Path,
    model: &DownloadableModel,
) -> Option<(PathBuf, Option<PathBuf>)> {
    let base = download_base_dir(model_root, &model.id);
    if model.kind == "translation" {
        let spec = model.file_specs.first()?;
        return Some((base.join(&spec.dest), None));
    }
    // OCR: 推断 det/rec 子目录
    let mut det_dir: Option<String> = None;
    let mut rec_dir: Option<String> = None;
    for spec in &model.file_specs {
        if let Some(parent) = Path::new(&spec.dest).parent().and_then(|p| p.to_str()) {
            if parent.contains("det") && det_dir.is_none() {
                det_dir = Some(parent.to_owned());
            }
            if parent.contains("rec") && rec_dir.is_none() {
                rec_dir = Some(parent.to_owned());
            }
        }
    }
    let det = det_dir?;
    let rec = rec_dir?;
    Some((base.join(det), Some(base.join(rec))))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteDownloadedModelRequest {
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivateDownloadedModelRequest {
    pub model_id: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadedModelInfo {
    pub model_id: String,
    pub downloaded: bool,
    pub base_dir: String,
}

#[tauri::command]
pub fn list_downloaded_models(state: State<'_, BackendState>) -> Result<Vec<DownloadedModelInfo>, String> {
    let model_root: PathBuf = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?;
        settings.model_root.clone()
    };
    let models = downloadable_models();
    let mut result = Vec::new();
    for model in models {
        let downloaded = is_model_downloaded_on_disk(&model_root, &model);
        let base = download_base_dir(&model_root, &model.id);
        result.push(DownloadedModelInfo {
            model_id: model.id,
            downloaded,
            base_dir: base.to_string_lossy().to_string(),
        });
    }
    Ok(result)
}

#[tauri::command]
pub fn get_downloaded_model_paths(
    state: State<'_, BackendState>,
    request: GetStatusRequest,
) -> Result<Option<DownloadedModelInfo>, String> {
    let model_id = request.model_id.trim().to_owned();
    if model_id.is_empty() {
        return Err("modelId 不能为空".to_owned());
    }
    let model_root: PathBuf = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?;
        settings.model_root.clone()
    };
    let model = downloadable_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("未知模型 {model_id}"))?;
    let downloaded = is_model_downloaded_on_disk(&model_root, &model);
    let base = download_base_dir(&model_root, &model.id);
    Ok(Some(DownloadedModelInfo {
        model_id,
        downloaded,
        base_dir: base.to_string_lossy().to_string(),
    }))
}

#[tauri::command]
pub fn activate_downloaded_model(
    state: State<'_, BackendState>,
    request: ActivateDownloadedModelRequest,
) -> Result<crate::backend::settings::BackendStatus, String> {
    let model_id = request.model_id.trim().to_owned();
    if model_id.is_empty() {
        return Err("modelId 不能为空".to_owned());
    }
    // 检查实时会话是否活跃
    if state.live_active.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("实时翻译进行中，请先停止后再切换模型".to_owned());
    }
    let models = downloadable_models();
    let model = models
        .iter()
        .find(|m| m.id == model_id)
        .cloned()
        .ok_or_else(|| format!("未知模型 {model_id}"))?;

    // 读取当前设置与 model_root
    let (model_root, current) = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?.clone();
        (settings.model_root.clone(), settings)
    };

    if !is_model_downloaded_on_disk(&model_root, &model) {
        return Err(format!("模型 {model_id} 尚未下载完成，请先下载"));
    }

    let mut updated = current.clone();
    if model.kind == "translation" {
        let (hy_path, _) = downloaded_paths_for_model(&model_root, &model)
            .ok_or_else(|| "无法解析翻译模型路径".to_owned())?;
        if !hy_path.is_file() {
            return Err(format!("翻译模型文件不存在: {}", hy_path.display()));
        }
        updated.hy_model = hy_path;
    } else {
        let (det_dir, rec_opt) = downloaded_paths_for_model(&model_root, &model)
            .ok_or_else(|| "无法解析 OCR 模型路径".to_owned())?;
        let rec_dir = rec_opt.ok_or_else(|| "无法解析识别模型路径".to_owned())?;
        if !det_dir.is_dir() {
            return Err(format!("检测模型目录不存在: {}", det_dir.display()));
        }
        if !rec_dir.is_dir() {
            return Err(format!("识别模型目录不存在: {}", rec_dir.display()));
        }
        // 校验两侧变体一致
        {
            use crate::models::ppocr::assets::{GraphRole, PpOcrVariant};
            match (
                PpOcrVariant::probe(GraphRole::Detector, &det_dir),
                PpOcrVariant::probe(GraphRole::Recognizer, &rec_dir),
            ) {
                (Some(d), Some(r)) if d != r => {
                    return Err(format!(
                        "检测模型变体 {} 与识别模型变体 {} 不一致",
                        d.label(),
                        r.label()
                    ));
                }
                _ => {}
            }
        }
        updated.detector_model_dir = det_dir;
        updated.recognizer_model_dir = rec_dir;
    }

    // 持久化
    if let Some(config_path) = state.config_path.as_deref() {
        let parent = config_path.parent().ok_or_else(|| "模型设置路径无效".to_owned())?;
        std::fs::create_dir_all(parent).map_err(|e| format!("创建模型设置目录失败: {e}"))?;
        let content = serde_json::to_vec_pretty(&updated.persisted())
            .map_err(|e| format!("序列化模型设置失败: {e}"))?;
        std::fs::write(config_path, content).map_err(|e| format!("保存模型设置失败: {e}"))?;
    }
    {
        let mut guard = state.settings.lock().map_err(|_| "无法写入设置".to_owned())?;
        *guard = Ok(updated.clone());
    }
    // 清空已加载引擎，下次推理重新加载
    if let Ok(mut engine) = state.engine.lock() {
        *engine = None;
    }
    state.touch_activity();
    Ok(updated.status(false))
}
#[tauri::command]
pub fn delete_downloaded_model(
    state: State<'_, BackendState>,
    request: DeleteDownloadedModelRequest,
) -> Result<(), String> {
    let model_id = request.model_id.trim().to_owned();
    if model_id.is_empty() {
        return Err("modelId 不能为空".to_owned());
    }
    if state.live_active.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("实时翻译进行中，请先停止后再删除模型".to_owned());
    }
    let model = downloadable_models()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("未知模型 {model_id}"))?;
    let model_root: PathBuf = {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        let settings = guard.as_ref().map_err(|err| err.clone())?;
        settings.model_root.clone()
    };
    let base = download_base_dir(&model_root, &model.id);
    if !base.exists() {
        return Err(format!("模型 {model_id} 未下载，无需删除"));
    }
    // 禁止删除当前已启用的模型
    {
        let guard = state.settings.lock().map_err(|_| "无法读取设置".to_owned())?;
        if let Ok(settings) = guard.as_ref() {
            let cur_hy = settings.hy_model.to_string_lossy().to_string().replace('\\', "/").to_lowercase();
            let cur_det = settings.detector_model_dir.to_string_lossy().to_string().replace('\\', "/").to_lowercase();
            let cur_rec = settings.recognizer_model_dir.to_string_lossy().to_string().replace('\\', "/").to_lowercase();
            let base_str = base.to_string_lossy().to_string().replace('\\', "/").to_lowercase();
            let is_active = if model.kind == "translation" {
                cur_hy.starts_with(&base_str)
            } else {
                cur_det.starts_with(&base_str) || cur_rec.starts_with(&base_str)
            };
            if is_active {
                return Err(format!("模型 {model_id} 当前已启用，请先切换到其他模型后再删除"));
            }
        }
    }
    // 若有正在进行的下载，先取消
    if let Ok(mut handles) = DOWNLOAD_HANDLES.lock() {
        if let Some(handle) = handles.remove(&model_id) {
            handle.abort();
        }
    }
    if let Ok(mut flags) = CANCEL_FLAGS.lock() {
        flags.remove(&model_id);
    }
    if let Ok(mut map) = TASK_MAP.lock() {
        map.remove(&model_id);
    }
    // 删除目录
    std::fs::remove_dir_all(&base).map_err(|e| format!("删除模型目录失败: {e}"))?;
    Ok(())
}

