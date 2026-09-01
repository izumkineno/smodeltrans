use crate::backend::commands::BackendState;
use crate::openai_compat::{config::OpenAiCompatConfig, server::OpenAiServerHandle};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiStatusResponse {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub bound_port: Option<u16>,
    pub running: bool,
    pub has_api_key: bool,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateOpenAiRequest {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
}

fn current_openai_status_sync(
    state: &BackendState,
    handle: &OpenAiServerHandle,
) -> OpenAiStatusResponse {
    let settings = state
        .settings
        .lock()
        .ok()
        .and_then(|s| s.as_ref().ok().cloned())
        .map(|s| s.openai_compat)
        .unwrap_or_default();
    let bound = handle.bound_port.load(std::sync::atomic::Ordering::SeqCst);
    let bound_opt = if bound == 0 { None } else { Some(bound) };
    let running = bound != 0;
    OpenAiStatusResponse {
        enabled: settings.enabled,
        host: settings.host.clone(),
        port: settings.port,
        bound_port: bound_opt,
        running: running && settings.enabled,
        has_api_key: settings.api_key.as_ref().map(|k| !k.is_empty()).unwrap_or(false),
        message: if settings.enabled && bound_opt.is_none() {
            "服务未运行，检查端口是否被占用".to_owned()
        } else if settings.enabled {
            format!("运行中 http://{}:{}", settings.host, bound_opt.unwrap_or(settings.port))
        } else {
            "已禁用".to_owned()
        },
    }
}

#[tauri::command]
pub async fn get_openai_status(
    state: State<'_, BackendState>,
    handle: State<'_, OpenAiServerHandle>,
) -> Result<OpenAiStatusResponse, String> {
    Ok(current_openai_status_sync(state.inner(), handle.inner()))
}

#[tauri::command]
pub async fn update_openai_config(
    request: UpdateOpenAiRequest,
    state: State<'_, BackendState>,
    handle: State<'_, OpenAiServerHandle>,
) -> Result<OpenAiStatusResponse, String> {
    let new_config = OpenAiCompatConfig {
        enabled: request.enabled,
        host: request.host.trim().to_owned(),
        port: request.port,
        api_key: request.api_key.as_ref().map(|k| k.trim().to_owned()).filter(|k| !k.is_empty()),
    };
    new_config.validate()?;

    // 更新 BackendSettings 中的 openai_compat 并持久化
    {
        let mut guard = state
            .settings
            .lock()
            .map_err(|_| "后端配置锁已损坏".to_owned())?;
        let current = guard
            .as_ref()
            .map_err(|e| format!("当前配置不可用: {}", e))?
            .clone();
        let mut next = current;
        next.openai_compat = new_config.clone();
        *guard = Ok(next.clone());

        // 持久化到文件
        if let Some(path) = &state.config_path {
            let persisted = next.persisted();
            let json = serde_json::to_string_pretty(&persisted).map_err(|e| format!("序列化失败: {}", e))?;
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, json).map_err(|e| format!("写入配置失败: {}", e))?;
        }
    }

    // 热重启服务
    let port: Arc<dyn crate::openai_compat::adapter::TranslationPort> =
        Arc::new(crate::openai_compat::adapter::BackendStateAdapter::new(state.inner().clone()));
    // 需 clone handle for async
    let h = handle.inner().clone();
    // update_config 会 abort 旧并尝试绑定新端口
    h.update_config(new_config.clone(), port)
        .await
        .map_err(|e| e.to_string())?;

    Ok(current_openai_status_sync(state.inner(), handle.inner()))
}
