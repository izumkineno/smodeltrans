use crate::backend::commands::BackendState;
use crate::openai_compat::{config::OpenAiCompatConfig, server::OpenAiServerHandle};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Instant};
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
    tracing::debug!(target: "openai_compat::server", "get_openai_status called");
    let start = Instant::now();
    let resp = current_openai_status_sync(state.inner(), handle.inner());
    tracing::info!(
        target: "openai_compat::server",
        enabled = resp.enabled,
        host = %resp.host,
        port = resp.port,
        bound_port = ?resp.bound_port,
        running = resp.running,
        has_api_key = resp.has_api_key,
        duration_ms = start.elapsed().as_millis() as u64,
        "get_openai_status success"
    );
    Ok(resp)
}

#[tauri::command]
pub async fn update_openai_config(
    request: UpdateOpenAiRequest,
    state: State<'_, BackendState>,
    handle: State<'_, OpenAiServerHandle>,
) -> Result<OpenAiStatusResponse, String> {
    let start_total = Instant::now();
    tracing::info!(
        target: "openai_compat::server",
        enabled = request.enabled,
        host = %request.host,
        port = request.port,
        has_api_key = request.api_key.is_some(),
        api_key_len = request.api_key.as_ref().map(|k| k.len()).unwrap_or(0),
        "update_openai_config called"
    );
    let new_config = OpenAiCompatConfig {
        enabled: request.enabled,
        host: request.host.trim().to_owned(),
        port: request.port,
        api_key: request.api_key.as_ref().map(|k| k.trim().to_owned()).filter(|k| !k.is_empty()),
    };
    if let Err(e) = new_config.validate() {
        tracing::error!(
            target: "openai_compat::server",
            error = %e,
            host = %new_config.host,
            port = new_config.port,
            enabled = new_config.enabled,
            "update_openai_config validation failed"
        );
        return Err(e);
    }
    tracing::debug!(
        target: "openai_compat::server",
        host = %new_config.host,
        port = new_config.port,
        enabled = new_config.enabled,
        "update_openai_config validation succeeded"
    );

    // 更新 BackendSettings 中的 openai_compat 并持久化
    {
        let mut guard = state
            .settings
            .lock()
            .map_err(|_| {
                tracing::error!(target: "openai_compat::server", "update_openai_config failed: backend settings lock poisoned");
                "后端配置锁已损坏".to_owned()
            })?;
        let current = guard
            .as_ref()
            .map_err(|e| {
                tracing::error!(target: "openai_compat::server", error = %e, "update_openai_config failed: current config unavailable");
                format!("当前配置不可用: {}", e)
            })?
            .clone();
        let mut next = current;
        next.openai_compat = new_config.clone();
        *guard = Ok(next.clone());
        tracing::debug!(
            target: "openai_compat::server",
            host = %new_config.host,
            port = new_config.port,
            enabled = new_config.enabled,
            "BackendSettings updated in memory"
        );

        // 持久化到文件
        if let Some(path) = &state.config_path {
            tracing::debug!(target: "openai_compat::server", config_path = %path.display(), "persisting openai config");
            let persisted = next.persisted();
            let json = serde_json::to_string_pretty(&persisted).map_err(|e| {
                tracing::error!(target: "openai_compat::server", error = %e, "update_openai_config serialization failed");
                format!("序列化失败: {}", e)
            })?;
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(path, json).map_err(|e| {
                tracing::error!(target: "openai_compat::server", config_path = %path.display(), error = %e, "update_openai_config write failed");
                format!("写入配置失败: {}", e)
            })?;
            tracing::info!(target: "openai_compat::server", config_path = %path.display(), "openai config persisted");
        } else {
            tracing::warn!(target: "openai_compat::server", "update_openai_config: no config_path, skipping persistence");
        }
    }

    // 热重启服务
    let port: Arc<dyn crate::openai_compat::adapter::TranslationPort> =
        Arc::new(crate::openai_compat::adapter::BackendStateAdapter::new(state.inner().clone()));
    // 需 clone handle for async
    let h = handle.inner().clone();
    tracing::info!(
        target: "openai_compat::server",
        host = %new_config.host,
        port = new_config.port,
        enabled = new_config.enabled,
        "update_openai_config restarting server via handle.update_config"
    );
    let restart_start = Instant::now();
    // update_config 会 abort 旧并尝试绑定新端口
    let result = h.update_config(new_config.clone(), port).await;
    match &result {
        Ok(bound) => {
            tracing::info!(
                target: "openai_compat::server",
                host = %new_config.host,
                port = new_config.port,
                bound = ?bound,
                duration_ms = restart_start.elapsed().as_millis() as u64,
                total_duration_ms = start_total.elapsed().as_millis() as u64,
                "update_openai_config restart succeeded"
            );
        }
        Err(e) => {
            tracing::error!(
                target: "openai_compat::server",
                host = %new_config.host,
                port = new_config.port,
                error = %e,
                duration_ms = restart_start.elapsed().as_millis() as u64,
                total_duration_ms = start_total.elapsed().as_millis() as u64,
                "update_openai_config restart failed"
            );
        }
    }
    result.map_err(|e| e.to_string())?;

    let resp = current_openai_status_sync(state.inner(), handle.inner());
    tracing::info!(
        target: "openai_compat::server",
        enabled = resp.enabled,
        host = %resp.host,
        port = resp.port,
        bound_port = ?resp.bound_port,
        running = resp.running,
        total_duration_ms = start_total.elapsed().as_millis() as u64,
        "update_openai_config completed"
    );
    Ok(resp)
}
