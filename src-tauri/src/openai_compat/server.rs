use crate::openai_compat::{adapter::TranslationPort, config::OpenAiCompatConfig, routes::{AppState, build_router}};
use std::{net::SocketAddr, sync::{Arc, atomic::{AtomicU16, Ordering}}, time::Instant};
use tokio::{sync::RwLock, task::JoinHandle};

pub struct OpenAiServerHandle {
    pub config: Arc<RwLock<OpenAiCompatConfig>>,
    pub bound_port: Arc<AtomicU16>,
    handle: Arc<tokio::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl OpenAiServerHandle {
    pub fn new(initial: OpenAiCompatConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(initial)),
            bound_port: Arc::new(AtomicU16::new(0)),
            handle: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    #[allow(dead_code)]
    pub async fn bound_port_value(&self) -> Option<u16> {
        let v = self.bound_port.load(Ordering::SeqCst);
        if v == 0 { None } else { Some(v) }
    }

    #[allow(dead_code)]
    pub async fn is_running(&self) -> bool {
        if let Some(h) = self.handle.lock().await.as_ref() {
            !h.is_finished()
        } else {
            false
        }
    }

    pub async fn update_config(&self, new_config: OpenAiCompatConfig, port: Arc<dyn TranslationPort>) -> Result<Option<u16>, String> {
        tracing::info!(
            target: "openai_compat::server",
            enabled = new_config.enabled,
            host = %new_config.host,
            port = new_config.port,
            has_api_key = new_config.api_key.is_some(),
            "update_config called"
        );
        if let Err(e) = new_config.validate() {
            tracing::error!(
                target: "openai_compat::server",
                error = %e,
                host = %new_config.host,
                port = new_config.port,
                "update_config validation failed"
            );
            return Err(e);
        }
        {
            let mut w = self.config.write().await;
            *w = new_config.clone();
        }
        if !new_config.enabled {
            tracing::info!(target: "openai_compat::server", host = %new_config.host, port = new_config.port, "update_config disabling server, stopping");
            self.stop().await;
            self.bound_port.store(0, Ordering::SeqCst);
            tracing::info!(target: "openai_compat::server", "update_config disabled, server stopped");
            return Ok(None);
        }
        tracing::info!(target: "openai_compat::server", host = %new_config.host, port = new_config.port, "update_config restarting server");
        self.stop().await;
        let start = Instant::now();
        match self.start_inner(port).await {
            Ok(bound) => {
                tracing::info!(
                    target: "openai_compat::server",
                    host = %new_config.host,
                    port = new_config.port,
                    bound = bound,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "update_config start succeeded"
                );
                Ok(Some(bound))
            }
            Err(e) => {
                tracing::error!(
                    target: "openai_compat::server",
                    error = %e,
                    host = %new_config.host,
                    port = new_config.port,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "update_config start failed"
                );
                Err(e)
            }
        }
    }

    pub async fn start(&self, port: Arc<dyn TranslationPort>) -> Result<u16, String> {
        let cfg = self.config.read().await.clone();
        tracing::info!(
            target: "openai_compat::server",
            enabled = cfg.enabled,
            host = %cfg.host,
            port = cfg.port,
            has_api_key = cfg.api_key.is_some(),
            "start called"
        );
        if !cfg.enabled {
            tracing::warn!(target: "openai_compat::server", host = %cfg.host, port = cfg.port, "start rejected: compat not enabled");
            return Err("openai compat not enabled".to_owned());
        }
        let start = Instant::now();
        match self.start_inner(port).await {
            Ok(bound) => {
                tracing::info!(
                    target: "openai_compat::server",
                    host = %cfg.host,
                    port = cfg.port,
                    bound = bound,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "start succeeded"
                );
                Ok(bound)
            }
            Err(e) => {
                tracing::error!(
                    target: "openai_compat::server",
                    error = %e,
                    host = %cfg.host,
                    port = cfg.port,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "start failed"
                );
                Err(e)
            }
        }
    }

    async fn start_inner(&self, port: Arc<dyn TranslationPort>) -> Result<u16, String> {
        let cfg = self.config.read().await.clone();
        let start_overall = Instant::now();
        tracing::info!(
            target: "openai_compat::server",
            host = %cfg.host,
            port = cfg.port,
            enabled = cfg.enabled,
            "start_inner begin"
        );
        if let Err(e) = cfg.validate() {
            tracing::error!(target: "openai_compat::server", error = %e, host = %cfg.host, port = cfg.port, "start_inner validation failed");
            return Err(e);
        }
        let host = cfg.host.clone();
        let base_port = cfg.port;
        let bound_port = Arc::clone(&self.bound_port);
        let config_shared = Arc::clone(&self.config);

        let mut last_err = String::new();
        for offset in 0..4 {
            let try_port = base_port.wrapping_add(offset);
            if try_port == 0 { continue; }
            let addr_str = format!("{}:{}", host, try_port);
            let addr: SocketAddr = addr_str.parse().map_err(|e| {
                let msg = format!("invalid addr {}: {}", addr_str, e);
                tracing::error!(target: "openai_compat::server", error = %e, addr = %addr_str, "addr parse failed");
                msg
            })?;
            let app_state = AppState {
                port: Arc::clone(&port),
                config: Arc::clone(&config_shared),
            };
            let router = build_router(app_state);
            let router = if cfg.enabled {
                router.layer(tower_http::cors::CorsLayer::permissive())
            } else {
                router
            };
            tracing::debug!(target: "openai_compat::server", addr = %addr_str, offset = offset, try_port = try_port, "attempting bind");
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    let bound = listener.local_addr().map(|a| a.port()).unwrap_or(try_port);
                    bound_port.store(bound, Ordering::SeqCst);
                    let addr_str_for_task = addr_str.clone();
                    let handle = tokio::spawn(async move {
                        if let Err(e) = axum::serve(listener, router).await {
                            tracing::error!(target: "openai_compat::server", error = %e, addr = %addr_str_for_task, "server error");
                        } else {
                            tracing::info!(target: "openai_compat::server", addr = %addr_str_for_task, "server stopped gracefully");
                        }
                    });
                    *self.handle.lock().await = Some(handle);
                    tracing::info!(
                        target: "openai_compat::server",
                        addr = %addr_str,
                        requested = base_port,
                        offset = offset,
                        bound = bound,
                        duration_ms = start_overall.elapsed().as_millis() as u64,
                        "listening"
                    );
                    return Ok(bound);
                }
                Err(e) => {
                    last_err = format!("bind {} failed: {}", addr_str, e);
                    tracing::warn!(target: "openai_compat::server", error = %e, addr = %addr_str, offset = offset, try_port = try_port, "bind failed, trying next offset");
                    continue;
                }
            }
        }
        tracing::error!(
            target: "openai_compat::server",
            error = %last_err,
            host = %host,
            base_port = base_port,
            duration_ms = start_overall.elapsed().as_millis() as u64,
            "all bind attempts failed"
        );
        Err(format!("all bind attempts failed, last: {}", last_err))
    }

    pub async fn stop(&self) {
        let prev_bound = self.bound_port.load(Ordering::SeqCst);
        tracing::info!(target: "openai_compat::server", prev_bound = prev_bound, "stop called");
        let mut guard = self.handle.lock().await;
        if let Some(h) = guard.take() {
            tracing::debug!(target: "openai_compat::server", prev_bound = prev_bound, "aborting server task");
            h.abort();
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            tracing::info!(target: "openai_compat::server", prev_bound = prev_bound, "server task aborted");
        } else {
            tracing::debug!(target: "openai_compat::server", "stop called with no active handle");
        }
        self.bound_port.store(0, Ordering::SeqCst);
        tracing::debug!(target: "openai_compat::server", prev_bound = prev_bound, "bound_port reset to 0");
        tracing::info!(target: "openai_compat::server", prev_bound = prev_bound, "stop completed");
    }
}

impl Clone for OpenAiServerHandle {
    fn clone(&self) -> Self {
        Self {
            config: Arc::clone(&self.config),
            bound_port: Arc::clone(&self.bound_port),
            handle: Arc::clone(&self.handle),
        }
    }
}

#[allow(dead_code)]
pub async fn start_server(
    port: Arc<dyn TranslationPort>,
    config: OpenAiCompatConfig,
) -> Result<(OpenAiServerHandle, u16), String> {
    let handle = OpenAiServerHandle::new(config);
    let bound = handle.start(Arc::clone(&port)).await?;
    Ok((handle, bound))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compat::{adapter::mock::MockPort, config::OpenAiCompatConfig};
    use std::sync::Arc;

    #[tokio::test]
    async fn start_and_stop() {
        let cfg = OpenAiCompatConfig {
            enabled: true,
            host: "127.0.0.1".to_owned(),
            port: 11880,
            api_key: None,
        };
        let port: Arc<dyn TranslationPort> = Arc::new(MockPort { return_text: "你好".to_owned() });
        let handle = OpenAiServerHandle::new(cfg);
        let bound = handle.start(port.clone()).await.expect("start");
        assert!(bound >= 11880);
        assert!(handle.is_running().await);
        handle.stop().await;
        assert!(!handle.is_running().await);
    }
}
