use crate::openai_compat::{adapter::TranslationPort, config::OpenAiCompatConfig, routes::{AppState, build_router}};
use std::{net::SocketAddr, sync::{Arc, atomic::{AtomicU16, Ordering}}};
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
        new_config.validate()?;
        {
            let mut w = self.config.write().await;
            *w = new_config.clone();
        }
        if !new_config.enabled {
            self.stop().await;
            self.bound_port.store(0, Ordering::SeqCst);
            return Ok(None);
        }
        self.stop().await;
        let bound = self.start_inner(port).await?;
        Ok(Some(bound))
    }

    pub async fn start(&self, port: Arc<dyn TranslationPort>) -> Result<u16, String> {
        let cfg = self.config.read().await.clone();
        if !cfg.enabled {
            return Err("openai compat not enabled".to_owned());
        }
        self.start_inner(port).await
    }

    async fn start_inner(&self, port: Arc<dyn TranslationPort>) -> Result<u16, String> {
        let cfg = self.config.read().await.clone();
        cfg.validate()?;
        let host = cfg.host.clone();
        let base_port = cfg.port;
        let bound_port = Arc::clone(&self.bound_port);
        let config_shared = Arc::clone(&self.config);

        let mut last_err = String::new();
        for offset in 0..4 {
            let try_port = base_port.wrapping_add(offset);
            if try_port == 0 { continue; }
            let addr_str = format!("{}:{}", host, try_port);
            let addr: SocketAddr = addr_str.parse().map_err(|e| format!("invalid addr {}: {}", addr_str, e))?;
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
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    let bound = listener.local_addr().map(|a| a.port()).unwrap_or(try_port);
                    bound_port.store(bound, Ordering::SeqCst);
                    let handle = tokio::spawn(async move {
                        if let Err(e) = axum::serve(listener, router).await {
                            eprintln!("[openai_compat] server error: {}", e);
                        }
                    });
                    *self.handle.lock().await = Some(handle);
                    eprintln!("[openai_compat] listening on {} (requested {}+{})", addr_str, base_port, offset);
                    return Ok(bound);
                }
                Err(e) => {
                    last_err = format!("bind {} failed: {}", addr_str, e);
                    eprintln!("[openai_compat] {}", last_err);
                    continue;
                }
            }
        }
        Err(format!("all bind attempts failed, last: {}", last_err))
    }

    pub async fn stop(&self) {
        let mut guard = self.handle.lock().await;
        if let Some(h) = guard.take() {
            h.abort();
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        }
        self.bound_port.store(0, Ordering::SeqCst);
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
