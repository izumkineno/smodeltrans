use crate::openai_compat::{
    adapter::TranslationPort,
    config::OpenAiCompatConfig,
    types::{
        ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, Delta, HealthResponse, ModelInfo,
        ModelList, new_chat_response, now_secs,
    },
};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde_json::json;
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub struct AppState {
    pub port: Arc<dyn TranslationPort>,
    pub config: Arc<RwLock<OpenAiCompatConfig>>,
}

impl AppState {
    #[allow(dead_code)]
    pub fn new(port: Arc<dyn TranslationPort>, config: OpenAiCompatConfig) -> Self {
        Self {
            port,
            config: Arc::new(RwLock::new(config)),
        }
    }
}

fn error_json(message: &str, code: &str) -> Json<serde_json::Value> {
    Json(json!({
        "error": {
            "message": message,
            "type": code,
            "code": code
        }
    }))
}

async fn check_auth(headers: &HeaderMap, config: &OpenAiCompatConfig) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if let Some(expected) = &config.api_key {
        if expected.trim().is_empty() {
            return Ok(());
        }
        let got = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let expected_header = format!("Bearer {}", expected.trim());
        if got != expected_header {
            return Err((StatusCode::UNAUTHORIZED, error_json("invalid api key", "invalid_api_key")));
        }
    }
    Ok(())
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let (ocr, hy) = state.port.model_states().unwrap_or((false, false));
    let cfg = state.config.read().await;
    let resp = HealthResponse {
        status: "ok".to_owned(),
        port: Some(cfg.port),
        model_loaded: hy,
        owned_by: "smodeltrans".to_owned(),
    };
    let _ = ocr;
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
}

async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let cfg = state.config.read().await.clone();
    if let Err(e) = check_auth(&headers, &cfg).await {
        return e.into_response();
    }
    let models = vec![ModelInfo {
        id: "hy2-mt".to_owned(),
        object: "model".to_owned(),
        created: now_secs(),
        owned_by: "smodeltrans".to_owned(),
    }];
    let list = ModelList {
        object: "list".to_owned(),
        data: models,
    };
    (StatusCode::OK, Json(serde_json::to_value(list).unwrap())).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let cfg = state.config.read().await.clone();
    if let Err(e) = check_auth(&headers, &cfg).await {
        return e.into_response();
    }

    if state.port.live_active() {
        return (StatusCode::SERVICE_UNAVAILABLE, error_json("live translation is active", "service_unavailable")).into_response();
    }

    if req.messages.is_empty() {
        return (StatusCode::BAD_REQUEST, error_json("messages 不能为空", "invalid_request_error")).into_response();
    }
    let source = req.plain_source_text();
    if source.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, error_json("messages content 不能为空", "invalid_request_error")).into_response();
    }
    if source.len() > 8 * 1024 * 1024 {
        return (StatusCode::BAD_REQUEST, error_json("text 超过 8 MiB 限制", "invalid_request_error")).into_response();
    }

    let target_language = req.target_language("Chinese");
    if target_language.len() > 64 {
        return (StatusCode::BAD_REQUEST, error_json("target_language 过长", "invalid_request_error")).into_response();
    }

    let generation = build_generation_override(&req);
    let is_stream = req.is_stream();
    let model = req.model.clone();

    if !is_stream {
        let port = Arc::clone(&state.port);
        let text = source.clone();
        let lang = target_language.clone();
        let r#gen = generation.clone();
        let blocking = tokio::task::spawn_blocking(move || port.translate_text(text, lang, r#gen));
        match blocking.await {
            Ok(Ok(translated)) => {
                let prompt_tokens = estimate_tokens(&source);
                let completion_tokens = estimate_tokens(&translated);
                let resp = new_chat_response(&model, &translated, prompt_tokens, completion_tokens);
                (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
            }
            Ok(Err(e)) => map_backend_error(e).into_response(),
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, error_json("worker 异常退出", "internal_error")).into_response(),
        }
    } else {
        let port = Arc::clone(&state.port);
        let text = source.clone();
        let lang = target_language.clone();
        let r#gen = generation.clone();
        let model_clone = model.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
        tokio::spawn(async move {
            let res = tokio::task::spawn_blocking(move || port.translate_text(text, lang, r#gen)).await;
            match res {
                Ok(Ok(full)) => {
                    let id = format!("chatcmpl-{}", now_secs());
                    let created = now_secs();
                    let _ = tx
                        .send(Ok(Event::default().json_data(serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_clone,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
                        })).unwrap()))
                        .await;
                    for chunk in split_for_stream(&full) {
                        let payload = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_owned(),
                            created,
                            model: model_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(chunk.to_owned()),
                                },
                                finish_reason: None,
                            }],
                        };
                        let _ = tx
                            .send(Ok(Event::default().json_data(payload).unwrap()))
                            .await;
                        tokio::time::sleep(Duration::from_millis(15)).await;
                    }
                    let _ = tx
                        .send(Ok(Event::default().json_data(serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_clone,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        })).unwrap()))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                }
                Ok(Err(e)) => {
                    let msg = e.to_string();
                    let _ = tx
                        .send(Ok(Event::default().json_data(json!({"error": {"message": msg, "type":"translation_error"}})).unwrap()))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                }
                Err(_) => {
                    let _ = tx
                        .send(Ok(Event::default().json_data(json!({"error": {"message":"worker 异常","type":"internal_error"}})).unwrap()))
                        .await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    }
}

fn build_generation_override(req: &ChatCompletionRequest) -> Option<crate::model_config::GenerationConfig> {
    if req.temperature.is_none() && req.top_p.is_none() && req.top_k.is_none() && req.max_tokens.is_none() && req.seed.is_none() {
        return None;
    }
    let mut r#gen = crate::model_config::GenerationConfig::default();
    if let Some(t) = req.temperature {
        r#gen.temperature = t.clamp(0.0, 2.0);
        r#gen.sampling = true;
    }
    if let Some(p) = req.top_p {
        r#gen.top_p = p.clamp(0.0, 1.0);
        r#gen.sampling = true;
    }
    if let Some(k) = req.top_k {
        r#gen.top_k = k.clamp(1, crate::model_config::MAX_TOP_K);
        r#gen.sampling = true;
    }
    if let Some(m) = req.max_tokens {
        r#gen.max_new_tokens = m.clamp(1, crate::model_config::MAX_NEW_TOKENS);
    }
    if let Some(seed) = req.seed {
        r#gen.seed = Some(seed);
    }
    if r#gen.validate().is_err() {
        return None;
    }
    Some(r#gen)
}

fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() + 1) / 2
}

fn split_for_stream(text: &str) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        buf.push(ch);
        if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '\n') {
            if buf.len() >= 6 {
                chunks.push(buf.clone());
                buf.clear();
            }
        }
        if buf.chars().count() >= 24 {
            chunks.push(buf.clone());
            buf.clear();
        }
    }
    if !buf.is_empty() {
        chunks.push(buf);
    }
    if chunks.is_empty() {
        chunks.push(text.to_owned());
    }
    chunks
}

fn map_backend_error(e: crate::backend::failure::BackendFailure) -> (StatusCode, Json<serde_json::Value>) {
    use crate::backend::failure::BackendFailureCode;
    let msg = e.to_string();
    if msg.contains("busy") {
        return (StatusCode::TOO_MANY_REQUESTS, error_json(&msg, "rate_limit"));
    }
    if msg.contains("live") {
        return (StatusCode::SERVICE_UNAVAILABLE, error_json(&msg, "service_unavailable"));
    }
    match e.code() {
        BackendFailureCode::Arguments => (StatusCode::BAD_REQUEST, error_json(&msg, "invalid_request_error")),
        BackendFailureCode::Cancelled => (StatusCode::REQUEST_TIMEOUT, error_json(&msg, "cancelled")),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, error_json(&msg, "internal_error")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_stream_basic() {
        let v = split_for_stream("Hello world. 你好世界。");
        assert!(!v.is_empty());
        assert!(v.join("").contains("Hello"));
    }

    #[test]
    fn generation_override_none_when_empty() {
        let req = ChatCompletionRequest {
            model: "hy-mt2-1.8b".to_owned(),
            messages: vec![],
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            target_language: None,
            language: None,
            extra: None,
        };
        assert!(build_generation_override(&req).is_none());
    }
}
