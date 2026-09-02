use crate::openai_compat::{
    config::OpenAiCompatConfig,
    history::{OpenAiHistoryEntry, OpenAiHistoryStore},
    types::{
        ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, Delta, HealthResponse, ModelInfo,
        ModelList, new_chat_response, new_chat_response_with_image, now_secs, peel_data_url,
    },
};
use base64::Engine as _;
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
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_request_id() -> String {
    let id = REQUEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("oai-{:08x}-{:04x}", id >> 16, id & 0xffff)
}

#[derive(Clone)]
pub struct AppState {
    pub port: Arc<dyn crate::openai_compat::adapter::TranslationPort>,
    pub config: Arc<RwLock<OpenAiCompatConfig>>,
    pub history: OpenAiHistoryStore,
}

impl AppState {
    #[allow(dead_code)]
    pub fn new(port: Arc<dyn crate::openai_compat::adapter::TranslationPort>, config: OpenAiCompatConfig) -> Self {
        Self {
            port,
            config: Arc::new(RwLock::new(config)),
            history: OpenAiHistoryStore::default(),
        }
    }

    pub fn with_history(
        port: Arc<dyn crate::openai_compat::adapter::TranslationPort>,
        config: OpenAiCompatConfig,
        history: OpenAiHistoryStore,
    ) -> Self {
        Self {
            port,
            config: Arc::new(RwLock::new(config)),
            history,
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
            tracing::debug!(target: "openai_compat::routes", has_api_key = false, "check_auth skipped: empty key");
            return Ok(());
        }
        let got = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let expected_header = format!("Bearer {}", expected.trim());
        if got != expected_header {
            tracing::warn!(
                target: "openai_compat::routes",
                has_auth_header = !got.is_empty(),
                "check_auth failed: invalid api key"
            );
            return Err((StatusCode::UNAUTHORIZED, error_json("invalid api key", "invalid_api_key")));
        }
        tracing::debug!(target: "openai_compat::routes", "check_auth succeeded");
    } else {
        tracing::trace!(target: "openai_compat::routes", "check_auth skipped: no api_key configured");
    }
    Ok(())
}

pub fn build_router(state: AppState) -> Router {
    tracing::info!(target: "openai_compat::routes", "build_router called");
    let router = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .with_state(state);
    tracing::debug!(target: "openai_compat::routes", "router built with 4 routes");
    router
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let request_id = next_request_id();
    let start = Instant::now();
    tracing::info!(target: "openai_compat::routes", request_id = %request_id, "health check started");
    let (ocr, hy) = state.port.model_states().unwrap_or((false, false));
    let cfg = state.config.read().await;
    let resp = HealthResponse {
        status: "ok".to_owned(),
        port: Some(cfg.port),
        model_loaded: hy,
        owned_by: "smodeltrans".to_owned(),
    };
    let _ = ocr;
    tracing::info!(
        target: "openai_compat::routes",
        request_id = %request_id,
        model_loaded = hy,
        ocr_loaded = ocr,
        port = cfg.port,
        duration_ms = start.elapsed().as_millis() as u64,
        "health check completed"
    );
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
}

async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let request_id = next_request_id();
    let start = Instant::now();
    tracing::info!(target: "openai_compat::routes", request_id = %request_id, "list_models started");
    let cfg = state.config.read().await.clone();
    if let Err(e) = check_auth(&headers, &cfg).await {
        tracing::warn!(
            target: "openai_compat::routes",
            request_id = %request_id,
            status = ?e.0,
            duration_ms = start.elapsed().as_millis() as u64,
            "list_models auth failed"
        );
        return e.into_response();
    }
    let models = vec![ModelInfo {
        id: "hy-mt2".to_owned(),
        object: "model".to_owned(),
        created: now_secs(),
        owned_by: "smodeltrans".to_owned(),
    }];
    let list = ModelList {
        object: "list".to_owned(),
        data: models,
    };
    tracing::info!(
        target: "openai_compat::routes",
        request_id = %request_id,
        model_count = 1,
        duration_ms = start.elapsed().as_millis() as u64,
        "list_models success"
    );
    (StatusCode::OK, Json(serde_json::to_value(list).unwrap())).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let request_id = next_request_id();
    let start = Instant::now();
    let model = req.model.clone();
    let is_stream = req.is_stream();
    // prompt length measured from plain_source_text for accurate translation input
    let source_preview_len = req.plain_source_text().chars().count();
    let target_language_preview = req.target_language("Chinese");
    tracing::info!(
        target: "openai_compat::routes",
        request_id = %request_id,
        model = %model,
        prompt_len = source_preview_len,
        prompt_bytes = req.plain_source_text().len(),
        streaming = is_stream,
        target_language = %target_language_preview,
        has_temperature = req.temperature.is_some(),
        has_top_p = req.top_p.is_some(),
        "chat_completions received"
    );

    let cfg = state.config.read().await.clone();
    if let Err(e) = check_auth(&headers, &cfg).await {
        tracing::warn!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            duration_ms = start.elapsed().as_millis() as u64,
            "chat_completions auth failed"
        );
        return e.into_response();
    }

    if state.port.live_active() {
        tracing::warn!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            duration_ms = start.elapsed().as_millis() as u64,
            "chat_completions rejected: live translation active"
        );
        return (StatusCode::SERVICE_UNAVAILABLE, error_json("live translation is active", "service_unavailable")).into_response();
    }

    if req.messages.is_empty() {
        tracing::warn!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            "chat_completions rejected: messages empty"
        );
        return (StatusCode::BAD_REQUEST, error_json("messages 不能为空", "invalid_request_error")).into_response();
    }
    let target_language = req.target_language("Chinese");
    if target_language.len() > 64 {
        tracing::warn!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            target_language = %target_language,
            target_len = target_language.len(),
            "chat_completions rejected: target_language too long"
        );
        return (StatusCode::BAD_REQUEST, error_json("target_language 过长", "invalid_request_error")).into_response();
    }

    let generation = build_generation_override(&req);
    let supplemental = req.supplemental_prompt();
    let supplemental_len = supplemental.chars().count();
    let image_urls = req.image_urls();
    let has_image = !image_urls.is_empty();
    if has_image {
        if is_stream {
            tracing::warn!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                streaming = is_stream,
                image_count = image_urls.len(),
                duration_ms = start.elapsed().as_millis() as u64,
                "chat_completions rejected: stream with image_url not supported"
            );
            return (StatusCode::BAD_REQUEST, error_json("stream with image_url not supported", "invalid_request_error")).into_response();
        }
        if image_urls.len() > 8 {
            tracing::warn!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                streaming = is_stream,
                image_count = image_urls.len(),
                duration_ms = start.elapsed().as_millis() as u64,
                "chat_completions rejected: too many images"
            );
            return (StatusCode::BAD_REQUEST, error_json("too many images, max 8", "invalid_request_error")).into_response();
        }
        let image_bytes_est: usize = image_urls
            .iter()
            .map(|u| peel_data_url(u).map(|b| b.len()).unwrap_or(u.len()))
            .sum();
        tracing::info!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            target_language = %target_language,
            image_count = image_urls.len(),
            image_bytes_est = image_bytes_est,
            supplemental_len = supplemental_len,
            has_generation_override = generation.is_some(),
            "chat_completions image branch entered"
        );
        tracing::debug!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            streaming = is_stream,
            target_language = %target_language,
            supplemental_len = supplemental_len,
            has_generation_override = generation.is_some(),
            image_count = image_urls.len(),
            image_bytes_est = image_bytes_est,
            "chat_completions validated, dispatching image translation"
        );
        let port = Arc::clone(&state.port);
        let history = state.history.clone();
        let urls = image_urls.clone();
        let lang = target_language.clone();
        let supp = supplemental.clone();
        let r#gen = generation.clone();
        let model_clone = model.clone();
        let request_id_clone = request_id.clone();
        let target_language_clone = target_language.clone();
        let image_count = image_urls.len();
        let blocking = tokio::task::spawn_blocking(move || -> Result<(String, Option<String>), crate::backend::failure::BackendFailure> {
            let mut texts = Vec::with_capacity(urls.len());
            let mut first_image_data_url: Option<String> = None;
            for (idx, url) in urls.iter().enumerate() {
                let b64 = peel_data_url(url).map_err(crate::backend::failure::BackendFailure::arguments)?;
                let file_name = format!("openai-image-{:03}.png", idx + 1);
                let out = port.translate_image(b64, file_name, lang.clone(), supp.clone(), r#gen.clone())?;
                texts.push(out.text.clone());
                if first_image_data_url.is_none() && !out.annotated_png.is_empty() {
                    let b64png = base64::engine::general_purpose::STANDARD.encode(&out.annotated_png);
                    first_image_data_url = Some(format!("data:image/png;base64,{b64png}"));
                }
            }
            let joined = texts.join("\n\n");
            Ok((joined, first_image_data_url))
        });
        match blocking.await {
            Ok(Ok((joined, image_opt))) => {
                let prompt_tokens = estimate_tokens(&joined);
                let completion_tokens = estimate_tokens(&joined);
                let resp = new_chat_response_with_image(&model_clone, joined.clone(), image_opt, prompt_tokens, completion_tokens);
                let duration_ms = start.elapsed().as_millis() as u64;
                let history_source = format!("[{} images] {}", image_count, joined.chars().take(200).collect::<String>());
                let entry = OpenAiHistoryEntry::new(
                    model_clone.clone(),
                    history_source,
                    joined.clone(),
                    target_language_clone.clone(),
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    false,
                );
                history.push(entry);
                tracing::info!(
                    target: "openai_compat::routes",
                    request_id = %request_id_clone,
                    model = %model_clone,
                    image_count = image_count,
                    image_bytes_est = image_bytes_est,
                    completion_len = joined.chars().count(),
                    completion_bytes = joined.len(),
                    prompt_tokens = prompt_tokens,
                    completion_tokens = completion_tokens,
                    streaming = is_stream,
                    target_language = %target_language_clone,
                    supplemental_len = supplemental_len,
                    duration_ms = duration_ms,
                    history_len = history.len(),
                    "chat_completions image non-stream success"
                );
                (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    target: "openai_compat::routes",
                    request_id = %request_id,
                    model = %model,
                    streaming = is_stream,
                    target_language = %target_language,
                    image_count = image_urls.len(),
                    image_bytes_est = image_bytes_est,
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "chat_completions image backend error"
                );
                map_backend_error(e).into_response()
            }
            Err(e) => {
                tracing::error!(
                    target: "openai_compat::routes",
                    request_id = %request_id,
                    model = %model,
                    streaming = is_stream,
                    image_count = image_urls.len(),
                    image_bytes_est = image_bytes_est,
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "chat_completions image worker panic"
                );
                (StatusCode::INTERNAL_SERVER_ERROR, error_json("worker 异常退出", "internal_error")).into_response()
            }
        }
    } else {
        let source = req.plain_source_text();
        let prompt_len = source.chars().count();
        let prompt_bytes = source.len();
        if source.trim().is_empty() {
            tracing::warn!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                streaming = is_stream,
                prompt_len = prompt_len,
                "chat_completions rejected: source empty"
            );
            return (StatusCode::BAD_REQUEST, error_json("messages content 不能为空", "invalid_request_error")).into_response();
        }
        if source.len() > 8 * 1024 * 1024 {
            tracing::warn!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                streaming = is_stream,
                prompt_bytes = prompt_bytes,
                "chat_completions rejected: text over 8MiB"
            );
            return (StatusCode::BAD_REQUEST, error_json("text 超过 8 MiB 限制", "invalid_request_error")).into_response();
        }
        tracing::debug!(
            target: "openai_compat::routes",
            request_id = %request_id,
            model = %model,
            prompt_len = prompt_len,
            prompt_bytes = prompt_bytes,
            streaming = is_stream,
            target_language = %target_language,
            has_generation_override = generation.is_some(),
            supplemental_len = supplemental_len,
            "chat_completions validated, dispatching"
        );
        if !is_stream {
            let port = Arc::clone(&state.port);
            let history = state.history.clone();
            let text = source.clone();
            let lang = target_language.clone();
            let supp = supplemental.clone();
            let r#gen = generation.clone();
            tracing::debug!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                prompt_len = prompt_len,
                streaming = is_stream,
                target_language = %lang,
                supplemental_len = supplemental_len,
                "chat_completions non-stream blocking dispatch"
            );
            let blocking = tokio::task::spawn_blocking(move || port.translate_text_with_supplemental(text, lang, supp, r#gen));
            match blocking.await {
                Ok(Ok(translated)) => {
                    let prompt_tokens = estimate_tokens(&source);
                    let completion_tokens = estimate_tokens(&translated);
                    let resp = new_chat_response(&model, &translated, prompt_tokens, completion_tokens);
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let entry = OpenAiHistoryEntry::new(
                        model.clone(),
                        source.clone(),
                        translated.clone(),
                        target_language.clone(),
                        duration_ms,
                        prompt_tokens,
                        completion_tokens,
                        false,
                    );
                    history.push(entry);
                    tracing::info!(
                        target: "openai_compat::routes",
                        request_id = %request_id,
                        model = %model,
                        prompt_len = prompt_len,
                        prompt_bytes = prompt_bytes,
                        completion_len = translated.chars().count(),
                        completion_bytes = translated.len(),
                        prompt_tokens = prompt_tokens,
                        completion_tokens = completion_tokens,
                        streaming = is_stream,
                        target_language = %target_language,
                        supplemental_len = supplemental_len,
                        duration_ms = duration_ms,
                        history_len = history.len(),
                        "chat_completions non-stream success"
                    );
                    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        target: "openai_compat::routes",
                        request_id = %request_id,
                        model = %model,
                        prompt_len = prompt_len,
                        streaming = is_stream,
                        target_language = %target_language,
                        error = %e,
                        duration_ms = start.elapsed().as_millis() as u64,
                        "chat_completions backend error"
                    );
                    map_backend_error(e).into_response()
                },
                Err(e) => {
                    tracing::error!(
                        target: "openai_compat::routes",
                        request_id = %request_id,
                        model = %model,
                        prompt_len = prompt_len,
                        streaming = is_stream,
                        error = %e,
                        duration_ms = start.elapsed().as_millis() as u64,
                        "chat_completions worker panic"
                    );
                    (StatusCode::INTERNAL_SERVER_ERROR, error_json("worker 异常退出", "internal_error")).into_response()
                },
            }
        } else {
            let port = Arc::clone(&state.port);
            let history = state.history.clone();
            let text = source.clone();
            let lang = target_language.clone();
            let supp = supplemental.clone();
            let r#gen = generation.clone();
            let model_clone = model.clone();
            let source_clone = source.clone();
            let lang_clone = lang.clone();
            let request_id_clone = request_id.clone();
            tracing::info!(
                target: "openai_compat::routes",
                request_id = %request_id,
                model = %model,
                prompt_len = prompt_len,
                prompt_bytes = prompt_bytes,
                streaming = is_stream,
                target_language = %target_language,
                supplemental_len = supplemental_len,
                "chat_completions streaming response initiated"
            );
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
            tokio::spawn(async move {
                let stream_start = Instant::now();
                tracing::debug!(
                    target: "openai_compat::routes",
                    request_id = %request_id_clone,
                    model = %model_clone,
                    prompt_len = prompt_len,
                    streaming = true,
                    target_language = %lang,
                    supplemental_len = supp.chars().count(),
                    "stream worker started"
                );
                let res = tokio::task::spawn_blocking(move || port.translate_text_with_supplemental(text, lang, supp, r#gen)).await;
                match res {
                    Ok(Ok(full)) => {
                        let chunk_count = split_for_stream(&full).len();
                        let duration_ms = stream_start.elapsed().as_millis() as u64;
                        let prompt_tokens = estimate_tokens(&source_clone);
                        let completion_tokens = estimate_tokens(&full);
                        let entry = OpenAiHistoryEntry::new(
                            model_clone.clone(),
                            source_clone.clone(),
                            full.clone(),
                            lang_clone.clone(),
                            duration_ms,
                            prompt_tokens,
                            completion_tokens,
                            true,
                        );
                        history.push(entry);
                        tracing::info!(
                            target: "openai_compat::routes",
                            request_id = %request_id_clone,
                            model = %model_clone,
                            completion_len = full.chars().count(),
                            completion_bytes = full.len(),
                            chunk_count = chunk_count,
                            duration_ms = duration_ms,
                            prompt_tokens = prompt_tokens,
                            completion_tokens = completion_tokens,
                            streaming = true,
                            history_len = history.len(),
                            "stream translation success, starting SSE"
                        );
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
                        tracing::debug!(
                            target: "openai_compat::routes",
                            request_id = %request_id_clone,
                            model = %model_clone,
                            duration_ms = stream_start.elapsed().as_millis() as u64,
                            "stream SSE completed"
                        );
                    }
                    Ok(Err(e)) => {
                        let msg = e.to_string();
                        tracing::warn!(
                            target: "openai_compat::routes",
                            request_id = %request_id_clone,
                            model = %model_clone,
                            error = %e,
                            error_msg = %msg,
                            duration_ms = stream_start.elapsed().as_millis() as u64,
                            streaming = true,
                            "stream translation backend error"
                        );
                        let _ = tx
                            .send(Ok(Event::default().json_data(json!({"error": {"message": msg, "type":"translation_error"}})).unwrap()))
                            .await;
                        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "openai_compat::routes",
                            request_id = %request_id_clone,
                            model = %model_clone,
                            error = %e,
                            duration_ms = stream_start.elapsed().as_millis() as u64,
                            streaming = true,
                            "stream worker panic"
                        );
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
}

fn build_generation_override(req: &ChatCompletionRequest) -> Option<crate::model_config::GenerationConfig> {
    let has_override = req.temperature.is_some() || req.top_p.is_some() || req.top_k.is_some() || req.max_tokens.is_some() || req.seed.is_some();
    if !has_override {
        tracing::trace!(target: "openai_compat::routes", "build_generation_override: no override");
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
    if let Err(e) = r#gen.validate() {
        tracing::warn!(target: "openai_compat::routes", error = %e, "build_generation_override validation failed, ignoring override");
        return None;
    }
    tracing::debug!(
        target: "openai_compat::routes",
        temperature = r#gen.temperature,
        top_p = r#gen.top_p,
        top_k = r#gen.top_k,
        max_new_tokens = r#gen.max_new_tokens,
        seed = ?r#gen.seed,
        "build_generation_override success"
    );
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
    tracing::debug!(target: "openai_compat::routes", error = %msg, code = ?e.code(), "map_backend_error");
    if msg.contains("busy") {
        tracing::warn!(target: "openai_compat::routes", error = %msg, "map_backend_error: rate_limit");
        return (StatusCode::TOO_MANY_REQUESTS, error_json(&msg, "rate_limit"));
    }
    if msg.contains("live") {
        tracing::warn!(target: "openai_compat::routes", error = %msg, "map_backend_error: service_unavailable live");
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
