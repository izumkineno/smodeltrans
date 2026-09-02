//! 解耦适配层：唯一允许触碰 `BackendState` 的文件。
//! 禁止在 `openai_compat` 其他文件中 `use crate::backend::engine` 或 `crate::models::hy`.

use crate::backend::{commands::BackendState, failure::BackendFailure};
use crate::model_config::GenerationConfig;
use crate::model_support::{CancellationToken, lock_with_cancellation};
use std::{
    sync::{
        Arc, Mutex,
        atomic::Ordering,
    },
    time::Instant,
};

/// 对外暴露的最小翻译能力。Send + Sync + 'static 以便跨 axum 线程共享。
/// 复用 Hy-MT2 官方模板：后端 `build_translation_prompt` 为主，`supplemental_prompt` 仅作附加约束（对应 `render_single_prompt` 的 Additional requirements）。
#[allow(dead_code)]
pub trait TranslationPort: Send + Sync + 'static {
    fn translate_text(
        &self,
        text: String,
        target_language: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure>;

    fn translate_text_with_supplemental(
        &self,
        text: String,
        target_language: String,
        _supplemental_prompt: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure> {
        // 默认回落：忽略 supplemental，保持兼容
        self.translate_text(text, target_language, generation_override)
    }
    fn is_ready(&self) -> bool;

    fn model_states(&self) -> Result<(bool, bool), BackendFailure>;

    fn live_active(&self) -> bool;
}

#[derive(Clone)]
pub struct BackendStateAdapter {
    state: BackendState,
    /// 串行队列：`Mutex` 阻塞式 FIFO 排队（`spawn_blocking` 可等待），替代 `AtomicBool -> 429`；
    /// 持有期间独占引擎，合并翻译仍经 Hy-MT2 `render_single_prompt` + `supplemental` 单模板，
    /// 后续可在持锁后 20ms 窗口内聚合同 `target_language` 的等待请求走 `translate_structured_batch`。
    queue: Arc<Mutex<()>>,
}

impl BackendStateAdapter {
    pub fn new(state: BackendState) -> Self {
        tracing::debug!(target: "openai_compat::adapter", "BackendStateAdapter::new queue init");
        Self {
            state,
            queue: Arc::new(Mutex::new(())),
        }
    }
}

impl TranslationPort for BackendStateAdapter {
    fn translate_text(
        &self,
        text: String,
        target_language: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure> {
        self.translate_text_with_supplemental(text, target_language, String::new(), generation_override)
    }

    fn translate_text_with_supplemental(
        &self,
        text: String,
        target_language: String,
        supplemental_prompt: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure> {
        let start = Instant::now();
        let text_len = text.len();
        let text_chars = text.chars().count();
        let supplemental_len = supplemental_prompt.chars().count();
        let has_override = generation_override.is_some();
        // keep original for logging after trim
        let target_raw = target_language.clone();
        tracing::debug!(
            target: "openai_compat::adapter",
            target_language = %target_raw,
            text_len = text_len,
            text_chars = text_chars,
            supplemental_len = supplemental_len,
            has_generation_override = has_override,
            "translate_text called"
        );
        // 入口触活 + 入口 live 检查
        self.state.touch_activity();
        if self.state.live_active.load(Ordering::SeqCst) {
            tracing::warn!(
                target: "openai_compat::adapter",
                target_language = %target_raw,
                text_len = text_len,
                duration_ms = start.elapsed().as_millis() as u64,
                "translate_text rejected: live active at entry"
            );
            return Err(BackendFailure::internal(
                "live translation is active, openai compat busy",
            ));
        }

        // 队列化：`live` 优先，引擎忙时排队而非 429；`spawn_blocking` 线程可阻塞等待
        // Hy-MT2 提示词仍经后端 `render_single_prompt(prompt, supplemental)` 单模板渲染，排队不丢约束
        // 后续可在持锁后加 20ms 窗口聚合同 target 的等待请求走 `translate_structured_batch`
        let _queue_guard = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        tracing::debug!(
            target: "openai_compat::adapter",
            target_language = %target_raw,
            "queue acquired"
        );
        if self.state.live_active.load(Ordering::SeqCst) {
            tracing::warn!(
                target: "openai_compat::adapter",
                target_language = %target_raw,
                text_len = text_len,
                duration_ms = start.elapsed().as_millis() as u64,
                "translate_text rejected: live became active while queued"
            );
            return Err(BackendFailure::internal(
                "live translation became active while queued",
            ));
        }

        // 取 settings 快照
        let settings = match self
            .state
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))
            .and_then(|guard| guard.clone().map_err(BackendFailure::arguments))
        {
            Ok(s) => {
                tracing::debug!(
                    target: "openai_compat::adapter",
                    target_language = %target_raw,
                    "translate_text settings snapshot acquired"
                );
                s
            }
            Err(e) => {
                tracing::error!(
                    target: "openai_compat::adapter",
                    target_language = %target_raw,
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "translate_text failed: settings lock error"
                );
                return Err(e);
            }
        };

        // 目标语言校验（复用 input 校验的简化版）
        let target_language = target_language.trim().to_owned();
        if target_language.is_empty() || target_language.len() > 64 {
            tracing::warn!(
                target: "openai_compat::adapter",
                target_language = %target_language,
                original = %target_raw,
                duration_ms = start.elapsed().as_millis() as u64,
                "translate_text rejected: invalid target_language"
            );
            return Err(BackendFailure::arguments("target_language 非法"));
        }
        if text.trim().is_empty() {
            tracing::warn!(
                target: "openai_compat::adapter",
                target_language = %target_language,
                text_len = text_len,
                duration_ms = start.elapsed().as_millis() as u64,
                "translate_text rejected: empty text"
            );
            return Err(BackendFailure::arguments("text 不能为空"));
        }
        if text.len() > 8 * 1024 * 1024 {
            tracing::warn!(
                target: "openai_compat::adapter",
                target_language = %target_language,
                text_len = text_len,
                duration_ms = start.elapsed().as_millis() as u64,
                "translate_text rejected: text over 8MiB"
            );
            return Err(BackendFailure::arguments("text 超过 8 MiB 限制"));
        }

        // 构造取消令牌（HTTP 侧暂不支持取消，按永不取消处理）
        let token = CancellationToken::new();

        // 获锁后二次 live 检查，消除 TOCTOU
        let result = {
            let mut engine_guard = match lock_with_cancellation(&self.state.engine, &token) {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!(
                        target: "openai_compat::adapter",
                        target_language = %target_language,
                        error = %e,
                        duration_ms = start.elapsed().as_millis() as u64,
                        "translate_text failed: engine lock error"
                    );
                    return Err(e);
                }
            };
            if engine_guard.is_none() {
                tracing::info!(
                    target: "openai_compat::adapter",
                    target_language = %target_language,
                    "translate_text engine uninitialized, creating BackendEngine"
                );
                let engine_init_start = Instant::now();
                match crate::backend::engine::BackendEngine::new(settings.clone()) {
                    Ok(engine) => {
                        tracing::info!(
                            target: "openai_compat::adapter",
                            target_language = %target_language,
                            duration_ms = engine_init_start.elapsed().as_millis() as u64,
                            "BackendEngine created"
                        );
                        *engine_guard = Some(engine);
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "openai_compat::adapter",
                            target_language = %target_language,
                            error = %e,
                            duration_ms = engine_init_start.elapsed().as_millis() as u64,
                            "BackendEngine creation failed"
                        );
                        return Err(e);
                    }
                }
            }
            // 二次检查
            if self.state.live_active.load(Ordering::SeqCst) {
                tracing::warn!(
                    target: "openai_compat::adapter",
                    target_language = %target_language,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "translate_text rejected: live became active while waiting for engine"
                );
                return Err(BackendFailure::internal(
                    "live translation became active while waiting for engine",
                ));
            }
            // 二次触活
            self.state.touch_activity();

            let engine = engine_guard
                .as_mut()
                .ok_or_else(|| {
                    tracing::error!(target: "openai_compat::adapter", target_language = %target_language, "translate_text failed: Candle backend not initialized");
                    BackendFailure::internal("Candle 后端未初始化")
                })?;

            // 若有 generation 覆盖，临时替换 engine.settings.generation
            let original_generation = engine.settings.generation.clone();
            let has_override = generation_override.is_some();
            if let Some(r#ov) = generation_override.clone() {
                tracing::debug!(
                    target: "openai_compat::adapter",
                    target_language = %target_language,
                    temperature = r#ov.temperature,
                    top_p = r#ov.top_p,
                    top_k = r#ov.top_k,
                    max_new_tokens = r#ov.max_new_tokens,
                    "applying generation override"
                );
                engine.settings.generation = r#ov;
            }

            let translate_start = Instant::now();
            tracing::debug!(
                target: "openai_compat::adapter",
                target_language = %target_language,
                text_len = text_len,
                supplemental_len = supplemental_len,
                has_override = has_override,
                "engine.translate_text started"
            );
            let res = engine.translate_text(
                &text,
                &target_language,
                &supplemental_prompt,
                &token,
                |_, _| {},
                |_| {},
            );

            // 恢复
            if has_override {
                engine.settings.generation = original_generation;
                tracing::trace!(target: "openai_compat::adapter", target_language = %target_language, "generation override restored");
            }
            match &res {
                Ok(translated) => {
                    tracing::debug!(
                        target: "openai_compat::adapter",
                        target_language = %target_language,
                        output_len = translated.len(),
                        output_chars = translated.chars().count(),
                        engine_duration_ms = translate_start.elapsed().as_millis() as u64,
                        total_duration_ms = start.elapsed().as_millis() as u64,
                        "engine.translate_text succeeded"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "openai_compat::adapter",
                        target_language = %target_language,
                        error = %e,
                        engine_duration_ms = translate_start.elapsed().as_millis() as u64,
                        total_duration_ms = start.elapsed().as_millis() as u64,
                        "engine.translate_text failed"
                    );
                }
            }
            res
        };

        self.state.touch_activity();
        match &result {
            Ok(translated) => {
                tracing::info!(
                    target: "openai_compat::adapter",
                    target_language = %target_language,
                    text_len = text_len,
                    text_chars = text_chars,
                    output_len = translated.len(),
                    output_chars = translated.chars().count(),
                    has_generation_override = has_override,
                    supplemental_len = supplemental_len,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "translate_text success"
                );
            }
            Err(e) => {
                tracing::error!(
                    target: "openai_compat::adapter",
                    target_language = %target_language,
                    text_len = text_len,
                    has_generation_override = has_override,
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "translate_text failed"
                );
            }
        }
        result
    }

    fn is_ready(&self) -> bool {
        let ready = self.model_states()
            .map(|(ocr, hy)| ocr || hy)
            .unwrap_or(false);
        tracing::debug!(target: "openai_compat::adapter", ready = ready, "is_ready checked");
        ready
    }

    fn model_states(&self) -> Result<(bool, bool), BackendFailure> {
        let start = Instant::now();
        let res = self.state.model_states();
        match &res {
            Ok((ocr, hy)) => {
                tracing::debug!(
                    target: "openai_compat::adapter",
                    ocr_loaded = *ocr,
                    hy_loaded = *hy,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "model_states success"
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "openai_compat::adapter",
                    error = %e,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "model_states failed"
                );
            }
        }
        res
    }

    fn live_active(&self) -> bool {
        let active = self.state.live_active.load(Ordering::SeqCst);
        tracing::trace!(target: "openai_compat::adapter", live_active = active, "live_active checked");
        active
    }
}

// 供测试的 Mock 实现（不触碰 BackendState）
#[cfg(test)]
pub mod mock {
    use super::*;
    use crate::backend::failure::BackendFailure;

    pub struct MockPort {
        pub return_text: String,
    }

    impl TranslationPort for MockPort {
        fn translate_text(
            &self,
            _text: String,
            _lang: String,
            _gen: Option<GenerationConfig>,
        ) -> Result<String, BackendFailure> {
            tracing::debug!(target: "openai_compat::adapter", mock_text_len = _text.len(), mock_lang = %_lang, "MockPort translate_text called");
            Ok(self.return_text.clone())
        }
        fn is_ready(&self) -> bool {
            tracing::trace!(target: "openai_compat::adapter", "MockPort is_ready");
            true
        }
        fn model_states(&self) -> Result<(bool, bool), BackendFailure> {
            tracing::trace!(target: "openai_compat::adapter", "MockPort model_states");
            Ok((true, true))
        }
        fn live_active(&self) -> bool {
            tracing::trace!(target: "openai_compat::adapter", "MockPort live_active");
            false
        }
    }
}
