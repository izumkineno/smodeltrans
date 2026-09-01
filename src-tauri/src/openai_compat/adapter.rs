//! 解耦适配层：唯一允许触碰 `BackendState` 的文件。
//! 禁止在 `openai_compat` 其他文件中 `use crate::backend::engine` 或 `crate::models::hy`.

use crate::backend::{commands::BackendState, failure::BackendFailure};
use crate::model_config::GenerationConfig;
use crate::model_support::{CancellationToken, lock_with_cancellation};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// 对外暴露的最小翻译能力。Send + Sync + 'static 以便跨 axum 线程共享。
#[allow(dead_code)]
pub trait TranslationPort: Send + Sync + 'static {
    fn translate_text(
        &self,
        text: String,
        target_language: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure>;

    fn is_ready(&self) -> bool;

    fn model_states(&self) -> Result<(bool, bool), BackendFailure>;

    fn live_active(&self) -> bool;
}

#[derive(Clone)]
pub struct BackendStateAdapter {
    state: BackendState,
    /// 轻量信号量：true 表示有请求正在翻译，用于 429 限流（简化为 AtomicBool 单并发）
    busy: Arc<AtomicBool>,
}

impl BackendStateAdapter {
    pub fn new(state: BackendState) -> Self {
        Self {
            state,
            busy: Arc::new(AtomicBool::new(false)),
        }
    }

    fn try_acquire(&self) -> Result<BusyGuard, BackendFailure> {
        if self
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(BackendFailure::internal(
                "translation engine busy, try again later",
            ));
        }
        Ok(BusyGuard {
            busy: Arc::clone(&self.busy),
        })
    }
}

struct BusyGuard {
    busy: Arc<AtomicBool>,
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::SeqCst);
    }
}

impl TranslationPort for BackendStateAdapter {
    fn translate_text(
        &self,
        text: String,
        target_language: String,
        generation_override: Option<GenerationConfig>,
    ) -> Result<String, BackendFailure> {
        // 入口触活 + 入口 live 检查
        self.state.touch_activity();
        if self.state.live_active.load(Ordering::SeqCst) {
            return Err(BackendFailure::internal(
                "live translation is active, openai compat busy",
            ));
        }

        // 轻量并发守卫：首版单并发，超限返回 busy（调用方映射为 429）
        let _guard = self.try_acquire();

        // 取 settings 快照
        let settings = self
            .state
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;

        // 目标语言校验（复用 input 校验的简化版）
        let target_language = target_language.trim().to_owned();
        if target_language.is_empty() || target_language.len() > 64 {
            return Err(BackendFailure::arguments("target_language 非法"));
        }
        if text.trim().is_empty() {
            return Err(BackendFailure::arguments("text 不能为空"));
        }
        if text.len() > 8 * 1024 * 1024 {
            return Err(BackendFailure::arguments("text 超过 8 MiB 限制"));
        }

        // 构造取消令牌（HTTP 侧暂不支持取消，按永不取消处理）
        let token = CancellationToken::new();

        // 获锁后二次 live 检查，消除 TOCTOU
        let result = {
            let mut engine_guard = lock_with_cancellation(&self.state.engine, &token)?;
            if engine_guard.is_none() {
                *engine_guard = Some(crate::backend::engine::BackendEngine::new(settings.clone())?);
            }
            // 二次检查
            if self.state.live_active.load(Ordering::SeqCst) {
                return Err(BackendFailure::internal(
                    "live translation became active while waiting for engine",
                ));
            }
            // 二次触活
            self.state.touch_activity();

            let engine = engine_guard
                .as_mut()
                .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;

            // 若有 generation 覆盖，临时替换 engine.settings.generation
            let original_generation = engine.settings.generation.clone();
            let has_override = generation_override.is_some();
            if let Some(r#ov) = generation_override {
                engine.settings.generation = r#ov;
            }

            let res = engine.translate_text(
                &text,
                &target_language,
                "",
                &token,
                |_, _| {},
                |_| {},
            );

            // 恢复
            if has_override {
                engine.settings.generation = original_generation;
            }
            res
        };

        self.state.touch_activity();
        result
    }

    fn is_ready(&self) -> bool {
        self.model_states()
            .map(|(ocr, hy)| ocr || hy)
            .unwrap_or(false)
    }

    fn model_states(&self) -> Result<(bool, bool), BackendFailure> {
        self.state.model_states()
    }

    fn live_active(&self) -> bool {
        self.state.live_active.load(Ordering::SeqCst)
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
            Ok(self.return_text.clone())
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn model_states(&self) -> Result<(bool, bool), BackendFailure> {
            Ok((true, true))
        }
        fn live_active(&self) -> bool {
            false
        }
    }
}
