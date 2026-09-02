use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_MAX_LEN: usize = 100;

/// 单条远程翻译历史，复用后端已验证的翻译结果，不触碰模型
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiHistoryEntry {
    pub id: String,
    pub timestamp_ms: u64,
    pub model: String,
    pub source_text: String,
    pub translated_text: String,
    pub target_language: String,
    pub duration_ms: u64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub streaming: bool,
}

impl OpenAiHistoryEntry {
    pub fn new(
        model: String,
        source_text: String,
        translated_text: String,
        target_language: String,
        duration_ms: u64,
        prompt_tokens: usize,
        completion_tokens: usize,
        streaming: bool,
    ) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let id = format!("oai-hist-{}-{}", now_ms, rand_id());
        Self {
            id,
            timestamp_ms: now_ms,
            model,
            source_text,
            translated_text,
            target_language,
            duration_ms,
            prompt_tokens,
            completion_tokens,
            streaming,
        }
    }
}

fn rand_id() -> String {
    // 轻量随机，避免额外依赖；在单进程内用时间 + 原子计数已足够唯一
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("{:04x}", c & 0xffff)
}

/// 线程安全的环形历史存储，供 Axum 与 Tauri 共享
#[derive(Clone, Debug)]
pub struct OpenAiHistoryStore {
    inner: Arc<Mutex<VecDeque<OpenAiHistoryEntry>>>,
    max_len: usize,
}

impl Default for OpenAiHistoryStore {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_LEN)
    }
}

impl OpenAiHistoryStore {
    pub fn new(max_len: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(max_len))),
            max_len: max_len.max(1),
        }
    }

    pub fn push(&self, entry: OpenAiHistoryEntry) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.len() >= self.max_len {
            guard.pop_front();
        }
        guard.push_back(entry);
        tracing::debug!(
            target: "openai_compat::history",
            len = guard.len(),
            max_len = self.max_len,
            "history push"
        );
    }

    pub fn list(&self) -> Vec<OpenAiHistoryEntry> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.iter().cloned().collect()
    }

    pub fn clear(&self) {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.clear();
        tracing::info!(target: "openai_compat::history", "history cleared");
    }

    pub fn len(&self) -> usize {
        match self.inner.lock() {
            Ok(g) => g.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_list() {
        let store = OpenAiHistoryStore::new(3);
        store.push(OpenAiHistoryEntry::new(
            "hy-mt2".into(),
            "hello".into(),
            "你好".into(),
            "Chinese".into(),
            10,
            5,
            5,
            false,
        ));
        assert_eq!(store.len(), 1);
        let list = store.list();
        assert_eq!(list[0].source_text, "hello");
    }

    #[test]
    fn truncate_works() {
        let store = OpenAiHistoryStore::new(2);
        for i in 0..3 {
            store.push(OpenAiHistoryEntry::new(
                "m".into(),
                format!("s{i}"),
                format!("t{i}"),
                "Chinese".into(),
                1,
                1,
                1,
                false,
            ));
        }
        assert_eq!(store.len(), 2);
        let list = store.list();
        assert_eq!(list[0].source_text, "s1");
        assert_eq!(list[1].source_text, "s2");
    }

    #[test]
    fn clear_works() {
        let store = OpenAiHistoryStore::default();
        store.push(OpenAiHistoryEntry::new(
            "m".into(),
            "a".into(),
            "b".into(),
            "Chinese".into(),
            1,
            1,
            1,
            false,
        ));
        store.clear();
        assert!(store.is_empty());
    }
}
