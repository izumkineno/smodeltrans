use serde::Serialize;
use std::{
    collections::VecDeque,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_RECENT_EVENTS: usize = 24;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RuntimeEvent {
    pub(crate) timestamp_ms: u64,
    pub(crate) category: &'static str,
    pub(crate) operation: String,
    pub(crate) duration_ms: u64,
    pub(crate) success: bool,
    pub(crate) message: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RuntimeMetricsSnapshot {
    pub(crate) started_at_epoch_ms: u64,
    pub(crate) uptime_ms: u64,
    pub(crate) idle_for_ms: u64,
    pub(crate) busy: bool,
    pub(crate) ocr_loaded: bool,
    pub(crate) translator_loaded: bool,
    pub(crate) request_count: u64,
    pub(crate) successful_requests: u64,
    pub(crate) failed_requests: u64,
    pub(crate) last_response_ms: Option<u64>,
    pub(crate) average_response_ms: Option<u64>,
    pub(crate) recent_events: Vec<RuntimeEvent>,
}

#[derive(Debug)]
pub(crate) struct RuntimeMetrics {
    started_at: Instant,
    started_at_epoch_ms: u64,
    ocr_loaded: bool,
    translator_loaded: bool,
    request_count: u64,
    successful_requests: u64,
    failed_requests: u64,
    total_response_ms: u128,
    last_response_ms: Option<u64>,
    recent_events: VecDeque<RuntimeEvent>,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            started_at_epoch_ms: epoch_millis(),
            ocr_loaded: false,
            translator_loaded: false,
            request_count: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_response_ms: 0,
            last_response_ms: None,
            recent_events: VecDeque::with_capacity(MAX_RECENT_EVENTS),
        }
    }
}

impl RuntimeMetrics {
    pub(crate) fn set_model_states(&mut self, ocr_loaded: bool, translator_loaded: bool) {
        self.ocr_loaded = ocr_loaded;
        self.translator_loaded = translator_loaded;
    }

    pub(crate) fn model_states(&self) -> (bool, bool) {
        (self.ocr_loaded, self.translator_loaded)
    }

    pub(crate) fn record_request(
        &mut self,
        operation: &str,
        duration: Duration,
        success: bool,
        message: &str,
    ) {
        let duration_ms = duration_millis(duration);
        self.request_count = self.request_count.saturating_add(1);
        if success {
            self.successful_requests = self.successful_requests.saturating_add(1);
        } else {
            self.failed_requests = self.failed_requests.saturating_add(1);
        }
        self.total_response_ms = self
            .total_response_ms
            .saturating_add(u128::from(duration_ms));
        self.last_response_ms = Some(duration_ms);
        self.push_event("request", operation, duration_ms, success, message);
    }

    pub(crate) fn record_control(
        &mut self,
        operation: &str,
        duration: Duration,
        success: bool,
        message: &str,
    ) {
        self.push_event(
            "control",
            operation,
            duration_millis(duration),
            success,
            message,
        );
    }

    pub(crate) fn snapshot(&self, idle_for: Duration, busy: bool) -> RuntimeMetricsSnapshot {
        let average_response_ms = (self.request_count > 0).then(|| {
            u64::try_from(self.total_response_ms / u128::from(self.request_count))
                .unwrap_or(u64::MAX)
        });
        RuntimeMetricsSnapshot {
            started_at_epoch_ms: self.started_at_epoch_ms,
            uptime_ms: duration_millis(self.started_at.elapsed()),
            idle_for_ms: duration_millis(idle_for),
            busy,
            ocr_loaded: self.ocr_loaded,
            translator_loaded: self.translator_loaded,
            request_count: self.request_count,
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            last_response_ms: self.last_response_ms,
            average_response_ms,
            recent_events: self.recent_events.iter().rev().cloned().collect(),
        }
    }

    fn push_event(
        &mut self,
        category: &'static str,
        operation: &str,
        duration_ms: u64,
        success: bool,
        message: &str,
    ) {
        if self.recent_events.len() == MAX_RECENT_EVENTS {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(RuntimeEvent {
            timestamp_ms: epoch_millis(),
            category,
            operation: operation.to_owned(),
            duration_ms,
            success,
            message: message.to_owned(),
        });
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(duration_millis)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_metrics_track_latency_outcomes_and_model_states() {
        let mut metrics = RuntimeMetrics::default();
        metrics.set_model_states(true, false);
        metrics.record_request("OCR 识别", Duration::from_millis(120), true, "处理完成");
        metrics.record_request("文本翻译", Duration::from_millis(80), false, "模型错误");

        let snapshot = metrics.snapshot(Duration::from_secs(3), true);

        assert!(snapshot.busy);
        assert!(snapshot.ocr_loaded);
        assert!(!snapshot.translator_loaded);
        assert_eq!(snapshot.request_count, 2);
        assert_eq!(snapshot.successful_requests, 1);
        assert_eq!(snapshot.failed_requests, 1);
        assert_eq!(snapshot.last_response_ms, Some(80));
        assert_eq!(snapshot.average_response_ms, Some(100));
        assert_eq!(snapshot.idle_for_ms, 3_000);
        assert_eq!(snapshot.recent_events[0].operation, "文本翻译");
    }

    #[test]
    fn recent_event_history_is_bounded() {
        let mut metrics = RuntimeMetrics::default();
        for index in 0..(MAX_RECENT_EVENTS + 3) {
            metrics.record_control(&format!("操作 {index}"), Duration::ZERO, true, "完成");
        }

        let snapshot = metrics.snapshot(Duration::ZERO, false);

        assert_eq!(snapshot.recent_events.len(), MAX_RECENT_EVENTS);
        assert_eq!(snapshot.recent_events[0].operation, "操作 26");
        assert_eq!(snapshot.recent_events[23].operation, "操作 3");
    }
}
