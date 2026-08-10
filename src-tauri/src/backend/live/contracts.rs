use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CaptureWindowInfo {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) process_name: String,
    pub(crate) process_id: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveRoi {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) client_width: u32,
    pub(crate) client_height: u32,
}

impl LiveRoi {
    pub(super) fn validate(self) -> Result<Self, &'static str> {
        if self.client_width == 0 || self.client_height == 0 {
            return Err("ROI 客户区尺寸必须大于零");
        }
        if self.width < 24 || self.height < 24 {
            return Err("ROI 宽度和高度至少需要 24 个物理像素");
        }
        if self.x >= self.client_width || self.y >= self.client_height {
            return Err("ROI 起点超出目标客户区");
        }
        let right = self.x.checked_add(self.width).ok_or("ROI 横向坐标溢出")?;
        let bottom = self.y.checked_add(self.height).ok_or("ROI 纵向坐标溢出")?;
        if right > self.client_width || bottom > self.client_height {
            return Err("ROI 超出目标客户区");
        }
        Ok(self)
    }

    pub(super) fn normalized(self) -> Result<NormalizedRoi, &'static str> {
        NormalizedRoi::from_physical(self.validate()?)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveOverlayMode {
    #[default]
    Subtitle,
    RegionReplace,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveOverlayAttachment {
    Top,
    #[default]
    Bottom,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveOverlaySettings {
    pub(crate) mode: LiveOverlayMode,
    pub(crate) attachment: LiveOverlayAttachment,
    pub(crate) offset: u32,
    pub(crate) show_source: bool,
}

impl Default for LiveOverlaySettings {
    fn default() -> Self {
        Self {
            mode: LiveOverlayMode::Subtitle,
            attachment: LiveOverlayAttachment::Bottom,
            offset: 0,
            show_source: true,
        }
    }
}

impl LiveOverlaySettings {
    pub(super) fn validate(self) -> Result<Self, &'static str> {
        if self.offset > 2_048 {
            return Err("实时翻译框外侧偏移必须在 0 到 2048 像素之间");
        }
        Ok(self)
    }

    pub(super) fn mode_query_value(self) -> &'static str {
        match self.mode {
            LiveOverlayMode::Subtitle => "subtitle",
            LiveOverlayMode::RegionReplace => "region_replace",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveMetrics {
    pub(crate) frames_captured: u64,
    pub(crate) frames_dropped: u64,
    pub(crate) ocr_runs: u64,
    pub(crate) translation_runs: u64,
    pub(crate) cache_hits: u64,
    pub(crate) subtitle_publishes: u64,
    pub(crate) last_ocr_ms: u64,
    pub(crate) last_translation_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveRecognitionMode {
    #[default]
    Automatic,
    KeyTrigger,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveRecognitionTrigger {
    #[default]
    Press,
    Release,
}

pub(super) const DEFAULT_STABILITY_WAIT_MS: u64 = 300;
pub(super) const MAX_STABILITY_WAIT_MS: u64 = 5_000;

fn default_stability_wait_ms() -> u64 {
    DEFAULT_STABILITY_WAIT_MS
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveRecognitionSettings {
    pub(crate) mode: LiveRecognitionMode,
    pub(crate) trigger_key: String,
    pub(crate) trigger_event: LiveRecognitionTrigger,
    #[serde(default = "default_stability_wait_ms")]
    pub(crate) stability_wait_ms: u64,
}

impl Default for LiveRecognitionSettings {
    fn default() -> Self {
        Self {
            mode: LiveRecognitionMode::Automatic,
            trigger_key: "F8".to_owned(),
            trigger_event: LiveRecognitionTrigger::Press,
            stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
        }
    }
}

fn is_supported_trigger_key(value: &str) -> bool {
    let key = value.trim();
    if key.is_empty() || key.len() > 64 || key.chars().any(|character| character.is_control()) {
        return false;
    }
    let Some(payload) = key.strip_prefix("vk:") else {
        return is_legacy_trigger_key(key);
    };
    let mut parts = payload.splitn(2, '|');
    let virtual_key = parts
        .next()
        .and_then(|number| number.parse::<u16>().ok())
        .filter(|number| (1..=0xff).contains(number));
    let label = parts.next();
    virtual_key.is_some()
        && label
            .map(|value| !value.is_empty() && value.len() <= 32)
            .unwrap_or(true)
}

fn is_legacy_trigger_key(key: &str) -> bool {
    if let Some(letter) = key.strip_prefix("Key") {
        let bytes = letter.as_bytes();
        return bytes.len() == 1 && bytes[0].is_ascii_uppercase();
    }
    if let Some(number) = key.strip_prefix("Digit") {
        let bytes = number.as_bytes();
        return bytes.len() == 1 && bytes[0].is_ascii_digit();
    }
    if let Some(number) = key.strip_prefix("Numpad") {
        let bytes = number.as_bytes();
        if bytes.len() == 1 && bytes[0].is_ascii_digit() {
            return true;
        }
    }
    if let Some(number) = key
        .strip_prefix('F')
        .and_then(|number| number.parse::<u16>().ok())
    {
        return (1..=24).contains(&number);
    }
    matches!(
        key,
        "Escape"
            | "CapsLock"
            | "Shift"
            | "ShiftLeft"
            | "ShiftRight"
            | "Control"
            | "ControlLeft"
            | "ControlRight"
            | "AltLeft"
            | "AltRight"
            | "MetaLeft"
            | "MetaRight"
            | "ContextMenu"
            | "Space"
            | "Enter"
            | "Tab"
            | "Backspace"
            | "Insert"
            | "Delete"
            | "Home"
            | "End"
            | "PageUp"
            | "PageDown"
            | "ArrowUp"
            | "ArrowDown"
            | "ArrowLeft"
            | "ArrowRight"
            | "PrintScreen"
            | "ScrollLock"
            | "Pause"
            | "NumLock"
            | "Minus"
            | "Equal"
            | "BracketLeft"
            | "BracketRight"
            | "Backslash"
            | "IntlBackslash"
            | "IntlYen"
            | "Semicolon"
            | "Quote"
            | "Backquote"
            | "Comma"
            | "Period"
            | "Slash"
            | "IntlRo"
            | "NumpadDecimal"
            | "NumpadDivide"
            | "NumpadMultiply"
            | "NumpadSubtract"
            | "NumpadAdd"
            | "NumpadEnter"
            | "NumpadEqual"
            | "Ctrl"
            | "Alt"
    )
}

impl LiveRecognitionSettings {
    pub(super) fn validate(mut self) -> Result<Self, &'static str> {
        let trigger_key = self.trigger_key.trim();
        let legacy_key = [
            "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "Space",
            "Enter", "Tab", "Shift", "Ctrl", "Alt",
        ]
        .into_iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(trigger_key));
        let canonical_key = legacy_key.unwrap_or(trigger_key);
        if self.stability_wait_ms > MAX_STABILITY_WAIT_MS {
            return Err("OCR 字幕稳定等待必须在 0 到 5000 毫秒之间");
        }

        if !is_supported_trigger_key(canonical_key) {
            return Err("实时翻译触发按键不受支持，请重新录入标准键盘按键");
        }
        self.trigger_key = canonical_key.to_owned();
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum LiveSessionState {
    #[default]
    Idle,
    Selecting,
    Warming,
    Running,
    Paused,
    Stopping,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveSessionStatus {
    pub(crate) state: LiveSessionState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) target: Option<CaptureWindowInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) roi: Option<LiveRoi>,
    pub(crate) message: String,
    pub(crate) latest_revision: u64,
    pub(crate) metrics: LiveMetrics,
}

impl Default for LiveSessionStatus {
    fn default() -> Self {
        Self {
            state: LiveSessionState::Idle,
            session_id: None,
            target: None,
            roi: None,
            message: "实时翻译尚未启动。".to_owned(),
            latest_revision: 0,
            metrics: LiveMetrics::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveSubtitleRegion {
    pub(crate) quad: [[i32; 2]; 4],
    pub(crate) source_text: String,
    pub(crate) translated_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveSubtitle {
    pub(crate) session_id: String,
    pub(crate) revision: u64,
    pub(crate) source_text: String,
    pub(crate) translated_text: String,
    pub(crate) roi: LiveRoi,
    pub(crate) regions: Vec<LiveSubtitleRegion>,
    pub(crate) observed_at_epoch_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveDebugStage {
    Ocr,
    Translation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LiveDebugOutcome {
    AwaitingConfirmation,
    Confirmed,
    CacheHit,
    Completed,
    SkippedEmptySource,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LiveDebugRecord {
    pub(crate) session_id: String,
    pub(crate) sequence: u64,
    pub(crate) stage: LiveDebugStage,
    pub(crate) outcome: LiveDebugOutcome,
    pub(crate) source_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) translated_text: Option<String>,
    pub(crate) target_language: String,
    pub(crate) region_count: u32,
    pub(crate) roi_version: u64,
    pub(crate) duration_ms: u64,
    pub(crate) cache_hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) message: Option<String>,
    pub(crate) observed_at_epoch_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct NormalizedRoi {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

impl NormalizedRoi {
    fn from_physical(roi: LiveRoi) -> Result<Self, &'static str> {
        let client_width = f64::from(roi.client_width);
        let client_height = f64::from(roi.client_height);
        let normalized = Self {
            left: f64::from(roi.x) / client_width,
            top: f64::from(roi.y) / client_height,
            right: f64::from(roi.x + roi.width) / client_width,
            bottom: f64::from(roi.y + roi.height) / client_height,
        };
        if !normalized.left.is_finite()
            || !normalized.top.is_finite()
            || !normalized.right.is_finite()
            || !normalized.bottom.is_finite()
            || normalized.left < 0.0
            || normalized.top < 0.0
            || normalized.right > 1.0
            || normalized.bottom > 1.0
            || normalized.left >= normalized.right
            || normalized.top >= normalized.bottom
        {
            return Err("ROI 归一化结果无效");
        }
        Ok(normalized)
    }

    pub(super) fn to_physical(self, client_width: u32, client_height: u32) -> Option<LiveRoi> {
        if client_width == 0 || client_height == 0 {
            return None;
        }
        let width = f64::from(client_width);
        let height = f64::from(client_height);
        let x = (self.left * width).round().clamp(0.0, width) as u32;
        let y = (self.top * height).round().clamp(0.0, height) as u32;
        let right = (self.right * width).round().clamp(f64::from(x + 1), width) as u32;
        let bottom = (self.bottom * height)
            .round()
            .clamp(f64::from(y + 1), height) as u32;
        Some(LiveRoi {
            x,
            y,
            width: right.saturating_sub(x).max(1),
            height: bottom.saturating_sub(y).max(1),
            client_width,
            client_height,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct LiveConfig {
    pub(super) roi: NormalizedRoi,
    pub(super) roi_version: u64,
    pub(super) target_language: String,
    pub(super) overlay_settings: LiveOverlaySettings,
    pub(super) recognition_settings: LiveRecognitionSettings,
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_STABILITY_WAIT_MS, LiveRecognitionMode, LiveRecognitionSettings,
        LiveRecognitionTrigger, LiveRoi, LiveSessionState, LiveSessionStatus,
        MAX_STABILITY_WAIT_MS,
    };

    #[test]
    fn status_serde_contract_is_camel_case() {
        let value = serde_json::to_value(LiveSessionStatus::default()).expect("serialize status");
        assert_eq!(value["state"], "idle");
        assert_eq!(value["latestRevision"], 0);
        assert_eq!(value["metrics"]["framesCaptured"], 0);
        assert!(value.get("sessionId").is_none());
    }

    #[test]
    fn roi_round_trip_preserves_scaled_region() {
        let roi = LiveRoi {
            x: 100,
            y: 600,
            width: 800,
            height: 240,
            client_width: 1920,
            client_height: 1080,
        };
        let scaled = roi
            .normalized()
            .expect("normalize")
            .to_physical(2560, 1440)
            .expect("scale");
        assert_eq!(
            scaled,
            LiveRoi {
                x: 133,
                y: 800,
                width: 1067,
                height: 320,
                client_width: 2560,
                client_height: 1440,
            }
        );
        assert_eq!(LiveSessionState::default(), LiveSessionState::Idle);
    }

    #[test]
    fn roi_rejects_out_of_bounds_values() {
        let invalid = LiveRoi {
            x: 1900,
            y: 1000,
            width: 100,
            height: 100,
            client_width: 1920,
            client_height: 1080,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn recognition_settings_validate_modes_keys_and_trigger_events() {
        let automatic = LiveRecognitionSettings::default()
            .validate()
            .expect("default");
        assert_eq!(automatic.mode, LiveRecognitionMode::Automatic);
        assert_eq!(automatic.trigger_event, LiveRecognitionTrigger::Press);
        assert_eq!(automatic.stability_wait_ms, DEFAULT_STABILITY_WAIT_MS);

        let key_trigger = LiveRecognitionSettings {
            mode: LiveRecognitionMode::KeyTrigger,
            trigger_key: " space ".to_owned(),
            trigger_event: LiveRecognitionTrigger::Release,
            stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
        }
        .validate()
        .expect("key trigger");
        assert_eq!(key_trigger.trigger_key, "Space");
        assert_eq!(key_trigger.trigger_event, LiveRecognitionTrigger::Release);
        let serialized = serde_json::to_value(key_trigger).expect("serialize settings");
        assert_eq!(serialized["mode"], "key_trigger");
        assert_eq!(serialized["triggerEvent"], "release");
        assert_eq!(serialized["stabilityWaitMs"], DEFAULT_STABILITY_WAIT_MS);
    }

    #[test]
    fn recognition_settings_reject_excessive_stability_wait() {
        let settings = LiveRecognitionSettings {
            mode: LiveRecognitionMode::Automatic,
            trigger_key: "F8".to_owned(),
            trigger_event: LiveRecognitionTrigger::Press,
            stability_wait_ms: MAX_STABILITY_WAIT_MS + 1,
        };
        assert!(settings.validate().is_err());
    }

    #[test]
    fn recognition_settings_accept_recorded_virtual_keys() {
        let key_code = LiveRecognitionSettings {
            mode: LiveRecognitionMode::KeyTrigger,
            trigger_key: "vk:65|KeyA".to_owned(),
            trigger_event: LiveRecognitionTrigger::Press,
            stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
        }
        .validate()
        .expect("recorded virtual key");
        assert_eq!(key_code.trigger_key, "vk:65|KeyA");

        let legacy_key = LiveRecognitionSettings {
            mode: LiveRecognitionMode::KeyTrigger,
            trigger_key: "KeyA".to_owned(),
            trigger_event: LiveRecognitionTrigger::Press,
            stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
        }
        .validate()
        .expect("legacy browser key code");
        assert_eq!(legacy_key.trigger_key, "KeyA");

        let virtual_key = LiveRecognitionSettings {
            mode: LiveRecognitionMode::KeyTrigger,
            trigger_key: "vk:121".to_owned(),
            trigger_event: LiveRecognitionTrigger::Press,
            stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
        }
        .validate()
        .expect("virtual key without display label");
        assert_eq!(virtual_key.trigger_key, "vk:121");
    }

    #[test]
    fn recognition_settings_accept_legacy_modifier_names() {
        for trigger_key in ["Shift", "Ctrl", "Alt"] {
            let settings = LiveRecognitionSettings {
                mode: LiveRecognitionMode::KeyTrigger,
                trigger_key: trigger_key.to_owned(),
                trigger_event: LiveRecognitionTrigger::Press,
                stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
            };
            assert!(
                settings.validate().is_ok(),
                "legacy modifier should remain supported: {trigger_key}"
            );
        }
    }
    #[test]
    fn recognition_settings_reject_unknown_or_out_of_range_keys() {
        for trigger_key in ["MouseButton1", "vk:0", "vk:256"] {
            let settings = LiveRecognitionSettings {
                mode: LiveRecognitionMode::KeyTrigger,
                trigger_key: trigger_key.to_owned(),
                trigger_event: LiveRecognitionTrigger::Press,
                stability_wait_ms: DEFAULT_STABILITY_WAIT_MS,
            };
            assert!(
                settings.validate().is_err(),
                "unsupported trigger key should be rejected: {trigger_key}"
            );
        }
    }
}
