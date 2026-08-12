use crate::backend::{
    failure::BackendFailure,
    live::{
        contracts::{
            CaptureWindowInfo, LiveConfig, LiveMetrics, LiveRecognitionMode,
            LiveRecognitionSettings,
        },
        scheduler::LatestFrameSlot,
    },
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend::live) struct TargetGeometry {
    pub(in crate::backend::live) x: i32,
    pub(in crate::backend::live) y: i32,
    pub(in crate::backend::live) width: u32,
    pub(in crate::backend::live) height: u32,
}

pub(in crate::backend::live) struct CaptureWorker;

impl CaptureWorker {
    pub(in crate::backend::live) fn stop(self) -> Result<(), BackendFailure> {
        Ok(())
    }
}

fn unsupported() -> BackendFailure {
    BackendFailure::arguments("实时窗口翻译目前仅支持 Windows")
}

pub(in crate::backend::live) fn list_target_windows()
-> Result<Vec<CaptureWindowInfo>, BackendFailure> {
    Err(unsupported())
}

pub(in crate::backend::live) fn target_geometry(
    _target_id: &str,
) -> Result<TargetGeometry, BackendFailure> {
    Err(unsupported())
}

pub(in crate::backend::live) fn activate_target_window(
    _target_id: &str,
) -> Result<(), BackendFailure> {
    Err(unsupported())
}

pub(in crate::backend::live) fn target_is_minimized(
    _target_id: &str,
) -> Result<bool, BackendFailure> {
    Err(unsupported())
}

pub(in crate::backend::live) fn recognition_is_active(settings: &LiveRecognitionSettings) -> bool {
    settings.mode == LiveRecognitionMode::Automatic
}

pub(in crate::backend::live) fn start_capture(
    _target_id: &str,
    _latest: Arc<LatestFrameSlot>,
    _terminal_error: Arc<Mutex<Option<String>>>,
    _config: Arc<Mutex<LiveConfig>>,
    _metrics: Arc<Mutex<LiveMetrics>>,
    _paused: Arc<AtomicBool>,
) -> Result<CaptureWorker, BackendFailure> {
    Err(unsupported())
}
