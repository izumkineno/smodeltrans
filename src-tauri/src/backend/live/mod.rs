pub(crate) mod contracts;
mod platform;
mod scheduler;

use self::{
    contracts::{
        CaptureWindowInfo, LiveConfig, LiveDebugOutcome, LiveDebugRecord, LiveDebugStage,
        LiveMetrics, LiveOverlayAttachment, LiveOverlayMode, LiveOverlaySettings,
        LiveRecognitionMode, LiveRecognitionSettings, LiveRecognitionTrigger, LiveRoi,
        LiveSessionState, LiveSessionStatus, LiveSubtitle, LiveSubtitleRegion,
        LiveSubtitleRegionBounds, LiveTranslationSettings, NormalizedRoi,
    },
    scheduler::{
        LatestFrameSlot, OwnedFrame, StabilityScheduler, finalize_live_regions,
        live_translated_region_text, normalized_live_region_text, plan_live_ocr_groups,
        roi_result_is_current,
    },
};
use super::{
    commands::{BackendError, BackendState},
    contracts::RegionRecord,
    engine::BackendEngine,
    failure::{BackendFailure, BackendFailureCode},
    input::{DecodedImage, validate_target_language},
};
use crate::{model_config::MemoryConfig, model_support::CancellationToken};
use image::RgbImage;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{
    Emitter, Manager, PhysicalPosition, PhysicalSize, State, WebviewUrl, WebviewWindowBuilder,
};

const STATUS_EVENT: &str = "live-status";
const SUBTITLE_EVENT: &str = "live-subtitle";
const DEBUG_RECORD_EVENT: &str = "live-debug-record";
const SELECTOR_LABEL: &str = "live-selector";
const OVERLAY_LABEL: &str = "live-overlay";
const OVERLAY_EDGE_THICKNESS: u32 = 168;
const DEBUG_TEXT_MAX_CHARS: usize = 2_000;
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

fn trace_live(message: std::fmt::Arguments<'_>) {
    if cfg!(debug_assertions) || std::env::var_os("SMODELTRANS_TRACE_LIVE").is_some() {
        eprintln!("[live] {message}");
    }
}

fn debug_text(value: &str) -> String {
    match value.char_indices().nth(DEBUG_TEXT_MAX_CHARS) {
        Some((index, _)) => format!("{}…", &value[..index]),
        None => value.to_owned(),
    }
}
fn key_trigger_timeout_reached(
    trigger_pending: bool,
    wait_elapsed: Option<Duration>,
    has_frame: bool,
    timeout: Duration,
) -> bool {
    trigger_pending && has_frame && wait_elapsed.is_some_and(|elapsed| elapsed >= timeout)
}

#[derive(Clone)]
pub(crate) struct LiveSessionManager {
    backend: BackendState,
    status: Arc<Mutex<LiveSessionStatus>>,
    inner: Arc<Mutex<ManagerInner>>,
}

#[derive(Default)]
struct ManagerInner {
    session_id: Option<String>,
    target_language: Option<String>,
    overlay_settings: Option<LiveOverlaySettings>,
    recognition_settings: Option<LiveRecognitionSettings>,
    translation_settings: Option<LiveTranslationSettings>,
    selection_previous_state: Option<LiveSessionState>,
    runtime: Option<SessionRuntime>,
}

struct SessionRuntime {
    id: String,
    config: Arc<Mutex<LiveConfig>>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    cancellation: CancellationToken,
    auto_paused: Arc<AtomicBool>,
    translation_cancellation: Arc<Mutex<Option<CancellationToken>>>,
    latest: Arc<LatestFrameSlot>,
    join: Option<JoinHandle<()>>,
}

impl LiveSessionManager {
    pub(crate) fn new(backend: BackendState) -> Self {
        Self {
            backend,
            status: Arc::new(Mutex::new(LiveSessionStatus::default())),
            inner: Arc::new(Mutex::new(ManagerInner::default())),
        }
    }

    fn current_status(&self) -> Result<LiveSessionStatus, BackendFailure> {
        self.status
            .lock()
            .map(|status| status.clone())
            .map_err(|_| BackendFailure::internal("实时会话状态锁已损坏"))
    }

    fn collect_finished(&self) -> Result<(), BackendFailure> {
        let finished = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.join.as_ref())
            .is_some_and(JoinHandle::is_finished);
        if !finished {
            return Ok(());
        }
        let runtime = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?
            .runtime
            .take();
        if let Some(mut runtime) = runtime {
            if let Some(join) = runtime.join.take() {
                let _ = join.join();
            }
        }
        Ok(())
    }

    fn begin_selection(
        &self,
        app: &tauri::AppHandle,
        target_id: String,
        target_language: String,
        overlay_settings: LiveOverlaySettings,
        recognition_settings: LiveRecognitionSettings,
        translation_settings: LiveTranslationSettings,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        self.collect_finished()?;
        let target_language = validate_target_language(&target_language)?;
        let overlay_settings = overlay_settings
            .validate()
            .map_err(BackendFailure::arguments)?;
        let recognition_settings = recognition_settings
            .validate()
            .map_err(BackendFailure::arguments)?;
        let translation_settings = translation_settings
            .validate()
            .map_err(BackendFailure::arguments)?;
        let mut target = target_by_id(&target_id)?;
        platform::activate_target_window(&target_id)?;
        let geometry = platform::target_geometry(&target_id)?;
        target.width = geometry.width;
        target.height = geometry.height;
        target.is_minimized = false;
        trace_live(format_args!(
            "selector request target={target_id} language={target_language} mode={:?} trigger={} stability_wait_ms={}",
            recognition_settings.mode,
            recognition_settings.trigger_key,
            recognition_settings.stability_wait_ms
        ));
        if self.backend.runs.is_busy()? {
            return Err(BackendFailure::arguments(
                "请等待当前图片或文本任务结束后再启动实时翻译",
            ));
        }
        trace_live(format_args!(
            "target window restored and activated target={target_id}"
        ));
        if self
            .backend
            .live_active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(BackendFailure::arguments("已有实时翻译会话正在运行"));
        }
        let session_id = format!(
            "live-{}-{}",
            std::process::id(),
            NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed)
        );
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            if inner.runtime.is_some() {
                self.backend.live_active.store(false, Ordering::SeqCst);
                return Err(BackendFailure::arguments("已有实时翻译会话尚未清理"));
            }
            inner.session_id = Some(session_id.clone());
            inner.target_language = Some(target_language);
            inner.overlay_settings = Some(overlay_settings);
            inner.recognition_settings = Some(recognition_settings);
            inner.translation_settings = Some(translation_settings);
            inner.selection_previous_state = None;
        }
        trace_live(format_args!(
            "selector status published session={session_id}; dispatching selector window"
        ));
        let status = LiveSessionStatus {
            state: LiveSessionState::Selecting,
            session_id: Some(session_id),
            target: Some(target),
            roi: None,
            message: "请在目标窗口上框选字幕区域。".to_owned(),
            latest_revision: 0,
            metrics: LiveMetrics::default(),
        };
        replace_status(&self.status, app, status.clone());
        if let Err(error) = create_selector_window(app, geometry) {
            trace_live(format_args!("selector setup failed: {error}"));
            self.reset(app);
            return Err(error);
        }
        Ok(status)
    }

    fn confirm_selection(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
        roi: LiveRoi,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let roi = roi.validate().map_err(BackendFailure::arguments)?;
        let normalized = roi.normalized().map_err(BackendFailure::arguments)?;
        let current = self.current_status()?;
        require_session(&current, session_id)?;
        if current.state != LiveSessionState::Selecting {
            return Err(BackendFailure::arguments("当前会话不在区域选择状态"));
        }
        let mut target = current
            .target
            .clone()
            .ok_or_else(|| BackendFailure::internal("实时会话缺少目标窗口"))?;
        platform::activate_target_window(&target.id)?;
        let geometry = platform::target_geometry(&target.id)?;
        target.width = geometry.width;
        target.height = geometry.height;
        target.is_minimized = false;
        let display_roi = normalized
            .to_physical(geometry.width, geometry.height)
            .ok_or_else(|| BackendFailure::arguments("ROI 无法映射到当前目标客户区"))?;
        let has_runtime = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?
            .runtime
            .is_some();
        if has_runtime {
            let (state, overlay_settings) = {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
                let state = inner
                    .selection_previous_state
                    .take()
                    .unwrap_or(LiveSessionState::Running);
                let runtime = inner
                    .runtime
                    .as_mut()
                    .ok_or_else(|| BackendFailure::internal("实时运行时已丢失"))?;
                let mut config = runtime
                    .config
                    .lock()
                    .map_err(|_| BackendFailure::internal("实时 ROI 锁已损坏"))?;
                config.roi = normalized;
                config.client_width = display_roi.client_width;
                config.client_height = display_roi.client_height;
                config.roi_version = config.roi_version.saturating_add(1);
                runtime
                    .paused
                    .store(state == LiveSessionState::Paused, Ordering::SeqCst);
                runtime.latest.clear();
                runtime.latest.wake();
                (state, config.overlay_settings)
            };
            close_window(app, SELECTOR_LABEL);
            position_overlay(app, geometry, overlay_settings);
            return update_status(&self.status, app, |status| {
                status.state = state;
                status.roi = Some(display_roi);
                status.message = if state == LiveSessionState::Paused {
                    "已更新字幕区域，会话保持暂停。".to_owned()
                } else {
                    "已更新字幕区域，实时翻译继续运行。".to_owned()
                };
            });
        }
        self.start_runtime(app, session_id, target, normalized, display_roi, geometry)
    }

    fn start_runtime(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
        target: CaptureWindowInfo,
        roi: NormalizedRoi,
        display_roi: LiveRoi,
        geometry: platform::TargetGeometry,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let (target_language, overlay_settings, recognition_settings, translation_settings) = {
            let inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            (
                inner
                    .target_language
                    .clone()
                    .ok_or_else(|| BackendFailure::internal("实时会话缺少目标语言"))?,
                inner
                    .overlay_settings
                    .ok_or_else(|| BackendFailure::internal("实时会话缺少浮层设置"))?,
                inner
                    .recognition_settings
                    .clone()
                    .ok_or_else(|| BackendFailure::internal("实时会话缺少识别触发设置"))?,
                inner
                    .translation_settings
                    .clone()
                    .ok_or_else(|| BackendFailure::internal("实时会话缺少翻译提示设置"))?,
            )
        };
        create_overlay_window(app, session_id, geometry, overlay_settings)?;
        let config = Arc::new(Mutex::new(LiveConfig {
            roi,
            roi_version: 1,
            client_width: display_roi.client_width,
            client_height: display_roi.client_height,
            target_language,
            overlay_settings,
            recognition_settings,
            translation_settings,
        }));
        let metrics = Arc::new(Mutex::new(LiveMetrics::default()));
        let latest = Arc::new(LatestFrameSlot::default());
        let terminal_error = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let cancellation = CancellationToken::new();
        let auto_paused = Arc::new(AtomicBool::new(false));
        let translation_cancellation = Arc::new(Mutex::new(None));
        let runner = SessionLoop {
            app: app.clone(),
            backend: self.backend.clone(),
            status: Arc::clone(&self.status),
            session_id: session_id.to_owned(),
            target_id: target.id.clone(),
            config: Arc::clone(&config),
            stop: Arc::clone(&stop),
            paused: Arc::clone(&paused),
            cancellation: cancellation.clone(),
            auto_paused: Arc::clone(&auto_paused),
            translation_cancellation: Arc::clone(&translation_cancellation),
            latest: Arc::clone(&latest),
            terminal_error,
            metrics,
        };
        let join = match thread::Builder::new()
            .name("smodeltrans-live-session".to_owned())
            .spawn(move || runner.run())
        {
            Ok(join) => join,
            Err(error) => {
                let failure =
                    BackendFailure::internal(format!("无法启动实时翻译工作线程: {error}"));
                self.fail(app, session_id, target, display_roi, failure.message());
                return Err(failure);
            }
        };
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            inner.selection_previous_state = None;
            inner.runtime = Some(SessionRuntime {
                id: session_id.to_owned(),
                config,
                stop,
                paused,
                cancellation,
                auto_paused,
                translation_cancellation,
                latest,
                join: Some(join),
            });
        }
        close_window(app, SELECTOR_LABEL);
        update_status(&self.status, app, |status| {
            status.state = LiveSessionState::Warming;
            status.target = Some(target);
            status.roi = Some(display_roi);
            status.message =
                "正在连接 Windows Graphics Capture，并预热 PP-OCRv5 与 Hy-MT2 模型。".to_owned();
        })
    }

    fn begin_roi_update(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let current = self.current_status()?;
        require_session(&current, session_id)?;
        if !matches!(
            current.state,
            LiveSessionState::Running | LiveSessionState::Paused
        ) {
            return Err(BackendFailure::arguments(
                "仅运行中或暂停的会话可以重新选择区域",
            ));
        }
        let target = current
            .target
            .as_ref()
            .ok_or_else(|| BackendFailure::internal("实时会话缺少目标窗口"))?;
        platform::activate_target_window(&target.id)?;
        let geometry = platform::target_geometry(&target.id)?;
        trace_live(format_args!("target window activated target={}", target.id));
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            inner.selection_previous_state = Some(current.state);
            let runtime = inner
                .runtime
                .as_mut()
                .ok_or_else(|| BackendFailure::internal("实时运行时已丢失"))?;
            runtime.auto_paused.store(false, Ordering::SeqCst);
            runtime.paused.store(true, Ordering::SeqCst);
            runtime.latest.wake();
        }
        let selecting = update_status(&self.status, app, |status| {
            status.state = LiveSessionState::Selecting;
            status.message = "请重新框选字幕区域；确认前不会发布新字幕。".to_owned();
        })?;
        if let Err(error) = create_selector_window(app, geometry) {
            let _ = self.cancel_selection(app, session_id);
            return Err(error);
        }
        Ok(selecting)
    }

    fn cancel_selection(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let current = self.current_status()?;
        require_session(&current, session_id)?;
        if current.state != LiveSessionState::Selecting {
            return Err(BackendFailure::arguments("当前会话不在区域选择状态"));
        }
        let restored = {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            let restored = inner
                .selection_previous_state
                .take()
                .unwrap_or(LiveSessionState::Idle);
            if let Some(runtime) = inner.runtime.as_mut() {
                runtime
                    .paused
                    .store(restored == LiveSessionState::Paused, Ordering::SeqCst);
                runtime.latest.wake();
            }
            restored
        };
        close_window(app, SELECTOR_LABEL);
        if restored == LiveSessionState::Idle {
            return Ok(self.reset(app));
        }
        update_status(&self.status, app, |status| {
            status.state = restored;
            status.message = if restored == LiveSessionState::Paused {
                "已取消重新选区，会话保持暂停。".to_owned()
            } else {
                "已取消重新选区，实时翻译继续运行。".to_owned()
            };
        })
    }

    fn set_paused(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
        paused: bool,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let current = self.current_status()?;
        require_session(&current, session_id)?;
        let target_id = current
            .target
            .as_ref()
            .ok_or_else(|| BackendFailure::internal("实时会话缺少目标窗口"))?
            .id
            .clone();
        let required = if paused {
            LiveSessionState::Running
        } else {
            LiveSessionState::Paused
        };
        if current.state != required {
            return Err(BackendFailure::arguments(if paused {
                "仅运行中的会话可以暂停"
            } else {
                "仅暂停的会话可以继续"
            }));
        }
        if !paused && platform::target_is_minimized(&target_id)? {
            return Err(BackendFailure::arguments(
                "目标窗口仍处于最小化状态，恢复窗口后才能继续",
            ));
        }
        let inner = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
        let runtime = inner
            .runtime
            .as_ref()
            .filter(|runtime| runtime.id == session_id)
            .ok_or_else(|| BackendFailure::internal("实时运行时已丢失"))?;
        if paused {
            runtime.auto_paused.store(false, Ordering::SeqCst);
        }
        runtime.paused.store(paused, Ordering::SeqCst);
        runtime.latest.wake();
        drop(inner);
        update_status(&self.status, app, |status| {
            status.state = if paused {
                LiveSessionState::Paused
            } else {
                LiveSessionState::Running
            };
            status.message = if paused {
                "实时翻译已暂停。".to_owned()
            } else {
                "实时翻译继续运行。".to_owned()
            };
        })
    }

    fn interrupt_translation(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let current = self.current_status()?;
        require_session(&current, session_id)?;
        let mut interrupted = false;
        {
            let inner = self
                .inner
                .lock()
                .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?;
            let runtime = inner
                .runtime
                .as_ref()
                .filter(|runtime| runtime.id == session_id)
                .ok_or_else(|| BackendFailure::internal("实时运行时已丢失"))?;
            if let Ok(active) = runtime.translation_cancellation.lock() {
                if let Some(token) = active.as_ref() {
                    token.cancel();
                    interrupted = true;
                }
            }
            if interrupted {
                runtime.latest.clear();
                runtime.latest.wake();
            }
        }
        update_status(&self.status, app, |status| {
            status.message = if interrupted {
                "已跳过当前翻译，等待下一条字幕。".to_owned()
            } else {
                "当前没有正在进行的翻译。".to_owned()
            };
            status.metrics = current.metrics.clone();
        })
    }

    fn stop(
        &self,
        app: &tauri::AppHandle,
        session_id: Option<&str>,
    ) -> Result<LiveSessionStatus, BackendFailure> {
        let current = self.current_status()?;
        if current.state == LiveSessionState::Idle {
            return Ok(self.reset(app));
        }
        if let (Some(requested), Some(active)) = (session_id, current.session_id.as_deref()) {
            if requested != active {
                return Ok(current);
            }
        }
        let _ = update_status(&self.status, app, |status| {
            status.state = LiveSessionState::Stopping;
            status.message = "正在停止实时翻译并释放捕获资源。".to_owned();
        });
        close_window(app, SELECTOR_LABEL);
        let runtime = self
            .inner
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话锁已损坏"))?
            .runtime
            .take();
        if let Some(mut runtime) = runtime {
            runtime.stop.store(true, Ordering::SeqCst);
            runtime.cancellation.cancel();
            if let Ok(active) = runtime.translation_cancellation.lock() {
                if let Some(token) = active.as_ref() {
                    token.cancel();
                }
            }
            runtime.latest.wake();
            if let Some(join) = runtime.join.take() {
                join.join()
                    .map_err(|_| BackendFailure::internal("实时翻译工作线程异常退出"))?;
            }
        }
        Ok(self.reset(app))
    }

    fn fail(
        &self,
        app: &tauri::AppHandle,
        session_id: &str,
        target: CaptureWindowInfo,
        roi: LiveRoi,
        message: &str,
    ) {
        self.backend.live_active.store(false, Ordering::SeqCst);
        close_window(app, SELECTOR_LABEL);
        close_window(app, OVERLAY_LABEL);
        replace_status(
            &self.status,
            app,
            LiveSessionStatus {
                state: LiveSessionState::Error,
                session_id: Some(session_id.to_owned()),
                target: Some(target),
                roi: Some(roi),
                message: message.to_owned(),
                latest_revision: 0,
                metrics: LiveMetrics::default(),
            },
        );
    }

    fn reset(&self, app: &tauri::AppHandle) -> LiveSessionStatus {
        self.backend.live_active.store(false, Ordering::SeqCst);
        close_window(app, SELECTOR_LABEL);
        close_window(app, OVERLAY_LABEL);
        if let Ok(mut inner) = self.inner.lock() {
            *inner = ManagerInner::default();
        }
        let idle = LiveSessionStatus::default();
        replace_status(&self.status, app, idle.clone());
        idle
    }
}

struct RecognizedFrame {
    source_text: String,
    region_count: u32,
    regions: Vec<RegionRecord>,
}

struct SessionLoop {
    app: tauri::AppHandle,
    backend: BackendState,
    status: Arc<Mutex<LiveSessionStatus>>,
    session_id: String,
    target_id: String,
    config: Arc<Mutex<LiveConfig>>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    cancellation: CancellationToken,
    auto_paused: Arc<AtomicBool>,
    translation_cancellation: Arc<Mutex<Option<CancellationToken>>>,
    latest: Arc<LatestFrameSlot>,
    terminal_error: Arc<Mutex<Option<String>>>,
    metrics: Arc<Mutex<LiveMetrics>>,
}

impl SessionLoop {
    fn run(self) {
        if let Err(error) = self.prepare_models() {
            self.finish(None, Some(error.message().to_owned()));
            return;
        }
        let capture = match self.await_capture_start() {
            Ok(Some(capture)) => capture,
            Ok(None) => {
                self.finish(None, None);
                return;
            }
            Err(error) => {
                self.finish(None, Some(error.message().to_owned()));
                return;
            }
        };
        self.run_capture(capture);
    }

    fn prepare_models(&self) -> Result<(), BackendFailure> {
        self.cancellation.check()?;
        let settings = self
            .backend
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        let (target_language, memory) = {
            let config = self
                .config
                .lock()
                .map_err(|_| BackendFailure::internal("实时配置锁已损坏"))?;
            (
                config.target_language.clone(),
                config.translation_settings.memory_config(),
            )
        };
        let (gpu, states) = {
            let mut engine = self
                .backend
                .engine
                .lock()
                .map_err(|_| BackendFailure::internal("模型引擎锁已损坏"))?;
            if engine.is_none() {
                *engine = Some(BackendEngine::new(settings)?);
            }
            let engine = engine
                .as_mut()
                .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
            engine.prepare_live_pipeline(&target_language, memory, &self.cancellation)?;
            (engine.gpu_resource_info()?, engine.model_states())
        };
        self.backend.set_model_states(states.0, states.1);
        self.backend.touch_activity();

        if let Some(gpu) = gpu {
            trace_live(format_args!(
                "GPU ready name={} memory={}/{} MiB mode={}",
                gpu.name, gpu.free_memory_mib, gpu.total_memory_mib, gpu.execution_mode
            ));
            self.update_metrics(|metrics| {
                metrics.gpu_name = gpu.name;
                metrics.gpu_total_memory_mib = gpu.total_memory_mib;
                metrics.gpu_free_memory_mib = gpu.free_memory_mib;
                metrics.gpu_execution_mode = gpu.execution_mode.to_owned();
            });
        }
        let _ = update_status(&self.status, &self.app, |status| {
            status.message = "模型已预热，正在连接 Windows Graphics Capture。".to_owned();
            status.metrics = self.metrics_snapshot();
        });
        Ok(())
    }

    fn await_capture_start(&self) -> Result<Option<platform::CaptureWorker>, BackendFailure> {
        let (sender, receiver) =
            mpsc::sync_channel::<Result<platform::CaptureWorker, BackendFailure>>(1);
        let target_id = self.target_id.clone();
        let latest = Arc::clone(&self.latest);
        let terminal_error = Arc::clone(&self.terminal_error);
        let config = Arc::clone(&self.config);
        let metrics = Arc::clone(&self.metrics);
        let paused = Arc::clone(&self.paused);
        thread::Builder::new()
            .name("smodeltrans-wgc-startup".to_owned())
            .spawn(move || {
                let result = platform::start_capture(
                    &target_id,
                    latest,
                    terminal_error,
                    config,
                    metrics,
                    paused,
                );
                if let Err(error) = sender.send(result) {
                    if let Ok(capture) = error.0 {
                        let _ = capture.stop();
                    }
                }
            })
            .map_err(|error| {
                BackendFailure::internal(format!("无法启动窗口捕获初始化线程: {error}"))
            })?;
        await_capture_start_result(&self.stop, &receiver)
    }

    fn run_capture(&self, capture: platform::CaptureWorker) {
        let mut stability = StabilityScheduler::default();
        let mut configured_stability_wait_ms = None;

        let mut roi_version = 0;
        let mut last_frame: Option<OwnedFrame> = None;
        let mut last_geometry_sync = Instant::now() - Duration::from_secs(1);
        let mut terminal_message = None;
        let mut debug_sequence = 0_u64;
        let mut last_target_state_sync = Instant::now() - Duration::from_secs(1);
        let mut target_minimized = None;

        let mut trigger_pending = false;
        let mut waiting_for_trigger = false;
        let mut previous_trigger_active = false;
        let mut trigger_wait_started: Option<Instant> = None;

        while !self.stop.load(Ordering::SeqCst) {
            if let Some(error) = self
                .terminal_error
                .lock()
                .ok()
                .and_then(|error| error.clone())
            {
                terminal_message = Some(error);
                break;
            }
            if last_target_state_sync.elapsed() >= Duration::from_millis(250) {
                if let Err(error) = self.sync_target_window_state(&mut target_minimized) {
                    terminal_message = Some(error.message().to_owned());
                    break;
                }
                last_target_state_sync = Instant::now();
            }
            if self.paused.load(Ordering::SeqCst) {
                let _ = self.latest.wait_take(Duration::from_millis(100));
                continue;
            }
            let config = match self.config.lock() {
                Ok(config) => config.clone(),
                Err(_) => {
                    terminal_message = Some("实时 ROI 锁已损坏".to_owned());
                    break;
                }
            };
            if configured_stability_wait_ms != Some(config.recognition_settings.stability_wait_ms) {
                stability.set_settle_ms(config.recognition_settings.stability_wait_ms);
                configured_stability_wait_ms = Some(config.recognition_settings.stability_wait_ms);
            }

            let trigger_active = platform::recognition_is_active(&config.recognition_settings);
            let trigger_fired = if config.recognition_settings.mode
                == LiveRecognitionMode::KeyTrigger
            {
                let fired = match config.recognition_settings.trigger_event {
                    LiveRecognitionTrigger::Press => trigger_active && !previous_trigger_active,
                    LiveRecognitionTrigger::Release => !trigger_active && previous_trigger_active,
                };
                previous_trigger_active = trigger_active;
                fired
            } else {
                previous_trigger_active = false;
                false
            };
            if config.recognition_settings.mode == LiveRecognitionMode::KeyTrigger
                && trigger_fired
                && !trigger_pending
            {
                trigger_pending = true;
                trigger_wait_started = Some(Instant::now());
            } else if config.recognition_settings.mode != LiveRecognitionMode::KeyTrigger {
                trigger_pending = false;
                trigger_wait_started = None;
            }
            if config.recognition_settings.mode == LiveRecognitionMode::KeyTrigger
                && !trigger_pending
            {
                trigger_wait_started = None;
                stability.reset();
                last_frame = None;
                if !waiting_for_trigger {
                    waiting_for_trigger = true;
                    let instruction = match config.recognition_settings.trigger_event {
                        LiveRecognitionTrigger::Press => {
                            "已暂停识别，请按下已配置的触发键执行一次 OCR。".to_owned()
                        }
                        LiveRecognitionTrigger::Release => {
                            "已暂停识别，请按下并松开已配置的触发键执行一次 OCR。".to_owned()
                        }
                    };
                    let _ = update_status(&self.status, &self.app, |status| {
                        status.state = LiveSessionState::Running;
                        status.message = instruction;
                        status.metrics = self.metrics_snapshot();
                    });
                }
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            if waiting_for_trigger {
                waiting_for_trigger = false;
                let timeout_ms = config.recognition_settings.key_trigger_timeout_ms;
                let _ = update_status(&self.status, &self.app, |status| {
                    status.state = LiveSessionState::Running;
                    status.message = format!(
                        "检测到触发键，等待字幕稳定，最多 {timeout_ms} 毫秒后执行一次 OCR。"
                    );
                    status.metrics = self.metrics_snapshot();
                });
            }
            if last_geometry_sync.elapsed() >= Duration::from_millis(500) {
                match platform::target_geometry(&self.target_id) {
                    Ok(geometry) => {
                        position_overlay(&self.app, geometry, config.overlay_settings);
                        if geometry.width != config.client_width
                            || geometry.height != config.client_height
                        {
                            let mut current = match self.config.lock() {
                                Ok(current) => current,
                                Err(_) => {
                                    terminal_message =
                                        Some("实时配置锁已损坏，无法同步窗口尺寸。".to_owned());
                                    break;
                                }
                            };
                            current.client_width = geometry.width;
                            current.client_height = geometry.height;
                            current.roi_version = current.roi_version.saturating_add(1);
                            self.latest.clear();
                            self.latest.wake();
                        }
                    }
                    Err(error) => {
                        terminal_message = Some(error.message().to_owned());
                        break;
                    }
                }
                last_geometry_sync = Instant::now();
            }
            if config.roi_version != roi_version {
                roi_version = config.roi_version;
                stability.reset();
                last_frame = None;
            }
            let received = self.latest.wait_take(Duration::from_millis(100));
            let now_ms = epoch_millis();
            let triggered_probe = config.recognition_settings.mode
                == LiveRecognitionMode::KeyTrigger
                && trigger_pending;
            let should_probe = if let Some(frame) = received {
                if frame.roi_version != roi_version {
                    false
                } else {
                    let probe = stability.observe(&frame, now_ms);
                    last_frame = Some(frame);
                    probe
                }
            } else {
                stability.tick(now_ms)
            };
            let trigger_timeout =
                Duration::from_millis(config.recognition_settings.key_trigger_timeout_ms);
            let forced_by_timeout = key_trigger_timeout_reached(
                triggered_probe,
                trigger_wait_started.map(|started| started.elapsed()),
                last_frame.is_some(),
                trigger_timeout,
            );
            if !should_probe && !forced_by_timeout {
                continue;
            }
            if triggered_probe {
                trigger_pending = false;
                trigger_wait_started = None;
            }
            if forced_by_timeout {
                let timeout_ms = config.recognition_settings.key_trigger_timeout_ms;
                let _ = update_status(&self.status, &self.app, |status| {
                    status.state = LiveSessionState::Running;
                    status.message =
                        format!("字幕在 {timeout_ms} 毫秒内未稳定，使用最新画面执行 OCR。");
                    status.metrics = self.metrics_snapshot();
                });
            }
            let Some(frame) = last_frame.as_ref() else {
                continue;
            };
            let frame_version = frame.roi_version;
            let observed_at_epoch_ms = frame.observed_at_epoch_ms;
            let ocr_started = Instant::now();
            let mut recognized = match self.recognize(frame, &config.target_language) {
                Ok(result) => result,
                Err(error) if error.code() == BackendFailureCode::Cancelled => break,
                Err(error) => {
                    debug_sequence = debug_sequence.saturating_add(1);
                    self.emit_debug_record(LiveDebugRecord {
                        session_id: self.session_id.clone(),
                        sequence: debug_sequence,
                        stage: LiveDebugStage::Ocr,
                        outcome: LiveDebugOutcome::Failed,
                        source_text: String::new(),
                        translated_text: None,
                        target_language: config.target_language.clone(),
                        region_count: 0,
                        roi_version: frame_version,
                        duration_ms: elapsed_millis(ocr_started.elapsed()),
                        message: Some(error.message().to_owned()),
                        observed_at_epoch_ms,
                    });
                    terminal_message = Some(error.message().to_owned());
                    break;
                }
            };
            let ocr_ms = elapsed_millis(ocr_started.elapsed());
            self.update_metrics(|metrics| {
                metrics.ocr_runs = metrics.ocr_runs.saturating_add(1);
                metrics.last_ocr_ms = ocr_ms;
            });
            self.backend.touch_activity();
            if !self.roi_is_current(frame_version) {
                continue;
            }
            let _ = update_status(&self.status, &self.app, |status| {
                if status.state == LiveSessionState::Warming {
                    status.state = LiveSessionState::Running;
                    status.message = "实时翻译正在运行。".to_owned();
                }
                status.metrics = self.metrics_snapshot();
            });
            let confirmed_text = recognized.source_text.clone();
            debug_sequence = debug_sequence.saturating_add(1);
            self.emit_debug_record(LiveDebugRecord {
                session_id: self.session_id.clone(),
                sequence: debug_sequence,
                stage: LiveDebugStage::Ocr,
                outcome: LiveDebugOutcome::Confirmed,
                source_text: debug_text(&recognized.source_text),
                translated_text: None,
                target_language: config.target_language.clone(),
                region_count: recognized.region_count,
                roi_version: frame_version,
                duration_ms: ocr_ms,
                message: None,
                observed_at_epoch_ms,
            });

            let needs_translation = !confirmed_text.is_empty() && !recognized.regions.is_empty();
            let translation_cancellation = if needs_translation {
                match self.begin_translation() {
                    Ok(token) => Some(token),
                    Err(error) if error.code() == BackendFailureCode::Cancelled => break,
                    Err(error) => {
                        terminal_message = Some(error.message().to_owned());
                        break;
                    }
                }
            } else {
                None
            };
            let revision = self.next_revision();
            let (translated_text, translation_ms, outcome) = match config.overlay_settings.mode {
                LiveOverlayMode::Subtitle if !needs_translation => {
                    (String::new(), 0, LiveDebugOutcome::SkippedEmptySource)
                }
                LiveOverlayMode::Subtitle => {
                    self.emit_subtitle(
                        revision,
                        frame,
                        &confirmed_text,
                        "",
                        &[],
                        LiveOverlayMode::Subtitle,
                        true,
                    );
                    let translation_started = Instant::now();
                    let translated = match self.translate_subtitle_streaming(
                        &confirmed_text,
                        &config.target_language,
                        &config.translation_settings.supplemental_prompt,
                        config.translation_settings.memory_config(),
                        translation_cancellation
                            .as_ref()
                            .expect("translation cancellation token is initialized"),
                        revision,
                        frame,
                    ) {
                        Ok(translated) => translated,
                        Err(error) if error.code() == BackendFailureCode::Cancelled => {
                            self.finish_translation();
                            break;
                        }
                        Err(error) => {
                            self.finish_translation();
                            debug_sequence = debug_sequence.saturating_add(1);
                            self.emit_debug_record(LiveDebugRecord {
                                session_id: self.session_id.clone(),
                                sequence: debug_sequence,
                                stage: LiveDebugStage::Translation,
                                outcome: LiveDebugOutcome::Failed,
                                source_text: debug_text(&confirmed_text),
                                translated_text: None,
                                target_language: config.target_language.clone(),
                                region_count: recognized.region_count,
                                roi_version: frame_version,
                                duration_ms: elapsed_millis(translation_started.elapsed()),
                                message: Some(error.message().to_owned()),
                                observed_at_epoch_ms,
                            });
                            terminal_message = Some(error.message().to_owned());
                            break;
                        }
                    };
                    let translation_ms = elapsed_millis(translation_started.elapsed());
                    self.update_metrics(|metrics| {
                        metrics.translation_runs = metrics.translation_runs.saturating_add(1);
                        metrics.last_translation_ms = translation_ms;
                    });
                    (translated, translation_ms, LiveDebugOutcome::Completed)
                }
                LiveOverlayMode::RegionReplace if !needs_translation => {
                    (String::new(), 0, LiveDebugOutcome::SkippedEmptySource)
                }
                LiveOverlayMode::RegionReplace => {
                    self.emit_subtitle(
                        revision,
                        frame,
                        &confirmed_text,
                        "",
                        &recognized.regions,
                        LiveOverlayMode::RegionReplace,
                        true,
                    );
                    let translation_started = Instant::now();
                    match self.translate_regions_streaming(
                        &mut recognized.regions,
                        &config.target_language,
                        &config.translation_settings.supplemental_prompt,
                        config.translation_settings.memory_config(),
                        translation_cancellation
                            .as_ref()
                            .expect("translation cancellation token is initialized"),
                        revision,
                        frame,
                        LiveOverlayMode::RegionReplace,
                    ) {
                        Ok(()) => {}
                        Err(error) if error.code() == BackendFailureCode::Cancelled => {
                            self.finish_translation();
                            break;
                        }
                        Err(error) => {
                            self.finish_translation();
                            debug_sequence = debug_sequence.saturating_add(1);
                            self.emit_debug_record(LiveDebugRecord {
                                session_id: self.session_id.clone(),
                                sequence: debug_sequence,
                                stage: LiveDebugStage::Translation,
                                outcome: LiveDebugOutcome::Failed,
                                source_text: debug_text(&confirmed_text),
                                translated_text: None,
                                target_language: config.target_language.clone(),
                                region_count: recognized.region_count,
                                roi_version: frame_version,
                                duration_ms: elapsed_millis(translation_started.elapsed()),
                                message: Some(error.message().to_owned()),
                                observed_at_epoch_ms,
                            });
                            terminal_message = Some(error.message().to_owned());
                            break;
                        }
                    }
                    let translation_ms = elapsed_millis(translation_started.elapsed());
                    self.update_metrics(|metrics| {
                        metrics.translation_runs = metrics.translation_runs.saturating_add(1);
                        metrics.last_translation_ms = translation_ms;
                    });
                    (
                        live_translated_region_text(&recognized.regions),
                        translation_ms,
                        LiveDebugOutcome::Completed,
                    )
                }
            };
            if translation_cancellation.is_some() {
                self.finish_translation();
            }
            debug_sequence = debug_sequence.saturating_add(1);
            self.emit_debug_record(LiveDebugRecord {
                session_id: self.session_id.clone(),
                sequence: debug_sequence,
                stage: LiveDebugStage::Translation,
                outcome,
                source_text: debug_text(&confirmed_text),
                translated_text: Some(debug_text(&translated_text)),
                target_language: config.target_language.clone(),
                region_count: recognized.region_count,
                roi_version: frame_version,
                duration_ms: translation_ms,
                message: None,
                observed_at_epoch_ms,
            });
            if !self.roi_is_current(frame_version) {
                continue;
            }
            self.update_metrics(|metrics| {
                metrics.subtitle_publishes = metrics.subtitle_publishes.saturating_add(1);
            });
            self.emit_subtitle(
                revision,
                frame,
                &confirmed_text,
                &translated_text,
                &recognized.regions,
                config.overlay_settings.mode,
                false,
            );
            let _ = update_status(&self.status, &self.app, |status| {
                status.latest_revision = revision;
                status.metrics = self.metrics_snapshot();
                if !self.paused.load(Ordering::SeqCst) {
                    status.state = LiveSessionState::Running;
                    status.message = format!("字幕已更新，OCR {ocr_ms} ms。");
                }
            });
        }
        self.finish(Some(capture), terminal_message);
    }

    fn finish(&self, capture: Option<platform::CaptureWorker>, terminal_message: Option<String>) {
        let final_metrics = self.metrics_snapshot();
        if let Some(capture) = capture {
            let _ = capture.stop();
        }
        close_window(&self.app, SELECTOR_LABEL);
        close_window(&self.app, OVERLAY_LABEL);
        self.backend.live_active.store(false, Ordering::SeqCst);
        if !self.stop.load(Ordering::SeqCst) {
            let message = terminal_message.unwrap_or_else(|| "窗口捕获意外停止。".to_owned());
            let _ = update_status(&self.status, &self.app, |status| {
                status.state = LiveSessionState::Error;
                status.message = message;
                status.metrics = final_metrics;
            });
        }
    }

    fn sync_target_window_state(
        &self,
        last_minimized: &mut Option<bool>,
    ) -> Result<(), BackendFailure> {
        let minimized = platform::target_is_minimized(&self.target_id)?;
        if last_minimized.replace(minimized) == Some(minimized) {
            return Ok(());
        }

        if minimized {
            let was_paused = self.paused.load(Ordering::SeqCst);
            if !was_paused {
                self.auto_paused.store(true, Ordering::SeqCst);
                self.paused.store(true, Ordering::SeqCst);
            }
            self.latest.clear();
            self.latest.wake();
            set_overlay_visible(&self.app, false);
            let _ = update_status(&self.status, &self.app, |status| {
                if !was_paused {
                    status.state = LiveSessionState::Paused;
                    status.message =
                        "目标窗口已最小化，实时翻译已自动暂停；恢复窗口后将继续。".to_owned();
                } else if status.state == LiveSessionState::Paused {
                    status.message = "目标窗口已最小化，实时翻译保持暂停。".to_owned();
                }
                if let Some(target) = status.target.as_mut() {
                    target.is_minimized = true;
                    target.width = 0;
                    target.height = 0;
                }
                status.metrics = self.metrics_snapshot();
            });
            return Ok(());
        }

        let geometry = platform::target_geometry(&self.target_id)?;
        let auto_resume = self.auto_paused.swap(false, Ordering::SeqCst);
        if auto_resume {
            self.paused.store(false, Ordering::SeqCst);
            self.latest.clear();
            self.latest.wake();
        }
        position_overlay(&self.app, geometry, self.config_overlay_settings());
        set_overlay_visible(&self.app, true);
        let _ = update_status(&self.status, &self.app, |status| {
            if auto_resume {
                status.state = LiveSessionState::Running;
                status.message = "目标窗口已恢复，实时翻译继续运行。".to_owned();
            } else if status.state == LiveSessionState::Paused {
                status.message = "目标窗口已恢复，实时翻译仍处于手动暂停。".to_owned();
            }
            if let Some(target) = status.target.as_mut() {
                target.is_minimized = false;
                target.width = geometry.width;
                target.height = geometry.height;
            }
            status.metrics = self.metrics_snapshot();
        });
        Ok(())
    }

    fn config_overlay_settings(&self) -> LiveOverlaySettings {
        self.config
            .lock()
            .map(|config| config.overlay_settings)
            .unwrap_or_default()
    }

    fn roi_is_current(&self, version: u64) -> bool {
        self.config
            .lock()
            .map(|config| roi_result_is_current(version, config.roi_version))
            .unwrap_or(false)
    }

    fn metrics_snapshot(&self) -> LiveMetrics {
        self.metrics
            .lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }

    fn update_metrics(&self, update: impl FnOnce(&mut LiveMetrics)) {
        if let Ok(mut metrics) = self.metrics.lock() {
            update(&mut metrics);
        }
    }

    fn emit_debug_record(&self, record: LiveDebugRecord) {
        let _ = self.app.emit(DEBUG_RECORD_EVENT, record);
    }
    fn begin_translation(&self) -> Result<CancellationToken, BackendFailure> {
        self.cancellation.check()?;
        let token = CancellationToken::new();
        let mut active = self
            .translation_cancellation
            .lock()
            .map_err(|_| BackendFailure::internal("实时翻译取消锁已损坏"))?;
        if let Some(previous) = active.replace(token.clone()) {
            previous.cancel();
        }
        Ok(token)
    }

    fn finish_translation(&self) {
        if let Ok(mut active) = self.translation_cancellation.lock() {
            *active = None;
        }
    }

    fn next_revision(&self) -> u64 {
        self.status
            .lock()
            .map(|status| status.latest_revision.saturating_add(1))
            .unwrap_or(1)
    }

    fn emit_subtitle(
        &self,
        revision: u64,
        frame: &OwnedFrame,
        source_text: &str,
        translated_text: &str,
        regions: &[RegionRecord],
        mode: LiveOverlayMode,
        is_streaming: bool,
    ) {
        let visible_regions = if mode == LiveOverlayMode::RegionReplace {
            regions
                .iter()
                .filter_map(|region| live_subtitle_region(region, frame.roi))
                .collect()
        } else {
            Vec::new()
        };
        let _ = self.app.emit(
            SUBTITLE_EVENT,
            LiveSubtitle {
                session_id: self.session_id.clone(),
                revision,
                source_text: source_text.to_owned(),
                translated_text: translated_text.to_owned(),
                roi: frame.roi,
                regions: visible_regions,
                is_streaming,
                observed_at_epoch_ms: frame.observed_at_epoch_ms,
            },
        );
    }

    fn translate_subtitle_streaming(
        &self,
        source_text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        revision: u64,
        frame: &OwnedFrame,
    ) -> Result<String, BackendFailure> {
        let mut last_emit = Instant::now() - Duration::from_millis(40);
        self.translate_live_subtitle(
            source_text,
            target_language,
            supplemental_prompt,
            memory,
            cancellation,
            |partial| {
                if last_emit.elapsed() >= Duration::from_millis(32) {
                    self.emit_subtitle(
                        revision,
                        frame,
                        source_text,
                        partial,
                        &[],
                        LiveOverlayMode::Subtitle,
                        true,
                    );
                    last_emit = Instant::now();
                }
            },
        )
    }

    fn translate_regions_streaming(
        &self,
        regions: &mut [RegionRecord],
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        revision: u64,
        frame: &OwnedFrame,
        mode: LiveOverlayMode,
    ) -> Result<(), BackendFailure> {
        let mut streamed_regions = regions.to_vec();
        let mut last_emit = Instant::now() - Duration::from_millis(40);
        self.translate_live_regions(
            regions,
            target_language,
            supplemental_prompt,
            memory,
            cancellation,
            |order, partial| {
                if let Some(region) = streamed_regions
                    .iter_mut()
                    .find(|region| region.order == order)
                {
                    region.translated_text = partial.to_owned();
                }
                if last_emit.elapsed() >= Duration::from_millis(32) {
                    self.emit_subtitle(
                        revision,
                        frame,
                        &normalized_live_region_text(&streamed_regions),
                        &live_translated_region_text(&streamed_regions),
                        &streamed_regions,
                        mode,
                        true,
                    );
                    last_emit = Instant::now();
                }
            },
        )
    }

    fn recognize(
        &self,
        frame: &OwnedFrame,
        target_language: &str,
    ) -> Result<RecognizedFrame, BackendFailure> {
        self.cancellation.check()?;
        let image = RgbImage::from_raw(frame.width, frame.height, frame.rgb.clone())
            .ok_or_else(|| BackendFailure::internal("实时捕获帧缓冲区无效"))?;
        let decoded = DecodedImage::from_rgb_image(image, "live-frame", target_language);
        let settings = self
            .backend
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        let text_grouping_enabled = self
            .config
            .lock()
            .map_err(|_| BackendFailure::internal("实时配置锁已损坏"))?
            .recognition_settings
            .text_grouping_enabled;
        let mut engine = self
            .backend
            .engine
            .lock()
            .map_err(|_| BackendFailure::internal("模型引擎锁已损坏"))?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        let detected = engine.recognize_regions(&decoded, &self.cancellation)?;
        let detected_count = detected.len();
        let mut regions = Self::refine_live_regions(detected, text_grouping_enabled);
        finalize_live_regions(&mut regions);
        trace_live(format_args!(
            "OCR regions detected={detected_count} finalized={} text_grouping_enabled={text_grouping_enabled}",
            regions.len()
        ));
        let region_count = u32::try_from(regions.len()).unwrap_or(u32::MAX);
        let source_text = normalized_live_region_text(&regions);
        let states = engine.model_states();
        self.backend.set_model_states(states.0, states.1);
        Ok(RecognizedFrame {
            source_text,
            region_count,
            regions,
        })
    }

    fn refine_live_regions(
        detected: Vec<RegionRecord>,
        text_grouping_enabled: bool,
    ) -> Vec<RegionRecord> {
        if !text_grouping_enabled {
            return detected;
        }
        plan_live_ocr_groups(detected)
            .into_iter()
            .map(|group| {
                let fallback_text = group.source_text();
                group.into_merged_region(fallback_text)
            })
            .collect()
    }

    fn translate_live_subtitle(
        &self,
        source_text: &str,
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        on_chunk: impl FnMut(&str),
    ) -> Result<String, BackendFailure> {
        cancellation.check()?;
        let settings = self
            .backend
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        let mut engine = self
            .backend
            .engine
            .lock()
            .map_err(|_| BackendFailure::internal("模型引擎锁已损坏"))?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        let result = engine.translate_live_subtitle(
            source_text,
            target_language,
            supplemental_prompt,
            memory,
            cancellation,
            on_chunk,
        );
        let states = engine.model_states();
        self.backend.set_model_states(states.0, states.1);
        self.backend.touch_activity();
        result
    }

    fn translate_live_regions(
        &self,
        regions: &mut [RegionRecord],
        target_language: &str,
        supplemental_prompt: &str,
        memory: MemoryConfig,
        cancellation: &CancellationToken,
        on_chunk: impl FnMut(u32, &str),
    ) -> Result<(), BackendFailure> {
        cancellation.check()?;
        let settings = self
            .backend
            .settings
            .lock()
            .map_err(|_| BackendFailure::internal("后端配置锁已损坏"))?
            .clone()
            .map_err(BackendFailure::arguments)?;
        let mut engine = self
            .backend
            .engine
            .lock()
            .map_err(|_| BackendFailure::internal("模型引擎锁已损坏"))?;
        if engine.is_none() {
            *engine = Some(BackendEngine::new(settings)?);
        }
        let engine = engine
            .as_mut()
            .ok_or_else(|| BackendFailure::internal("Candle 后端未初始化"))?;
        let result = engine.translate_live_regions(
            regions,
            target_language,
            supplemental_prompt,
            memory,
            cancellation,
            on_chunk,
        );
        let states = engine.model_states();
        self.backend.set_model_states(states.0, states.1);
        self.backend.touch_activity();
        result
    }
}

fn live_subtitle_region(region: &RegionRecord, roi: LiveRoi) -> Option<LiveSubtitleRegion> {
    let local_left = region.quad_points.iter().map(|point| point[0]).min()?;
    let local_top = region.quad_points.iter().map(|point| point[1]).min()?;
    let local_right = region.quad_points.iter().map(|point| point[0]).max()?;
    let local_bottom = region.quad_points.iter().map(|point| point[1]).max()?;
    let client_width = i64::from(roi.client_width);
    let client_height = i64::from(roi.client_height);
    let left = (i64::from(roi.x) + i64::from(local_left)).clamp(0, client_width);
    let top = (i64::from(roi.y) + i64::from(local_top)).clamp(0, client_height);
    let right = (i64::from(roi.x) + i64::from(local_right)).clamp(0, client_width);
    let bottom = (i64::from(roi.y) + i64::from(local_bottom)).clamp(0, client_height);
    (right > left && bottom > top).then(|| LiveSubtitleRegion {
        bounds: LiveSubtitleRegionBounds {
            left: left as u32,
            top: top as u32,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
        },
        source_text: region.source_text.clone(),
        translated_text: region.translated_text.clone(),
    })
}

fn await_capture_start_result<T>(
    stop: &AtomicBool,
    receiver: &mpsc::Receiver<Result<T, BackendFailure>>,
) -> Result<Option<T>, BackendFailure> {
    loop {
        if stop.load(Ordering::SeqCst) {
            return Ok(None);
        }
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(result) => return result.map(Some),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                return Err(BackendFailure::internal(
                    "Windows Graphics Capture 启动线程意外退出",
                ));
            }
        }
    }
}

fn target_by_id(target_id: &str) -> Result<CaptureWindowInfo, BackendFailure> {
    platform::list_target_windows()?
        .into_iter()
        .find(|target| target.id == target_id)
        .ok_or_else(|| BackendFailure::arguments("目标窗口已关闭或不再可捕获"))
}

fn require_session(status: &LiveSessionStatus, session_id: &str) -> Result<(), BackendFailure> {
    if status.session_id.as_deref() != Some(session_id) {
        return Err(BackendFailure::arguments("实时会话标识已过期"));
    }
    Ok(())
}

fn replace_status(
    shared: &Arc<Mutex<LiveSessionStatus>>,
    app: &tauri::AppHandle,
    status: LiveSessionStatus,
) {
    if let Ok(mut current) = shared.lock() {
        *current = status.clone();
    }
    let _ = app.emit(STATUS_EVENT, status);
}

fn update_status(
    shared: &Arc<Mutex<LiveSessionStatus>>,
    app: &tauri::AppHandle,
    update: impl FnOnce(&mut LiveSessionStatus),
) -> Result<LiveSessionStatus, BackendFailure> {
    let snapshot = {
        let mut status = shared
            .lock()
            .map_err(|_| BackendFailure::internal("实时会话状态锁已损坏"))?;
        update(&mut status);
        status.clone()
    };
    let _ = app.emit(STATUS_EVENT, snapshot.clone());
    Ok(snapshot)
}

fn elapsed_millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn overlay_bounds(
    geometry: platform::TargetGeometry,
    settings: LiveOverlaySettings,
) -> (i32, i32, u32, u32) {
    if settings.mode == LiveOverlayMode::RegionReplace {
        return (geometry.x, geometry.y, geometry.width, geometry.height);
    }
    let offset = i32::try_from(settings.offset).unwrap_or(i32::MAX);
    match settings.attachment {
        LiveOverlayAttachment::Top => {
            let height = OVERLAY_EDGE_THICKNESS.min(geometry.height);
            (
                geometry.x,
                geometry
                    .y
                    .saturating_sub(i32::try_from(height).unwrap_or(i32::MAX))
                    .saturating_sub(offset),
                geometry.width,
                height,
            )
        }
        LiveOverlayAttachment::Bottom => {
            let height = OVERLAY_EDGE_THICKNESS.min(geometry.height);
            (
                geometry.x,
                geometry
                    .y
                    .saturating_add(i32::try_from(geometry.height).unwrap_or(i32::MAX))
                    .saturating_add(offset),
                geometry.width,
                height,
            )
        }
        LiveOverlayAttachment::Left => {
            let width = OVERLAY_EDGE_THICKNESS.min(geometry.width);
            (
                geometry
                    .x
                    .saturating_sub(i32::try_from(width).unwrap_or(i32::MAX))
                    .saturating_sub(offset),
                geometry.y,
                width,
                geometry.height,
            )
        }
        LiveOverlayAttachment::Right => {
            let width = OVERLAY_EDGE_THICKNESS.min(geometry.width);
            (
                geometry
                    .x
                    .saturating_add(i32::try_from(geometry.width).unwrap_or(i32::MAX))
                    .saturating_add(offset),
                geometry.y,
                width,
                geometry.height,
            )
        }
    }
}

fn overlay_url_path(session_id: &str, settings: LiveOverlaySettings) -> String {
    let source_visibility = if settings.show_source { "1" } else { "0" };
    let region_box_visibility = if settings.show_region_boxes { "1" } else { "0" };
    format!(
        "index.html?liveSessionId={session_id}&liveOverlayMode={}&showSource={source_visibility}&showRegionBoxes={region_box_visibility}",
        settings.mode_query_value(),
    )
}

fn create_overlay_window(
    app: &tauri::AppHandle,
    session_id: &str,
    geometry: platform::TargetGeometry,
    settings: LiveOverlaySettings,
) -> Result<(), BackendFailure> {
    let app = app.clone();
    let window_app = app.clone();
    let session_id = session_id.to_owned();
    run_window_operation_on_main_thread(app, move || {
        close_window_now(&window_app, OVERLAY_LABEL);
        let (x, y, width, height) = overlay_bounds(geometry, settings);
        let url = WebviewUrl::App(overlay_url_path(&session_id, settings).into());
        let window = WebviewWindowBuilder::new(&window_app, OVERLAY_LABEL, url)
            .title("smodeltrans 实时字幕")
            .decorations(false)
            .transparent(true)
            .shadow(false)
            .always_on_top(true)
            .focusable(false)
            .skip_taskbar(true)
            .resizable(false)
            // Bounds are applied as physical pixels below. Logical builder
            // bounds are incorrect on a display whose scale factor is not 1.
            .visible(false)
            .build()
            .map_err(|error| BackendFailure::internal(format!("创建实时字幕窗口失败: {error}")))?;
        window
            .set_position(PhysicalPosition::new(x, y))
            .map_err(|error| {
                BackendFailure::internal(format!("设置实时字幕窗口位置失败: {error}"))
            })?;
        window
            .set_size(PhysicalSize::new(width, height))
            .map_err(|error| {
                BackendFailure::internal(format!("设置实时字幕窗口尺寸失败: {error}"))
            })?;
        window
            .set_ignore_cursor_events(true)
            .map_err(|error| BackendFailure::internal(format!("启用字幕鼠标穿透失败: {error}")))?;
        window
            .show()
            .map_err(|error| BackendFailure::internal(format!("显示实时字幕窗口失败: {error}")))?;
        Ok(())
    })
}

fn set_selector_window_bounds(
    window: &tauri::WebviewWindow,
    geometry: platform::TargetGeometry,
) -> Result<(), BackendFailure> {
    window
        .set_position(PhysicalPosition::new(geometry.x, geometry.y))
        .map_err(|error| {
            BackendFailure::internal(format!("设置字幕区域选择器位置失败: {error}"))
        })?;
    window
        .set_size(PhysicalSize::new(geometry.width, geometry.height))
        .map_err(|error| {
            BackendFailure::internal(format!("设置字幕区域选择器尺寸失败: {error}"))
        })?;
    Ok(())
}

fn create_selector_window(
    app: &tauri::AppHandle,
    geometry: platform::TargetGeometry,
) -> Result<(), BackendFailure> {
    trace_live(format_args!(
        "selector window dispatch requested bounds={}x{}+{},{}",
        geometry.width, geometry.height, geometry.x, geometry.y
    ));
    let app = app.clone();
    let window_app = app.clone();
    run_window_operation_on_main_thread(app, move || {
        trace_live(format_args!(
            "selector window operation entered Tauri main thread"
        ));
        close_window_now(&window_app, SELECTOR_LABEL);
        trace_live(format_args!("selector window builder started"));
        let window = WebviewWindowBuilder::new(
            &window_app,
            SELECTOR_LABEL,
            WebviewUrl::App("index.html".into()),
        )
        .title("选择实时字幕区域")
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .resizable(false)
        // Apply physical client bounds before showing the selector.
        .visible(false)
        .build()
        .map_err(|error| {
            trace_live(format_args!("selector window build failed: {error}"));
            BackendFailure::internal(format!("创建字幕区域选择器失败: {error}"))
        })?;
        set_selector_window_bounds(&window, geometry)?;
        trace_live(format_args!("selector native window created"));
        window.show().map_err(|error| {
            BackendFailure::internal(format!("显示字幕区域选择器失败: {error}"))
        })?;
        let _ = window.set_focus();
        trace_live(format_args!("selector window ready"));
        Ok(())
    })
}

fn position_overlay(
    app: &tauri::AppHandle,
    geometry: platform::TargetGeometry,
    settings: LiveOverlaySettings,
) {
    let app = app.clone();
    let window_app = app.clone();
    let _ = run_window_operation_on_main_thread(app, move || {
        let Some(window) = window_app.get_webview_window(OVERLAY_LABEL) else {
            return Ok(());
        };
        let (x, y, width, height) = overlay_bounds(geometry, settings);
        let _ = window.set_position(PhysicalPosition::new(x, y));
        let _ = window.set_size(PhysicalSize::new(width, height));
        Ok(())
    });
}

fn close_window(app: &tauri::AppHandle, label: &str) {
    let app = app.clone();
    let label = label.to_owned();
    let _ = run_window_operation_on_main_thread(app.clone(), move || {
        close_window_now(&app, &label);
        Ok(())
    });
}

fn close_window_now(app: &tauri::AppHandle, label: &str) {
    if let Some(window) = app.get_webview_window(label) {
        let _ = window.close();
    }
}

fn run_window_operation_on_main_thread<F>(
    app: tauri::AppHandle,
    operation: F,
) -> Result<(), BackendFailure>
where
    F: FnOnce() -> Result<(), BackendFailure> + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    app.run_on_main_thread(move || {
        let _ = sender.send(operation());
    })
    .map_err(|error| BackendFailure::internal(format!("调度实时窗口操作失败: {error}")))?;
    receiver
        .recv()
        .map_err(|error| BackendFailure::internal(format!("实时窗口主线程操作意外退出: {error}")))?
}

fn set_overlay_visible(app: &tauri::AppHandle, visible: bool) {
    let app = app.clone();
    let _ = run_window_operation_on_main_thread(app.clone(), move || {
        let Some(window) = app.get_webview_window(OVERLAY_LABEL) else {
            return Ok(());
        };
        if visible {
            let _ = window.show();
        } else {
            let _ = window.hide();
        }
        Ok(())
    });
}
async fn run_live_operation<T, F>(operation: F) -> Result<T, BackendError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BackendFailure> + Send + 'static,
{
    let result = tauri::async_runtime::spawn_blocking(operation)
        .await
        .map_err(|error| {
            BackendFailure::internal(format!("实时会话命令工作线程意外退出: {error}"))
        });
    result.and_then(|result| result).map_err(BackendError::from)
}

#[tauri::command]
pub(crate) async fn begin_live_selection(
    app: tauri::AppHandle,
    target_id: String,
    target_language: String,
    overlay_settings: LiveOverlaySettings,
    recognition_settings: LiveRecognitionSettings,
    translation_settings: LiveTranslationSettings,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    trace_live(format_args!(
        "begin_live_selection received target={target_id} language={target_language}"
    ));
    let manager = manager.inner().clone();
    run_live_operation(move || {
        manager.begin_selection(
            &app,
            target_id,
            target_language,
            overlay_settings,
            recognition_settings,
            translation_settings,
        )
    })
    .await
}

#[tauri::command]
pub(crate) async fn confirm_live_selection(
    app: tauri::AppHandle,
    session_id: String,
    roi: LiveRoi,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.confirm_selection(&app, &session_id, roi)).await
}

#[tauri::command]
pub(crate) async fn begin_live_roi_update(
    app: tauri::AppHandle,
    session_id: String,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.begin_roi_update(&app, &session_id)).await
}

#[tauri::command]
pub(crate) async fn cancel_live_selection(
    app: tauri::AppHandle,
    session_id: String,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.cancel_selection(&app, &session_id)).await
}

#[tauri::command]
pub(crate) async fn pause_live_session(
    app: tauri::AppHandle,
    session_id: String,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.set_paused(&app, &session_id, true)).await
}

#[tauri::command]
pub(crate) async fn resume_live_session(
    app: tauri::AppHandle,
    session_id: String,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.set_paused(&app, &session_id, false)).await
}

#[tauri::command]
pub(crate) async fn interrupt_live_translation(
    app: tauri::AppHandle,
    session_id: String,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.interrupt_translation(&app, &session_id)).await
}

#[tauri::command]
pub(crate) async fn stop_live_session(
    app: tauri::AppHandle,
    session_id: Option<String>,
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    let manager = manager.inner().clone();
    run_live_operation(move || manager.stop(&app, session_id.as_deref())).await
}

#[tauri::command]
pub(crate) fn list_capture_windows() -> Result<Vec<CaptureWindowInfo>, BackendError> {
    platform::list_target_windows().map_err(Into::into)
}

#[tauri::command]
pub(crate) fn get_live_session_status(
    manager: State<'_, LiveSessionManager>,
) -> Result<LiveSessionStatus, BackendError> {
    manager.collect_finished()?;
    manager.current_status().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::{
        await_capture_start_result,
        contracts::{LiveOverlayAttachment, LiveOverlayMode, LiveOverlaySettings, LiveRoi},
        key_trigger_timeout_reached, live_subtitle_region, live_translated_region_text,
        normalized_live_region_text, overlay_bounds, overlay_url_path,
        platform::TargetGeometry,
    };
    use crate::{
        backend::contracts::RegionRecord, models::hy::translation::build_translation_prompt,
    };
    use std::{
        sync::{atomic::AtomicBool, mpsc},
        time::Duration,
    };

    #[test]
    fn capture_start_wait_returns_the_worker_result() {
        let stop = AtomicBool::new(false);
        let (sender, receiver) = mpsc::sync_channel(1);
        sender
            .send(Ok("capture-ready"))
            .expect("send startup result");

        let result = await_capture_start_result(&stop, &receiver)
            .expect("receive startup result")
            .expect("capture worker result");

        assert_eq!(result, "capture-ready");
    }
    #[test]
    fn capture_start_wait_exits_when_the_session_stops() {
        let stop = AtomicBool::new(true);
        let (_sender, receiver) = mpsc::sync_channel::<Result<(), super::BackendFailure>>(1);

        assert!(
            await_capture_start_result(&stop, &receiver)
                .expect("stopped session is not a capture error")
                .is_none()
        );
    }

    #[test]
    fn key_trigger_forces_ocr_after_its_configured_timeout_with_a_frame() {
        let timeout = Duration::from_millis(700);
        assert!(!key_trigger_timeout_reached(
            true,
            Some(Duration::from_millis(699)),
            true,
            timeout,
        ));
        assert!(key_trigger_timeout_reached(
            true,
            Some(timeout),
            true,
            timeout,
        ));
        assert!(!key_trigger_timeout_reached(
            false,
            Some(timeout),
            true,
            timeout,
        ));
        assert!(!key_trigger_timeout_reached(
            true,
            Some(timeout),
            false,
            timeout,
        ));
    }

    #[test]
    fn overlay_bounds_attach_subtitle_to_each_requested_edge() {
        let geometry = TargetGeometry {
            x: 100,
            y: 200,
            width: 800,
            height: 600,
        };
        let settings = LiveOverlaySettings {
            offset: 25,
            ..LiveOverlaySettings::default()
        };

        assert_eq!(overlay_bounds(geometry, settings), (100, 825, 800, 168));
        assert_eq!(
            overlay_bounds(
                geometry,
                LiveOverlaySettings {
                    attachment: LiveOverlayAttachment::Top,
                    ..settings
                },
            ),
            (100, 7, 800, 168),
        );
        assert_eq!(
            overlay_bounds(
                geometry,
                LiveOverlaySettings {
                    attachment: LiveOverlayAttachment::Left,
                    ..settings
                },
            ),
            (-93, 200, 168, 600),
        );
        assert_eq!(
            overlay_bounds(
                geometry,
                LiveOverlaySettings {
                    attachment: LiveOverlayAttachment::Right,
                    ..settings
                },
            ),
            (925, 200, 168, 600),
        );
    }

    #[test]
    fn region_replace_overlay_matches_the_target_client_area() {
        let geometry = TargetGeometry {
            x: 100,
            y: 200,
            width: 800,
            height: 600,
        };
        let settings = LiveOverlaySettings {
            mode: LiveOverlayMode::RegionReplace,
            attachment: LiveOverlayAttachment::Left,
            offset: 2_048,
            show_source: false,
            show_region_boxes: false,
        };

        assert_eq!(overlay_bounds(geometry, settings), (100, 200, 800, 600));
    }

    #[test]
    fn overlay_url_carries_region_box_visibility() {
        let settings = LiveOverlaySettings {
            mode: LiveOverlayMode::RegionReplace,
            show_source: false,
            show_region_boxes: true,
            ..LiveOverlaySettings::default()
        };

        assert_eq!(
            overlay_url_path("live-42", settings),
            "index.html?liveSessionId=live-42&liveOverlayMode=region_replace&showSource=0&showRegionBoxes=1"
        );
    }
    #[test]
    fn subtitle_translation_preserves_all_ocr_regions() {
        let mut first =
            RegionRecord::untranslated(1, [[0, 0], [100, 0], [100, 20], [0, 20]], "Aden");
        first.translated_text = "阿登".to_owned();
        let mut second = RegionRecord::untranslated(
            2,
            [[0, 30], [300, 30], [300, 50], [0, 50]],
            "Passionate seems like a better word.",
        );
        second.translated_text = "激情似乎是个更好的词。".to_owned();

        assert_eq!(
            live_translated_region_text(&[first, second]),
            "阿登\n激情似乎是个更好的词。"
        );
    }
    #[test]
    fn live_subtitle_regions_use_distinct_client_space_vertical_bounds() {
        let roi = LiveRoi {
            x: 40,
            y: 50,
            width: 400,
            height: 180,
            client_width: 800,
            client_height: 600,
        };
        let first =
            RegionRecord::untranslated(1, [[10, 20], [210, 20], [210, 50], [10, 50]], "First line");
        let second = RegionRecord::untranslated(
            2,
            [[20, 80], [220, 80], [220, 110], [20, 110]],
            "Second line",
        );

        let first = live_subtitle_region(&first, roi).expect("first live subtitle region");
        let second = live_subtitle_region(&second, roi).expect("second live subtitle region");

        assert_eq!(first.bounds.left, 50);
        assert_eq!(first.bounds.top, 70);
        assert_eq!(first.bounds.width, 200);
        assert_eq!(first.bounds.height, 30);
        assert_eq!(second.bounds.top, 130);
        assert_eq!(second.bounds.top - first.bounds.top, 60);
    }

    #[test]
    fn subtitle_prompt_contains_all_ocr_regions_in_one_request() {
        let regions = vec![
            RegionRecord::untranslated(
                1,
                [[0, 0], [300, 0], [300, 20], [0, 20]],
                "Don't go causing any trouble. I'll tie",
            ),
            RegionRecord::untranslated(
                2,
                [[0, 30], [260, 30], [260, 50], [0, 50]],
                "you up real tight if you do.",
            ),
        ];

        let source_text = normalized_live_region_text(&regions);
        assert_eq!(
            build_translation_prompt(&source_text, "Chinese"),
            "Translate the following text into Chinese. Output only the translation: \
Don't go causing any trouble. I'll tie\n\
you up real tight if you do."
        );
    }
}
