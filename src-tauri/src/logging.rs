use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use tauri::Manager;
use time::format_description::well_known::Rfc3339;
use tracing_appender::{non_blocking::WorkerGuard, rolling};
use tracing_subscriber::{
    EnvFilter, fmt::time::OffsetTime, layer::SubscriberExt, util::SubscriberInitExt,
};

static TRACING_GUARDS: OnceLock<(WorkerGuard, WorkerGuard)> = OnceLock::new();

fn current_local_offset() -> Option<time::UtcOffset> {
    time::UtcOffset::current_local_offset().ok()
}

pub fn resolve_app_log_directory(app: &tauri::AppHandle) -> PathBuf {
    if let Ok(dir) = app.path().app_log_dir() {
        return dir;
    }
    if let Ok(dir) = app.path().app_data_dir() {
        return dir.join("logs");
    }
    if let Ok(dir) = app.path().app_config_dir() {
        return dir.join("logs");
    }
    // fallback to temp
    std::env::temp_dir().join("smodeltrans-logs")
}

pub fn prepare_log_directory(dir: &Path) -> Result<(), String> {
    if let Err(e) = fs::create_dir_all(dir) {
        return Err(format!("创建日志目录失败 {}: {}", dir.display(), e));
    }
    // B 方案：latest.log 先删后建，保证每次启动 latest 仅含本次会话
    let latest = dir.join("latest.log");
    if latest.exists() {
        let _ = fs::remove_file(&latest);
    }
    Ok(())
}

pub fn build_session_log_file_name() -> String {
    // 使用本地时间，若失败则回退 UTC
    let now = time::OffsetDateTime::now_local()
        .or_else(|_| Ok::<_, time::error::IndeterminateOffset>(time::OffsetDateTime::now_utc()))
        .unwrap_or_else(|_| time::OffsetDateTime::now_utc());
    let offset = current_local_offset().unwrap_or(time::UtcOffset::UTC);
    let local_now = now.to_offset(offset);
    let fmt = time::format_description::parse_borrowed::<2>("[year][month][day]-[hour][minute][second]")
        .unwrap_or_else(|_| time::format_description::parse_borrowed::<2>("[year]-[month]-[day]").unwrap());
    let ts = local_now.format(&fmt).unwrap_or_else(|_| "session".to_string());
    format!("smodeltrans-{ts}.log")
}

pub fn init_tracing(log_directory: PathBuf, session_file_name: String) {
    // 桥接 log crate
    let _ = tracing_log::LogTracer::init();

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let offset = current_local_offset().unwrap_or(time::UtcOffset::UTC);
    let timer = OffsetTime::new(offset, Rfc3339);

    // 控制台层：带颜色，使用 stdout
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_timer(timer.clone())
        .with_ansi(true)
        .with_writer(std::io::stdout);

    // 会话文件层：无颜色
    let session_appender = rolling::never(&log_directory, session_file_name);
    let (session_writer, session_guard) = tracing_appender::non_blocking(session_appender);
    let session_layer = tracing_subscriber::fmt::layer()
        .with_timer(timer.clone())
        .with_ansi(false)
        .with_writer(session_writer);

    // latest.log 层：无颜色，始终覆盖
    let latest_appender = rolling::never(&log_directory, "latest.log");
    let (latest_writer, latest_guard) = tracing_appender::non_blocking(latest_appender);
    let latest_layer = tracing_subscriber::fmt::layer()
        .with_timer(timer)
        .with_ansi(false)
        .with_writer(latest_writer);

    let result = tracing_subscriber::registry()
        .with(filter)
        .with(stdout_layer)
        .with(session_layer)
        .with(latest_layer)
        .try_init();

    match result {
        Ok(_) => {
            let _ = TRACING_GUARDS.set((session_guard, latest_guard));
            tracing::info!(
                target: "app::logging",
                dir = %log_directory.display(),
                "tracing 初始化完成"
            );
        }
        Err(e) => {
            // OnceLock 已初始化或重复 init，保留 guard 防止丢失
            let _ = TRACING_GUARDS.set((session_guard, latest_guard));
            eprintln!("tracing 初始化失败（可能重复初始化）: {e}");
        }
    }
}

#[tauri::command]
pub fn frontend_log(level: String, message: String) {
    match level.to_ascii_lowercase().as_str() {
        "trace" => tracing::trace!(target: "frontend", "{}", message),
        "debug" => tracing::debug!(target: "frontend", "{}", message),
        "info" => tracing::info!(target: "frontend", "{}", message),
        "warn" => tracing::warn!(target: "frontend", "{}", message),
        "error" => tracing::error!(target: "frontend", "{}", message),
        _ => tracing::info!(target: "frontend", "[{}] {}", level, message),
    }
}

#[tauri::command]
pub fn list_log_files(
    app: tauri::AppHandle,
) -> Result<Vec<String>, String> {
    let dir = resolve_app_log_directory(&app);
    let entries = fs::read_dir(&dir).map_err(|e| format!("读取日志目录失败: {e}"))?;
    let mut files: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.is_file() {
                p.file_name()?.to_str().map(|s| s.to_owned())
            } else {
                None
            }
        })
        .filter(|name| name.ends_with(".log"))
        .collect();
    files.sort();
    Ok(files)
}

#[tauri::command]
pub fn read_log_file(
    app: tauri::AppHandle,
    file_name: String,
    lines: Option<usize>,
) -> Result<String, String> {
    // 防止路径穿越
    if file_name.contains('/') || file_name.contains('\\') || file_name.contains("..") {
        return Err("非法文件名".into());
    }
    let dir = resolve_app_log_directory(&app);
    let path = dir.join(&file_name);
    let content = fs::read_to_string(&path).map_err(|e| format!("读取日志失败 {}: {e}", path.display()))?;
    if let Some(n) = lines {
        // 返回最后 N 行
        let all: Vec<&str> = content.lines().collect();
        let start = all.len().saturating_sub(n);
        Ok(all[start..].join("\n"))
    } else {
        Ok(content)
    }
}

#[tauri::command]
pub fn open_log_directory(app: tauri::AppHandle) -> Result<(), String> {
    let dir = resolve_app_log_directory(&app);
    prepare_log_directory(&dir)?;
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&dir)
            .spawn()
            .map_err(|e| format!("打开日志目录失败: {e}"))?;
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&dir)
            .spawn()
            .map_err(|e| format!("打开日志目录失败: {e}"))?;
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&dir)
            .spawn()
            .map_err(|e| format!("打开日志目录失败: {e}"))?;
        return Ok(());
    }
    #[allow(unreachable_code)]
    Err("当前平台不支持自动打开目录".into())
}
