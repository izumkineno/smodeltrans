//! 系统托盘 + 轻量模式开关。
//!
//! Tauri 2 核查结论（docs.rs `tauri 2.11.5` + `reference/config` + 源码 `search_code`）：
//! 官方没有“轻量模式”概念。本仓的轻量模式 = 产品层开关，持久化于
//! `model-settings.json#lightweightMode`（默认开）：
//! 开 = 主窗口关闭时直接销毁前端 webview（释放内存），仅托盘 + 后端服务常驻；
//! 关 = 普通模式，主窗口关闭即退出整个应用。
//! 托盘图标复用打包图标（`app.default_window_icon()`），不新增资源文件。
//!
//! 注意：只销毁 `main` 窗口。`quick-translation` 悬浮窗虽小但全局快捷键依赖它常在
//! （`quick_translation::trigger` 无窗口时静默跳过）；实时字幕/选择器窗口是会话功能
//! 窗口，会话期间不动。内存大头是主工作区 webview，销毁它即达到轻量目的。

use crate::backend::commands::{persist_backend_settings, BackendState};
use tauri::{
    menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, WebviewUrl, WebviewWindowBuilder,
};

const MAIN_WINDOW_LABEL: &str = "main";
const TRAY_ID: &str = "main";
const MENU_SHOW: &str = "tray-show-main";
const MENU_LIGHTWEIGHT: &str = "tray-lightweight-mode";
const MENU_QUIT: &str = "tray-quit";

// 与 tauri.conf.json 的 main 窗口声明保持一致，重建时复用同一配置。
const MAIN_TITLE: &str = "smodeltrans - 翻译工作区";
const MAIN_WIDTH: f64 = 1180.0;
const MAIN_HEIGHT: f64 = 760.0;
const MAIN_MIN_WIDTH: f64 = 720.0;
const MAIN_MIN_HEIGHT: f64 = 560.0;

pub(crate) fn setup(app: &mut tauri::App) {
    if let Err(e) = build(app) {
        eprintln!("创建系统托盘失败（不影响主窗口）: {e:#}");
    }
}

fn build(app: &mut tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, MENU_SHOW, "显示主窗口", true, None::<&str>)?;
    let lightweight = CheckMenuItem::with_id(
        app,
        MENU_LIGHTWEIGHT,
        "轻量模式（关闭前端窗口）",
        true,
        read_lightweight(app.handle()),
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, MENU_QUIT, "退出", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &show,
            &lightweight,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;
    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or_else(|| {
            tauri::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "缺少默认窗口图标，无法创建托盘图标",
            ))
        })?;

    let lightweight_item = lightweight.clone();
    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .tooltip("smodeltrans - 翻译工作区")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, event| match event.id.as_ref() {
            MENU_SHOW => ensure_main(app),
            MENU_LIGHTWEIGHT => {
                let next = !read_lightweight(app);
                write_lightweight(app, next);
                let _ = lightweight_item.set_checked(next);
                if next {
                    destroy_main(app);
                } else {
                    ensure_main(app);
                }
            }
            MENU_QUIT => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| match event {
            TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } => toggle_main(tray.app_handle()),
            TrayIconEvent::DoubleClick { .. } => ensure_main(tray.app_handle()),
            _ => {}
        })
        .build(app)?;
    Ok(())
}

/// 主窗口关闭行为：轻量开 = 销毁前端窗口（`destroy` 不经过 `CloseRequested`），
/// 仅托盘常驻；轻量关 = 直接退出应用。
pub(crate) fn handle_close_to_tray(window: &tauri::Window, event: &tauri::WindowEvent) {
    if window.label() != MAIN_WINDOW_LABEL {
        return;
    }
    if !matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
        return;
    }
    if !read_lightweight(window.app_handle()) {
        window.app_handle().exit(0);
        return;
    }
    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        api.prevent_close();
    }
    destroy_main(window.app_handle());
}

fn read_lightweight(app: &tauri::AppHandle) -> bool {
    let Some(state) = app.try_state::<BackendState>() else {
        return true;
    };
    let Ok(guard) = state.settings.lock() else {
        return true;
    };
    guard.as_ref().map(|s| s.lightweight_mode).unwrap_or(true)
}

fn write_lightweight(app: &tauri::AppHandle, next: bool) {
    let Some(state) = app.try_state::<BackendState>() else {
        return;
    };
    let Ok(mut guard) = state.settings.lock() else {
        return;
    };
    let Ok(settings) = guard.as_mut() else {
        return;
    };
    settings.lightweight_mode = next;
    if let Some(path) = state.config_path.as_deref() {
        if let Err(e) = persist_backend_settings(path, settings) {
            eprintln!("持久化轻量模式失败: {e:#}");
        }
    }
}

fn destroy_main(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        // destroy 不发射 CloseRequested，直接释放 webview；托盘与后端服务不受影响。
        if let Err(e) = window.destroy() {
            eprintln!("销毁主窗口失败: {e:#}");
        }
    }
}

/// 重建与 tauri.conf.json 同配置的主窗口（轻量模式下托盘唤起用）。
fn create_main(app: &tauri::AppHandle) -> tauri::Result<()> {
    WebviewWindowBuilder::new(app, MAIN_WINDOW_LABEL, WebviewUrl::App("index.html".into()))
        .title(MAIN_TITLE)
        .inner_size(MAIN_WIDTH, MAIN_HEIGHT)
        .min_inner_size(MAIN_MIN_WIDTH, MAIN_MIN_HEIGHT)
        .decorations(false)
        .build()?;
    Ok(())
}

/// 主窗口存在则显示，不存在则按配置重建。
fn ensure_main(app: &tauri::AppHandle) {
    match app.get_webview_window(MAIN_WINDOW_LABEL) {
        Some(window) => {
            let _ = window.show();
            let _ = window.set_focus();
        }
        None => {
            if let Err(e) = create_main(app) {
                eprintln!("重建主窗口失败: {e:#}");
            }
        }
    }
}

fn toggle_main(app: &tauri::AppHandle) {
    match app.get_webview_window(MAIN_WINDOW_LABEL) {
        Some(window) => match window.is_visible() {
            Ok(true) => {
                let _ = window.hide();
            }
            _ => ensure_main(app),
        },
        // 轻量常驻中（无主窗口）：左键直接重建唤起。
        None => ensure_main(app),
    }
}
