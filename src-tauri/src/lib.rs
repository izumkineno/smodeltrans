mod backend;
mod logging;
mod model_config;
mod model_support;
mod models;
mod openai_compat;
mod output;
mod quick_translation;
mod selection;
use std::sync::Arc;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build());

    #[cfg(debug_assertions)]
    {
        builder = builder.plugin(tauri_plugin_mcp_bridge::init());
    }

    builder
        .setup(|app| {
            // tracing B 方案：尽早初始化，双文件落盘（session + latest）
            let log_dir = logging::resolve_app_log_directory(app.handle());
            if let Err(e) = logging::prepare_log_directory(&log_dir) {
                eprintln!("准备日志目录失败: {e}");
            } else {
                let session_name = logging::build_session_log_file_name();
                logging::init_tracing(log_dir.clone(), session_name);
                tracing::info!(
                    target: "app::startup",
                    version = env!("CARGO_PKG_VERSION"),
                    log_dir = %log_dir.display(),
                    "应用启动"
                );
            }

            let resource_root = app.path().resource_dir().ok();
            let config_path = app
                .path()
                .app_config_dir()
                .ok()
                .map(|directory| directory.join("model-settings.json"));
            let state = backend::BackendState::new_with_resource_root_and_config(
                resource_root,
                config_path,
            );
            state.start_idle_monitor();
            let live_manager = backend::live::LiveSessionManager::new(state.clone());

            // OpenAI 兼容服务：独立生命周期，严格通过 TranslationPort 解耦
            let openai_initial = state
                .settings
                .lock()
                .ok()
                .and_then(|guard| guard.as_ref().ok().map(|s| s.openai_compat.clone()))
                .unwrap_or_default();
            let openai_handle = openai_compat::server::OpenAiServerHandle::new(openai_initial.clone());
            let openai_handle_for_spawn = openai_handle.clone();
            let state_for_spawn = state.clone();
            if openai_initial.enabled {
                tauri::async_runtime::spawn(async move {
                    let port: Arc<dyn openai_compat::adapter::TranslationPort> =
                        Arc::new(openai_compat::adapter::BackendStateAdapter::new(state_for_spawn));
                    if let Err(e) = openai_handle_for_spawn.start(port).await {
                        tracing::error!(target: "openai_compat", error = %e, "auto-start 失败");
                    } else {
                        tracing::info!(target: "openai_compat", "auto-start 成功");
                    }
                });
            }

            app.manage(state);
            app.manage(live_manager);
            app.manage(openai_handle);
            quick_translation::setup(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            logging::frontend_log,
            logging::list_log_files,
            logging::read_log_file,
            logging::open_log_directory,
            quick_translation::configure_quick_translation,
            backend::commands::get_backend_status,
            backend::commands::get_model_runtime_status,
            backend::commands::control_model,
            backend::commands::update_backend_settings,
            backend::commands::cancel_translation,
            backend::commands::list_model_catalog,
            backend::commands::save_model_catalog,
            backend::commands::translate_image,
            backend::commands::translate_text,
            backend::commands::ocr_image,
            backend::model_download::list_downloadable_models,
            backend::model_download::get_model_download_status,
            backend::model_download::start_model_download,
            backend::model_download::cancel_model_download,
            backend::model_download::list_downloaded_models,
            backend::model_download::get_downloaded_model_paths,
            backend::model_download::activate_downloaded_model,
            backend::model_download::delete_downloaded_model,
            backend::live::list_capture_windows,
            backend::live::start_live_session,
            backend::live::update_live_overlay_layout,
            backend::live::begin_live_overlay_drag,
            backend::live::begin_live_overlay_resize,
            backend::live::finish_live_overlay_resize,
            backend::live::update_live_overlay_position,
            backend::live::confirm_live_selection,
            backend::live::begin_live_roi_update,
            backend::live::cancel_live_selection,
            backend::live::get_live_session_status,
            backend::live::get_live_subtitle,
            backend::live::pause_live_session,
            backend::live::resume_live_session,
            backend::live::interrupt_live_translation,
            backend::live::stop_live_session,
            openai_compat::commands::get_openai_status,
            openai_compat::commands::update_openai_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running smodeltrans application");
}
