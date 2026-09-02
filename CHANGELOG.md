# Changelog

所有显著变更将记录于此，格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.4.1] - 2026-09-02

### 新增

- **OpenAI 兼容历史**：环形存储 `OpenAiHistoryStore`（100 条）持久化 `POST /v1/chat/completions` 结果，新增 `get_openai_history` / `clear_openai_history` Tauri 指令及前端 `OpenAiRequestHistory.vue`（`naive-ui` + 主题变量 `var(--surface-*)`，`NScrollbar`，深色适配）
- **Hy-MT2 模板复用**：`openai_compat` 经 `TranslationPort::translate_text_with_supplemental` 透传 `system/developer` 为 `supplemental_prompt`，后端 `render_single_prompt` 按官方 Default / 自定义单模板（含 `{source_text}` 覆盖）统一渲染，`supplemental` 仅作 `Additional requirements`

### 变更

- **API 模型重命名**：`GET /v1/models` 与 `POST /v1/chat/completions` 模型标识由 `hy2-mt` 统一为 `hy-mt2`（`routes.rs` / `history.rs`），兼容 `hy-mt2:English` 后缀
- **日志降噪**：`ModelConfig::from_parts`、`Generation/Memory/Prompt validate`、`engine::load_translator` / `load_translator_with_memory` / `translate_text` 及 `adapter::translate_text called` 由 `INFO` 降为 `DEBUG`，`INFO` 仅保留 `adapter success` + `routes success`（每译 2 行），`debug` 可复现全链路

### 修复

- 移除 `types.rs` 中 `parse_language_from_instruction` 机械解析，`target_language()` 仅保留 `target_language/language` → `model:后缀` → `Translate to:` 前缀 → `default`，杜绝 `You ... to English` 与 `Chinese default` 冲突的误判
- 修复 `OpenAiRequestHistory.vue` 辣眼睛内联样式，改用 `var(--surface)/--surface-soft/--border` 等主题变量
- 修复 `ModelManagerPage.vue`、`model_config.rs` 残留 `页/分隔符` 文案

## [0.4.0] - 2026-09-01

### 新增

- **模型目录**：补齐 `Hy-MT2 1.8B/7B` 全量 `Q4_K_M/Q6_K/Q8_0` GGUF（`model_download.rs` / `model-download-provider.ts`），`Tencent-Hunyuan/Hy-MT2-*`，修复 7B 冗余输出
- **日志体系**：引入 `tracing` + `tracing-subscriber` 全链路，`BackendState::touch_activity`，`cargo check` 无 `candle-flash-attn` 重编

### 修复

- 移除不支持的 `jy` 小模型，修复字体导入/标注路径 `font_path` 持久化与防守式校验
- 统一提示词为单模板（`PromptConfig{ template }` 支持 `{source_text}` 覆盖），修复 `BackendStatus::configuration_error` `invalid` 设备显示

## [0.3.0] - 2026-09-01

### 新增

- **OpenAI 兼容服务**（`feat(openai)`）：新增独立文件夹 `src-tauri/src/openai_compat`，基于 `axum 0.7` 与 `TranslationPort` 解耦实现 OpenAI 兼容 HTTP 服务
  - `GET /v1/models` 仅返回 `hy-mt2` 单一模型（`routes.rs`）
  - `POST /v1/chat/completions` 支持非流式与 `text/event-stream` 伪流式，映射 `model/messages/stream/temperature/top_p/top_k/max_tokens/seed` 至 `GenerationConfig`，`target_language` 支持 `model` 后缀及显式字段回退
  - `GET /health` / `GET /v1/health` 免鉴权探活，支持 `live` 互斥（503）与单并发限流（429）
  - `OpenAiCompatConfig { enabled, host, port, apiKey }` 持久化于 `model-settings.json` 的 `openaiCompat` 段，`Bearer` 可选鉴权，`CORS` 随启用开关挂载，`port+1..+3` 自动重试

- **配置与生命周期**：`BackendSettings` 扩展 `openai_compat`，`lib.rs` 通过 `OpenAiServerHandle` 独立启停与热重启（`update_openai_config`），`touch_activity` 防空闲卸载

- **前端**：新增 `src/services/openai-compat.ts` 与 `src/components/OpenAiCompatCard.vue`，设置页集成开关、Host/端口、API Key、状态及 `Base URL` 复制

### 修复

- 修复 `commands.rs` 对 `AtomicU16` 误用 `try_lock` 导致编译失败，改为 `load(SeqCst)`
- 修复 `adapter.rs` 中 `GenerationConfig` 移动后二次 `is_some` 借用错误，改为 `has_override` 标记
- 修复 `SettingsPage` 与 `OpenAiCompatCard` 卡片半宽问题（`app.css` 2 列栅格需 `settings-card-wide`），段落 `max-width` 放开至 `none`，端口 `n-input-number` 加减按钮未对齐问题（补 `settings-number-field` 类以命中 `app.css:906-916` 居中样式）
- 清理 `openai_compat/mod.rs` 未使用导入及 `dead_code` 警告，`cargo check` 0 警告通过，`bun run build` 通过

### 文档

- 新增本 `CHANGELOG.md`，版本单一来源仍为 `package.json`，`scripts/sync-version.mjs` 同步 `Cargo.toml` 与 `tauri.conf.json`

## [0.2.0] - 2025-08-25

- 初始实时翻译桌面应用：窗口捕获（Graphics Capture）→ PP-OCR V5/V6 → Hy-MT2 1.8B 本地推理，支持实时字幕、文本翻译、OCR、OCR 翻译、模型管理与空闲卸载
- 详见 `README.md`、`docs/LIVE_OCR_PIPELINE.md` 及 `release/smodeltrans_0.2.0_x64-setup.exe`

---
