# Changelog

所有显著变更将记录于此，格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.3.0] - 2026-09-01

### 新增

- **OpenAI 兼容服务**（`feat(openai)`）：新增独立文件夹 `src-tauri/src/openai_compat`，基于 `axum 0.7` 与 `TranslationPort` 解耦实现 OpenAI 兼容 HTTP 服务
  - `GET /v1/models` 仅返回 `hy2-mt` 单一模型（`routes.rs`）
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

> 版本管理：`package.json` 为单一来源，`bun run sync:version` 同步 `src-tauri/Cargo.toml` 与 `src-tauri/tauri.conf.json`。
