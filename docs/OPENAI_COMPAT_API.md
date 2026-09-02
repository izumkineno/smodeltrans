# OpenAI 兼容 API — 文本翻译与 Vision 图片 OCR-翻译

> `smodeltrans` 在 `127.0.0.1:11438` 暴露 OpenAI 兼容 HTTP 服务（`src-tauri/src/openai_compat/` 独立文件夹，`adapter.rs` 唯一触碰 `BackendState`，严格 `TranslationPort` 解耦）。首版文本翻译已在 `fa55fda` 前落地，本期 `fa55fda` 新增 Vision 图片 OCR-翻译队列回传，截图后续零改复用同一端口。

---

## 1. 服务与发现

| 路由 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | `{status:"ok", port, ocr_loaded, model_loaded}` |
| `/v1/models` | GET | `{data:[{id:"hy-mt2", object:"model"}]}` 401 需 `Authorization: Bearer <api_key>`（仅当配置 `openaiCompat.apiKey`） |
| `/v1/chat/completions` | POST | 文本翻译与 Vision 图片 OCR-翻译统一入口 |

`host/port/apiKey` 来自 `model-settings.json#openaiCompat {enabled, host:"127.0.0.1", port:11438, apiKey?}`，`src-tauri/src/lib.rs:68` 注入 `Arc<dyn TranslationPort>`。默认 `127.0.0.1:11438`，失败尝试 `port+1` 三次；`live_active==true` 时图片/文本均 `503 service_unavailable`。

---

## 2. 文本翻译（既有）

```json
POST /v1/chat/completions
{
  "model": "hy-mt2:Chinese",
  "messages": [{"role":"user","content":"Hello"}],
  "stream": false,
  "temperature": 0.7
}
```
`model` 后缀即 `target_language`，亦可用 `extra_body.target_language` / `language` / `Translate to X:` 前缀，缺省 `Chinese`。`temperature/top_p/top_k/max_tokens/seed` 透传 `GenerationConfig`（`routes.rs:536 build_generation_override`）。非流返回 `choices[0].message.content: string`，`stream:true` 返回 `text/event-stream` + `data: [DONE]`。

---

## 3. Vision 图片 OCR-翻译（`fa55fda` 新增）

### 3.1 请求

```json
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "hy-mt2:Chinese",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Translate to Chinese"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
    ]
  }]
}
```

* `content` 为 `MessageContent::Parts(Vec<ContentPart>)`（`types.rs:39`），`ContentPart::ImageUrl{image_url: ImageUrl{url}}`（`types.rs:61`）。`url` 接受 `data:image/png;base64,...` / `data:image/jpeg;base64,...` / `data:image/webp;base64,...`（`charset=utf-8;base64` 兼容，大小写不敏感）或裸 `base64`；`peel_data_url`（`types.rs:169`）剥离 `data:` 前缀后取 `,` 后纯 b64，`http://`/`https://`/`blob:` 等一律 `400 remote url not supported`。
* `types.rs:146 image_urls()` 按 `messages[]` 原序收集，`has_image()`/`image_count()` 供路由分支。
* 多图：同一 `content` 内放 `N` 个 `image_url`（`<=8`，`routes.rs:172` 超限 `400 too many images, max 8`），服务端单 `spawn_blocking` 内逐图串行 `port.translate_image("openai-image-001.png"..)` 竞争同一 `queue: Arc<Mutex<()>>` FIFO，文本按 `\n\n` 有序拼接，标注图取首图；队列等待期阻塞等待（`spawn_blocking` 语义）。
* `stream:true + image_url` 显式 `400 stream with image_url not supported`（SF-1），文本 `stream:true` 仍走 SSE。

### 3.2 流水线

`decode_image(base64, file_name, target_language)`（`backend/input.rs:88`，`MAX_BASE64_CHARS=14M / MAX_ENCODED_BYTES=10MiB / MAX_IMAGE_SIDE=8192 / MAX_IMAGE_PIXELS=33M / MAX_CANVAS_BYTES=128MiB / MAX_REGIONS=256`）→ `BackendEngine::translate(&DecodedImage, &CancellationToken)`（`engine.rs:391`：`recognize → translate_regions_with_progress → ImageOutput::render`）→ `TranslationOutput{ text, markdown, annotated_png, is_translated }`。`adapter.rs:367 translate_image` 复刻 `translate_text_with_supplemental` 的 `touch_activity` → live 入口 → `queue` → live 队列后 → `settings` 快照 → `generation` 临时替换（`out_res` 先捕获后恢复防毒化 MF-1）→ `lock_with_cancellation` → `BackendEngine::new` 按需 → live 第三检 → `engine.translate`。

`supplemental_prompt`（`messages` 中 `system/developer` 拼接）对图片分支暂忽略并 `debug!(supplemental_ignored=true)`（MF-4），预留未来 `engine.translate_with_supplemental`。

### 3.3 回传

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "hy-mt2:Chinese",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "中文翻译文本（多图为 \\n\\n 拼接，零区域为 \"\"）"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
      ]
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 8}
}
```

* `types.rs:225 MessageContentOut::Text(String) | Parts(Vec<ContentPartOut>) untagged`：文本请求仍 `content:"string"`，图片请求才 `content:[...]`，`Choice:175` 不变，新增 `ChatMessageOut::with_image` + `new_chat_response_with_image`（`types.rs:253,319`）。
* 第二段 `image_url.url` 即 `output.rs:38 ImageOutput::render` 的 `annotated_png` 单次 `BASE64.encode`，含红框+译文叠加，可直接 `<img src>`。
* 零区域仍 `200` 且双段齐全（`engine.rs:434` 空 `records` 仍编码原图 PNG）；多图失败即整体 `400/503`，无部分提交。

### 3.4 错误

| 场景 | 状态 | `error.type` | 触发 |
|------|------|--------------|------|
| 非 `data:` 远端 / 坏 `data:` / 空 url | 400 | `invalid_request_error` | `peel_data_url → Err("remote url not supported"/"malformed data URL") → BackendFailure::arguments` |
| `>14M base64 / >10MiB / >8192 / >33M像素` | 400 | `invalid_request_error` | `input.rs:111,120,142` 透传含 `imageBase64`/`dimensions` |
| `>8 张` | 400 | `invalid_request_error` | `too many images, max 8` |
| `live_active` | 503 | `service_unavailable` | `routes:216` 入口 + `adapter:97/119/241` 三处 |
| `stream:true + image` | 400 | `invalid_request_error` | `stream with image_url not supported` |
| OCR零区域 | 200 | — | `content[0].text==""` 且 `content[1].image_url` 仍在 |

`target_language` 优先级：`model:hy-mt2:English` 后缀 > `target_language` 字段 > `language` 字段 > `Translate to X:` 前缀 > 默认 `Chinese`（`types.rs:86 target_language()`）。

---

## 4. 调用示例

### curl 单图

```bash
curl -X POST http://127.0.0.1:11438/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hy-mt2:Chinese","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,'$(base64 -w0 shot.png)'"}}]}]}' | jq
# 断言 .choices[0].message.content[1].image_url.url | startswith("data:image/png;base64,")
```

### curl 多图（3张队列）

```bash
curl -X POST http://127.0.0.1:11438/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hy-mt2:English","messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'$(base64 -w0 a.png)'"}},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'$(base64 -w0 b.png)'"}},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'$(base64 -w0 c.png)'"}}
  ]}]}'
# text = t1 + "\n\n" + t2 + "\n\n" + t3，queue 日志 "queue acquired" 按 idx 顺序
```

### Python

```python
import base64, requests
b64 = base64.b64encode(open("shot.png","rb").read()).decode()
r = requests.post("http://127.0.0.1:11438/v1/chat/completions", json={
  "model":"hy-mt2:Chinese",
  "messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]
})
j = r.json()
text = j["choices"][0]["message"]["content"][0]["text"]                 # 纯文本 client 亦可取 j["choices"][0]["message"]["content"] 若为 string
img_data_url = j["choices"][0]["message"]["content"][1]["image_url"]["url"]  # <img src=img_data_url>
```

### JS / 截图接入（后续）

```js
// 任意截图源取 dataURL 后直接填 image_url.url，与单图同接口
const dataUrl = canvas.toDataURL("image/png") // 已含 data:image/png;base64,
const r = await fetch("http://127.0.0.1:11438/v1/chat/completions", {
  method:"POST", headers:{"Content-Type":"application/json"},
  body: JSON.stringify({model:"hy-mt2:Chinese", messages:[{role:"user", content:[{type:"image_url", image_url:{url:dataUrl}}]}]})
})
const j = await r.json()
// 后续截图模块直接复用同一端口：
// Rust 侧截图捕获 base64 后 `port.translate_image(b64, "screenshot-20250902-001.png", lang, supp, gen)`
```

---

## 5. 排障：Failed to fetch / 无法连接 http://127.0.0.1:11438/v1

> 常见于 `smodeltrans` 未启动、`openaiCompat.enabled=false`、错用 `http://127.0.0.1:11438/v1`（该路径不存在）或端口被占。服务仅在 Tauri 桌面端启动后监听，`bun run dev` 网页预览不含后端。

**1. 启动与配置**
* 启动桌面端：`bun run tauri dev`（或已安装 exe）。`设置 → OpenAI 兼容` 打开 `enabled`，确认 `host=127.0.0.1 port=11438` 后点保存；或直接改 `%APPDATA%/smodeltrans/model-settings.json` 的 `openaiCompat: {enabled:true, host:"127.0.0.1", port:11438}` 后重启。
* 端口递增：若 `11438` 被占，`server.rs` 自动试 `+1` 三次，实际以 `health.port` 为准。

**2. 正确路径**
* `http://127.0.0.1:11438/v1` 不是接口，单独请求必失败。健康检查：`GET http://127.0.0.1:11438/health` 或 `GET http://127.0.0.1:11438/v1/health`（`routes.rs:109` 双路由）。
* 模型列表：`GET http://127.0.0.1:11438/v1/models`
* 翻译/图片：`POST http://127.0.0.1:11438/v1/chat/completions`

**3. Windows 自检**
```powershell
netstat -ano | findstr 11438          # 应见 LISTENING + Tauri PID
curl -i http://127.0.0.1:11438/health  # 期望 200 + {"status":"ok","port":11438,"model_loaded":...}
curl -i http://127.0.0.1:11438/v1/models
# 浏览器 fetch 失败而 curl 通，多为 CORS/前端代理；Tauri 内 fetch不受限，外部网页需经 Tauri 侧中转而非直连
```
若 `curl /health` 不通：未启动 / `enabled:false` / 防火墙拦截（`wf.msc` 放行 `smodeltrans.exe`）/ 端口被占后落在 `11439/11440`（以 `health.port` 为准）。

**4. 模型未就绪**
* `health.model_loaded==false` 仍可联通但翻译为空转。需在 `模型管理` 下载 `Hy-MT2 Q4_K_M` + `PP-OCR V5/V6` 任一档，日志 `tracing` 目标 `openai_compat::routes` 可见 `model_loaded/hy/ocr`。
* `503 service_unavailable` 表示 `live_active==true`（实时翻译占用引擎），关掉实时翻译重试。

---

## 6. 可观测与约束

* 解耦：`openai_compat` 除 `adapter.rs` 头部 `use crate::backend::{commands::BackendState, contracts::TranslationOutput, failure::BackendFailure}` 外，`rg "BackendEngine|crate::models::hy" src-tauri/src/openai_compat --glob '!adapter.rs'` 必须 `0`；`cargo test openai_compat` 以 `mock::MockPort` 在无 CUDA 环境通过。
* 日志：图片分支 `tracing::info!(target:"openai_compat::routes", request_id, image_count, image_bytes_est, target_language, duration_ms)`，`history.push(OpenAiHistoryEntry::new("[N images] ..."))` 不存 base64（SF-4）。
* 显存：`BackendEngine` 以 `Mutex<Option<BackendEngine>>` 串行，`GpuExecutionPolicy` 已处理；`image_count<=8` 防 `spawn_blocking` 饥饿；零触 `Cargo.toml` / `candle-* rev 31f35b` / `profile.dev.package.candle-flash-attn`，`cargo check` 增量 <40s。
* 截图预留：`TranslationPort::translate_image(file_name: String, ...)` 以 `screenshot-*.png` 前缀区分来源，无需新增路由。

---

## 7. 规格与计划

* 访谈：`.omc/specs/deep-interview-img-ocr-translate.md` 142L 18.0% PASSED（5轮+Round0，4 active+1 deferred，6实体100%收敛）
* 计划：`.omc/plans/deep-interview-img-ocr-translate-plan.md` 826L **APPROVED**（Architect 4 MF + 6 SF 固化，Critic 10/10 AC 可测 PASS，`pending approval` 已执行于 `fa55fda`）
* 实现：`.omc/specs/deep-*.md` 关联 `openai_compat:types:146/225/169/253/319`、`routes:178/172/536/606`、`adapter:36/367/728`

