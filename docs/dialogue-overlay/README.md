# 对话贴图主流方案调研

> 来源：`web_search` 当时全部搜索源不可用（Startpage/DuckDuckGo/Ecosia/Google/Mojeek 均被机器人墙拦截，重试仍失败），改为直读三个开源项目一手 README/仓库文档。读取时间 2026-09-03。
>
> - BallonsTranslator（`dev` 分支 `README_EN.md`，via `xd://github file_read`）：`dmMaze/BallonsTranslator`
> - manga-image-translator（`main` 分支 `README.md`）：`zyddnys/manga-image-translator`
> - comic-translate（`main` 分支 `README.md`）：`ogkalu2/comic-translate`
> - comic-text-detector（`README.md`）：`dmMaze/comic-text-detector`（BallonsTranslator 文本检测的上游训练说明）

## 1. 结论先行

主流方案都是五阶段管线，与本仓库当前“检出即贴黑底白字”有三处本质差异：

| 本仓库现状 | 主流做法 | 差距 |
| --- | --- | --- |
| `src-tauri/src/output.rs render`：按检测框原位黑底盖写，单行、无换行，超宽截断 | 按**气泡框**排版：自动换行、自动缩小字号、居中，保留气泡底色 | 排版层缺失 |
| 无擦除：原文被黑块盖住 | 先 **inpaint 擦除原文**（LaMA/AOT-GAN/PatchMatch），再在干净底上排字 | 擦除层缺失 |
| `engine.rs:475-481` 静态图 `contextual=false`，逐框独立翻译；顺序=轮廓发现顺序 | **整页一次送 LLM**（带上下文/术语表/行级 1:1 约束）+ 气泡级阅读排序（日漫右→左） | 排序+上下文缺失 |

本仓库已有且可直接复用的思想：`translation.rs:718-742 build_contextual_translation_batch_prompt` 的上下文批译提示词；`LiveRegionReplaceOverlay.vue:50-158 wrapText/drawRegionText` 的逐字换行+居中（但只在实时浮层，静态贴图没用）。

- `01-pipeline.md`：三家五阶段管线对照（检测→OCR→擦除→翻译→排版），含模块选型表。
- `02-bubble-detect-inpaint.md`：气泡/文本检测与原文擦除（模型、mask 扩张、GIMP 分层导出）。
- `03-order-translate-render.md`：阅读排序、整页上下文翻译、气泡排版与人工后编辑。
- `04-smodeltrans-changes.md`：本次对齐实现清单、影响面与已知局限。

## 3. 与 smodeltrans 的对照落地（2026-09-03 已实现 1–2，详见 `04`）

1. ~~静态 `render` 引入换行+按框缩小字号~~ → 已实现：`draw_bubble_text`（换行+自适应字号+居中），另加亮底气泡白底黑字。
2. ~~气泡分组+阅读排序+打开 `contextual` 批译~~ → 已实现：`sort_static_reading_order` + `contextual=true`。
3. 长期：引入气泡级 mask + inpaint（LaMA/AOT）替代色块覆盖（见 `02`）。
