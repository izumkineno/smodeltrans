# 03 阅读排序、上下文翻译与气泡排版

> 一手来源：comic-translate `README.md`（Translation/Text Rendering）；manga-image-translator `README.md`（GPT Configuration Reference 全量 YAML、Future Plans 第 5 条）；BallonsTranslator `README_EN.md`（Typesetting + Context-aware LLM translation & Glossary 折叠节）。

## 1. 阅读排序：以气泡为单位

- comic-translate：渲染框取自气泡+文本双框（`Wrapped text in bounding boxes obtained from bubbles and text`），排序隐含在气泡检测顺序里。
- manga-image-translator：作者在 Future Plans 第 5 条自认现状不足——`The text rendering area is determined by the detected text, not the bubbles... cannot perfectly perform English typesetting. There is currently no good solution.` 即按文本行排版对英文这种长译文天然吃亏，气泡适配仍是开放问题；过渡方案是 `--manga2eng` 渲染器尝试按气泡而非文本行适配。
- BallonsTranslator：`Improved manga->English, English->Chinese typesetting (based on the extraction of balloon regions.)`，即英译排版 explicitly 依赖气泡区提取。

对照本仓库：静态图 `order` = 检测轮廓顺序（`adapter.rs:detector_quads/recognize_regions`），无气泡分组、无右→左/上→下排序。日漫单页多气泡时顺序错误会直接污染翻译上下文。实时路径 `scheduler.rs:plan_live_ocr_groups` 有分组/阅读顺序修正，静态图没有——这是最便宜的补齐点：先分气泡组、组内定序，再定 `order`。

## 2. 翻译：整页一次送 LLM，1:1 行约束 + 术语表

三家翻译对话时都不逐框独立调用，而是整页拼好一次送模型：

- comic-translate：`All LLMs are fed the entire page text`，可选附原图增上下文；支持 GPT-4.1/Claude-4.5/Gemini-2.5。
- manga-image-translator：`context-size`（整页上下文页数，仅 OpenAI 系生效）、`glossary`（OpenAI 系加载）、`pre/post-dict`、全套 GPT 模板（`chat_system_template/prompt_template/chat_sample/json_mode`）。其 CoT 模板对对话有硬约束：`NEVER combine multiple source lines into single translations`、`NEVER split 1 source line`、`ALWAYS maintain 1:1 Input-to-Output line ID correspondence`，行数对不上就删掉重来；损坏行原样输出，不允许半译。
- BallonsTranslator：`LLM Context=+history` 带前文已译页面（`Token budget` 默认 4096，新页优先，约 70% 上下文上限为宜）+ 可复用术语表（`source->translation` / TSV / JSON，大小写不敏感字面匹配，命中才发送；冲突/缺文件直接中止翻译）。

对照本仓库：`engine.rs:475-481` 静态图走 `contextual=false` 的独立批译（`translation.rs:694-715`，`translate each source_text independently`），跨气泡代词/断句必断。仓库里其实已有上下文批译提示词（`translation.rs:718-742 build_contextual_translation_batch_prompt`：`regions are one visual reading sequence... Use surrounding regions as context, but return one natural translation for every input region`），静态图暂未启用。建议启用顺序：先修排序→再开上下文批译→再补术语表/历史（实时路径已有 `memoryMaxTurns/memoryMaxTokens`，静态图可抄思想）。

## 3. 竖排/拟声词等对话特例

- BallonsTranslator：`YSGDetector`（`YSGforMTL/YSGYoloDetector`）专门过滤 CG/漫画拟声词；字体识别（`YuzuMarker.FontDetection`，置信度>60% 才写入 `_detected_font_name`）供导出嵌字用。
- manga-image-translator：`--font-path` 自选字体（如 `anime_ace_3.ttf`）；颜色曾试从 OCR 取，失败后退回 DPGMM 取色（自认效果不理想）。
- comic-translate：按目标语言选字体是用户须知（`Make sure the selected Font supports characters of the target language`）。

## 4. 排版：气泡框内换行+自适应字号+居中

- 共同点：自动字号/颜色，失败时可切全局设置（BallonsTranslator `Typesetting: decide by program / use global setting`）；长译文换行而非截断。
- comic-translate：气泡+文本双框包裹换行。
- manga-image-translator：译文太小看不清时调 `font_size_offset` 或切 `--manga2eng`。

对照本仓库：静态 `output.rs:83-99` 是单行 `draw_text_mut`（`clean_annotation` 已把 `\n` 压成空格），超宽只缩小到 `12px` 下限后截断；实时 `LiveRegionReplaceOverlay.vue:50-158` 反而有逐字 `wrapText`+居中+`min(fontSize,h*0.72,w/(len*0.55))` 自适应。静态贴图补换行+居中是纯前端/绘制层工作，不碰模型，优先级最高。
