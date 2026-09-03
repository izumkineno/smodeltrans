# 01 主流五阶段管线对照

> 一手来源见 `README.md`。三家都是“检测→OCR→擦除→翻译→排版”，对话按**气泡**而不是按 OCR 行处理。

## 1. 管线总览

```text
原图
 → 文本/气泡检测（boxes + mask）
 → OCR（逐块识别+颜色/字形信息）
 → Inpaint 擦除（mask 扩张后重绘背景）
 → 翻译（整页送 LLM，带上下文/术语表）
 → 排版渲染（按气泡框换行+自适应字号）
 → 人工后编辑（WYSIWYG/分层导出）
```

本仓库当前只有其中两段：PP-OCR 检测识别 + Hy-MT2 逐框翻译 + 黑底盖写（`src-tauri/src/backend/engine.rs:translate`、`src-tauri/src/output.rs:render`），缺中间擦除与气泡排版。

## 2. 三家选型对照

| 阶段 | BallonsTranslator（`dmMaze/BallonsTranslator`，`README_EN.md`） | manga-image-translator（`zyddnys/manga-image-translator`，`README.md`） | comic-translate（`ogkalu2/comic-translate`，`README.md`） |
| --- | --- | --- | --- |
| 检测 | 自研 `comic-text-detector`（英日）+ 云端团子 OCR 可选 + `YSGDetector` 过滤拟声词 | 默认检测器 + `ctd` 可选（`{"detector":{"detector":"ctd"}}` 增加检出）；`detection_size`/`box_threshold` 可调 | `bubble-and-text-detector`（`ogkalu/comic-text-and-bubble-detector`，RT-DETR-v2，11k 漫画/网漫/欧漫图训练）+ 算法分割 |
| OCR | mit 系（英/日/韩）+ `manga_ocr`（日漫）+ PaddleOCR（含 `PaddleOCRVLManga` 日漫微调）+ OneOCR 可选 | 48px 识别（日/韩推荐）；`mask_dilation_offset 10~30`、`kernel_size`、`box_threshold` 可调 | 默认日语 `manga-ocr`、韩语 `Pororo`、其余 `PPOCRv5`；可选 Gemini 2.0 Flash / Azure Vision |
| 擦除 | AOT（来自 manga-image-translator）+ LaMA 微调系列 + PatchMatch（`dmMaze/PyPatchMatchInpaint` 魔改版）；支持 mask 编辑/仿制图章式修复 | `lama_large` 推荐；`inpainting_size`（高分辨率加大防漏字）、`kernel_size`（视野与残留 tradeoff） | `dreMaz/AnimeMangaInpainting` 微调的 LaMA（经 `lama-cleaner` 实现）+ zyddnys 的 AOT-GAN |
| 翻译 | `doc/modules/translators.md` 列表；`LLMTranslator` 支持历史上下文+可复用术语表（详见 03） | Sugoi（日→英推荐）、Sakura（日→中）、OpenAI/Gemini/DeepSeek/Groq/自建 OpenAI 兼容；`context-size` 整页上下文、`glossary`、`pre/post-dict`、GPT 提示词模板全量可配 | GPT-4.1 / Claude-4.5 / Gemini-2.5；整页文本一次送模型，可选附原图增上下文 |
| 排版 | 按原文格式估计排版；日→英/英→中基于**气泡区提取**优化；字体/字号/颜色自动，`Typesetting` 可切“程序决定/全局设置” | 按检测文本区排版（作者自认弱于 Adobe 引擎）；`--manga2eng` 渲染器尝试按气泡而非文本行适配；`font_size_offset`、字体路径可配；`xcf/psd/pdf` 经 GIMP 分层渲染 | 按气泡+文本框双框包裹换行渲染 |
| 后编辑 | WYSIWYG 富文本、样式预设、文本特效/变换、全局查找替换、Word 导入导出、页间撤销 | `--prep-manual` 输出镂空+重绘底图供人工嵌字；`--save-text/--load-text` 文本先行后排版 | Manual Mode：自动跑完后可撤销纠正（漏检/错识/擦除残留），Viewer 内边读边翻 |

## 3. 可直接借鉴的配置思想

- `detection_size` 按分辨率调（低分辨率调低防漏句，高分辨率调高防误检）；`upscale_ratio 2` 救小字。
- mask 一定扩张（`mask_dilation_offset 10~30`），否则描边残留；高分辨率同步加大 `inpainting_size`。
- 渲染字小到不可读时优先 `font_size_offset` 或切气泡适配渲染器，而不是硬缩到框内。
