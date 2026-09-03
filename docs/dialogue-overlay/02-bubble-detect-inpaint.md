# 02 气泡检测与原文擦除

> 一手来源：BallonsTranslator `README_EN.md`（Automation modules 章节）+ `dmMaze/comic-text-detector README.md`；manga-image-translator `README.md`（Render/Inpainter/Detector/OCR Options）；comic-translate `README.md`（How it works 章节）。

## 1. 检测：文本行 + 气泡框双输出

主流不只检文本行，还检气泡框，后者同时服务擦除和排版：

- BallonsTranslator：`comic-text-detector` 输出 bbox + 文本行 + 分割 mask（`dmMaze/comic-text-detector` 原话：`extract bounding-boxes, text lines and segmentation ... to help text-removal, recognition, lettering`）。训练数据约 13k 动漫/漫画图（Manga109-s、三方漫画库、合成数据各 1/3），DBNet 变体做行检测 + YOLOv5 做文本块检测。
- comic-translate：`ogkalu/comic-text-and-bubble-detector`，RT-DETR-v2，在 11k 漫画/网漫/欧漫图上训练，检测后做算法分割（`Algorithmic segmentation based on the boxes`）。
- manga-image-translator：检测器可切 `ctd` 增检出；`detection_size` 按分辨率调；`box_threshold` 过滤 OCR 误检的乱码。

对照本仓库：`src-tauri/src/models/ppocr/adapter.rs:detector_quads` 只输出文本 quad，无气泡框、无 mask，顺序=轮廓发现顺序。这是多气泡顺序错乱的根因之一。

## 2. 擦除：mask 扩张 + 生成式 inpaint

三家一致：先由检测 mask 定擦除区，再生成式重绘背景，而不是黑块覆盖：

| 方案 | 来源 | 说明 |
| --- | --- | --- |
| LaMA（含漫画微调） | BallonsTranslator（`All lama* are finetuned using LaMa`）、comic-translate（`dreMaz/AnimeMangaInpainting` 微调检查点，经 `Sanster/lama-cleaner` 实现）、上游 `advimman/lama` | 气泡白底/网点背景主力，质量最高，慢 |
| AOT-GAN | BallonsTranslator（`AOT is from manga-image-translator`）、comic-translate、manga-image-translator 内置 | 轻量，速度快，复杂背景弱于 LaMA |
| PatchMatch | BallonsTranslator（`vacancy/PyPatchMatch` + `dmMaze/PyPatchMatchInpaint` 魔改） | 传统算法，无模型开销，复杂图文易穿帮 |
| 手工修复 | BallonsTranslator（mask 编辑+仿制图章式 inpaint 画笔，右键擦除多余重绘；适配条漫极端长宽比） | 机器兜不住时人工补 |

关键参数（manga-image-translator 实战建议）：

- `mask_dilation_offset 10~30`：mask 必须比字稍大，包住描边，否则漏字边。
- `kernel_size`：卷积核管擦除面积，大视野少残留但背景糊；与 `inpainting_size`（高分辨率加大）配合。
- 漏字归因法：译文与原文一致→擦除漏字；不一致→检测/OCR 问题。

对照本仓库：`output.rs:78-82` 是 `draw_filled_rect_mut(inset, OVERLAY)` 黑底盖写，无 mask、无 inpaint。短期不动模型时这是可接受的降级，但文档需明确这是与主流差距最大的一环。

## 3. 分层导出（人工嵌字友好）

- manga-image-translator：输出 `xcf/psd/pdf` 经 GIMP 渲染，原图最低层 + inpaint 独立层 + 每个译文框独立文本层（层名存原文）。
- BallonsTranslator：页内撤销/重做（翻页清空）、Word 导入导出、PS 导出脚本（`scripts/export to photoshop`）。

对照本仓库：`TranslationOutput{annotated_png, markdown, text}`（`backend/contracts.rs`）+ 前端 `annotatedImageDataUrl` 预览/保存（`OcrTranslationPage.vue:saveAnnotatedImage`），无分层。后续若要人工校对，优先补“原文/译文/底图分层导出”而不是先上重模型。
