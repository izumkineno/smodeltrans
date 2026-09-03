# 04 smodeltrans 对齐实现（2026-09-03）

> 本次修改对照 `01–03` 的主流方案，用无模型开销的方式补齐三处差距。未引入气泡检测模型与生成式 inpaint（重模型、高成本，列为长期项）。

## 1. 改动清单

| 主流做法 | 本仓库修改 | 位置 |
| --- | --- | --- |
| 按气泡阅读排序（上→下、左→右行容差） | 静态翻译前先排序：行容差=中位框高中位值一半，同行内从左到右 | `src-tauri/src/backend/engine.rs:sort_static_reading_order` |
| 同气泡碎片合并为一个翻译单元 | 同行小间隙碎片合并为气泡行（垂直重叠≥45%或中心偏移≤45%，水平间隙≤字宽×1.65且≤行高×2，阈值与实时 `scheduler.rs` 同款），CJK 感知拼接（无空格），`order` 重排 1..N | 同上 + `static_fragments_belong_together`、`join_static_fragment_parts` |
| 整页上下文送译、行级 1:1 | 静态图 `contextual: false → true`，复用已有 `build_contextual_translation_batch_prompt`（外层框作上下文、每框独立返回） | `engine.rs:translate` 调用点 |
| 按气泡框换行+自适应字号+居中 | `draw_bubble_text`：显式换行保留，CJK 任意断、拉丁优先空格断；字号初值沿用 0.72×框高，按总高收敛（下限 12px），块级居中；字宽估算 CJK/全角按 1.0（旧 0.60 低估致溢出） | `src-tauri/src/output.rs:wrap_annotation`、`draw_bubble_text`、`char_width_ratio` |
| 保留气泡底色（白底黑字） | 贴字前沿框内圈采样平均亮度（权重与实时签名一致 77/150/29），亮底用白底墨字、暗底沿用黑底白字；`covers()` 字体检查豁免空白符（含保留的 `\n`） | `output.rs:bubble_style`、`covers`、`clean_annotation` |
| `render_ocr` 同步 | OCR 标注图复用同一排版绘制（此前同样单行截断） | `output.rs:render_ocr` |

## 2. 影响面

- `translate_image`（Tauri 命令）与 OpenAI 兼容 `translate_image` 共用 `BackendEngine::translate`，一次修改两路生效。
- 对外契约不变：`TranslationResponse{markdown, annotated_image_data_url, text, provider_label, is_translated, duration_ms}` 字段零增删；`text`/`markdown` 行数可能因合并变少（气泡级，更符合阅读）；OCR 路径 `OcrRegionResponse` 字段不变，框变为气泡并框、文本保留 `\n` 段落。
- 实时检测同样经过排序+合并（`live/mod.rs:1908` 经 `engine.recognize_regions` 取数），其自有分组/去重在其之上再跑一次，单测因直调构造不受影响；浮层绘制零触碰。
- 红框描边保留（调试可视性契约不变）。
## 3. 已知局限（与主流仍有差距）

1. 日漫右→左页级排序未做（上游 manga-image-translator 同样列为开放问题）。
2. 无真 inpaint：白底气泡用不透明白底重绘近似，复杂纹理背景仍是色块覆盖。
3. 未做跨页历史/术语表（BallonsTranslator 的 LLM Context + Glossary 思想，实时路径的记忆可作后续参照）。
4. 未执行 `cargo build/check/test`（仓库 `AGENT.md` 禁止触发 `candle-flash-attn` 重编）；校验方式为逐 hunk 重读 + 编辑器解析检查，建议在已有机模环境手动跑一次 `translate_image` 回归。

## 4. 追补：气泡级纵向合并（碎片化投诉后）

> 原因：首版只合并同行碎片；对话框内多行纵向堆叠仍是多个 region，逐行送译必然碎片化。

- `sort_static_reading_order` 加 Phase 3：纵向同气泡行合并（行间隙≤1.0×行高、水平交叠≥40%最小框宽、中心偏移≤1框宽），行间用 `\n` 保留段落，排版按段换行，一次送译。
- 横向合并抽取 `absorb_region` 复用（并框+CJK 拼接/换行拼接+置信度取小+字框拼接重排）。
- 竖排栏：高≥2倍宽视为竖排栏，同带内从右到左排序、邻接方向反转 joining（CJK 拼接本身无空格）。
- `recognize_regions`（纯 OCR 路径）同样走 `sort_static_reading_order`：前端选词/标注图按 `order` 展示，同等受益。
- 取舍：纵向堆叠的非对话文本（如游戏 UI 菜单列表）也可能被并为一个气泡（翻译仍连贯，贴图变为一整框）。无气泡检测模型时无法完美区分，对话场景优先合并。

## 5. 纯 GPU 实验（速度优化）

> 模型侧纯 GPU；合成侧保持 CPU（见下）。

- `BackendEngine::new`：`device_kind == Cuda` 时强制 `gpu_resident`，PP-OCR 与 Hy 同时驻显存，消除此前 Balanced/Constrained 档每次 `translate_image` 的互踢重载（`load_ocr` 踢 translator、`load_translator` 踢 OCR）。`for_memory` 档位仅记日志（阈值：常驻需 8GB 总显存/4GB 空闲，`engine.rs:17-18`）。
- 风险：小显存可能 OOM 直接报错而非变慢；回退即恢复此前档位逻辑。`device=cpu` 用户显式选项保留，不做硬拦截（UI 明确 CPU 仅用于状态检查，`ModelManagerPage.vue:82-85`）。
- 审计结论：推理链无暗 CPU 回退——`models/` 内 `Device::Cpu` 全是单测；`device=cuda` 时检测/识别/翻译/采样（`select_token` 全 device 内算子，每 token 仅一次 u32 D2H）都在 GPU；`create_device` 无 CUDA 时直接报错。分词、切框、拼图为像素侧 CPU 活。
- 合成图片未上 GPU：`output.rs` 基于 `image`/`imageproc`，依赖树无 wgpu/vulkan/cudarc；文字光栅+混合上 GPU 需新子系统，相对分钟级推理占比 <5%，暂不做。如需，另立 wgpu 任务。
- 验证：日志搜 `pure-GPU` 看显存与档位；`load_ocr`/`load_translator_with_memory` 应只在首次出现大 `duration_ms`，后续调用走 `already loaded`。
