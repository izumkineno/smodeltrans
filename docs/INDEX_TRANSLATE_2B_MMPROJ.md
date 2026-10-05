# Index-Translate-2B mmproj 视觉接入调研

> 约束：**仅 Candle 原生接入，不考虑 sidecar**（沿用 `docs/INDEX_TRANSLATE_2B.md:3`）。
> 结论先行：mmproj 是官方原配视觉补充文件（ViT 24 层 + 两层 MLP projector，自测解剖见 §2）；llama.cpp 侧经 `libmtmd` 把图片编码为 embeddings 后替换 prompt 中 `<image>` 槽位；本仓库无视觉代码可复用（§4），Candle 侧需手写 ViT。**动手前先跑 §5 的零成本实验探 prompt，否则不写代码**。

## 1. 官方口径

- `IndexTeam/Index-Translate-2B-GGUF` 文件清单含 `mmproj-Q8_0`（0.36GB）/ `mmproj-f16`（0.67GB），标注为 `multi-modal projector supplement`；原文：*they are only needed for image-input use with `llama-mtmd-cli`. For text-only translation, the plain GGUF files suffice.*
- 2B 本体即 `Qwen3_5ForConditionalGeneration`，**自带 24 层 vision tower**（第三方架构拆解，slm.expert），mmproj 与主 GGUF 同源，不存在跨模型混配问题。
- 注意：官方只公布了**文本** prompt 格式（中文翻译指令模板，`temperature=0` + `enable_thinking: false`），**未公布图像输入的 prompt/chat-template 占位符写法**——这是接入前最大的未知数，需实验探明，不许猜。

## 2. 本地文件解剖（实测，非推测）

`models/Index-Translate-2B.mmproj-Q8_0.gguf`：`general.architecture=clip`，`general.type=mmproj`，331M 参数，298 tensors。

| 部件 | tensor 前缀 | 规格 |
|---|---|---|
| patch_embd | `v.patch_embd.*` | conv 16×16（`weight` + `weight.1` 双份，Q8_0 常见）；768px / patch 16 → **2304 patches/图** |
| 位置编码 | `v.position_embd.weight` | `(1024, 2304)` |
| 视觉塔 | `v.blk.0..23.*` | ViT：hidden 1024，24 层，16 头，`ln1/ln2/post_ln` 均为**带 bias 的 LayerNorm**（非 RMSNorm）；`attn_qkv` fused 3072，`ffn_up/down` 1024↔4096 |
| projector | `mm.0` / `mm.2` | 两层 MLP：1024→4096→**2048**（输出 dim = 文本 hidden 2048，直接对接） |
| 量化 | 全线性层 type 8（Q8_0），norm/emb F32 | 与主模型 `from_qtensor` 保留量化策略一致 |

KV（以 `clip.vision.*` 为准）：`image_size=768`，`patch_size=16`，`embedding_length=1024`，`feed_forward_length=4096`，`block_count=24`，`attention.head_count=16`，`projection_dim=2048`，`has_vision_encoder=True`。

代价预警：**一张图 = 2304 个 image tokens**（无 patch merger，不降采样）。按当前 prefill 速度，TTFT 约涨 ~1s；decode 不受影响（prefill 后即普通 KV）。

## 3. llama.cpp 接线原理（libmtmd）

来源：`llama.cpp: tools/mtmd/README.md`、`docs/multimodal.md`、`tools/mtmd/mtmd-cli.cpp`。

- 需要两个文件：标准语言模型 + mmproj；`--mmproj` 指定，默认 GPU offload（`--no-mmproj-offload` 可关）。
- 流程：图片 → mmproj 编码为 embeddings → **替换 prompt 里 `mtmd_default_marker()`（即 `<image>`）所在位置** → 文本模型照常解码。视觉信息是 2304 个连续 embedding 槽位拼进文本序列，无旁路。
- 工具：`llama-mtmd-cli`（实验性，仅调试）、`llama-cli`、`llama-server`（OpenAI `/chat/completions` 兼容）。
- 跨模型混用必炸：`mtmd_init_from_file` 会校验 `n_embd`（如 text 2816 vs mmproj 1536 即报错），变相证明 projector 输出 dim 必须等于文本 hidden。

## 4. 本项目现状

- **无视觉代码**：`src-tauri/src` 内 `clip|siglip|mmproj|pixel_values` 命中均为 PP-OCR backbone（`models/ppocr/v5|v6/backbone.rs`，OCR 检测器）与注释措辞，与 ViT 无关，不可复用。
- 可复用基础设施：GGUF 读取 + `load_qmat/from_qtensor` 量化保留（`models/index/model.rs`）、GPU 常驻/F16 链路（`models/index/session.rs`）。
- 必须新写：24 层 ViT 前向（LN 版，注意 `qkv` fused + bias 全套）、图片预处理（768 resize + 归一化 mean/std，**参数未知，待 §5 实验或官方 preprocessor_config 补**）、`embed_ids` 后 embedding 拼接（占位符位置协议同样待实验）。

## 5. 下一步：零成本实验（先探后写）

```sh
llama-mtmd-cli -m models/Index-Translate-2B.Q4_K_M.gguf --mmproj models/Index-Translate-2B.mmproj-Q8_0.gguf --image <带中文字的图片> -p "<image>请将图片中的文字翻译为英文"
```

测三组：纯 OCR 式翻译 / 指令式翻译 / 无指令。看输出质量再决策：若可用 → 走 deep-interview + plan 进入 candle 原生实现；若不可用（翻译专用模型对视觉指令遵循差）→ 归档，视觉 OCR 翻译维持现有 PP-OCR 两段式。

产品契合度备注：视觉版 Index = “图里文字直翻”，对位本项目的 OCR 翻译场景；但 2304 tokens/图的 prefill 代价 + 未知的 prompt 协议是两个前置门槛。

## 6. 参考链接

- 2B-GGUF（含 mmproj 说明）：https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF
- 9B-GGUF（同构说明）：https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF
- libmtmd：https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md ｜ multimodal：https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md ｜ mtmd-cli 源码：https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-cli.cpp
- 论文：https://arxiv.org/abs/2609.40181 ｜ 官方 repo：https://github.com/bilibili/Index-Translate ｜ 主模型卡：https://huggingface.co/IndexTeam/Index-Translate-2B
