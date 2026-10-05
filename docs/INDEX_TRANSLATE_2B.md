# Index-Translate-2B 接入调研

> 约束：**仅 Candle 原生接入，不考虑 sidecar**（llama.cpp / ollama / vLLM 均不采用）。
> 结论先行：Candle 官方（rev 31f35b1）**无 Qwen3.5 实现**——`candle-transformers/src/models/` 仅有 `qwen2/3(+moe)` 与 `quantized_qwen2/3`，无 `qwen3_5`；`quantized_qwen3.rs` 为纯 GQA 注意力，无 linear-attention/Mamba 分支。社区 GGUF（如 `prithivMLmods/Qwen3.5-2B-f32-GGUF`）已用 `qwen3_5` arch 标签，llama.cpp 先行支持。必须**自研 loader**。

## 1. 模型速览

- 基于 **Qwen3.5**（`config.json: architectures=Qwen3_5ForConditionalGeneration`，24 层，hidden 2048，vocab 248320，`max_position_embeddings=262144`），150 语言文本翻译 + 指令遵循（术语硬约束 / 风格软约束）。
- 解码默认：贪心 `temperature=0`、`max_tokens=1024`、`enable_thinking=False`；prompt 为中文翻译指令模板，`--source auto` 时省略源语言名（见本地 `models/Index-Translate-2B/README.md:146-173`）。
- 权重形态：BF16 safetensors（`model.safetensors` 单文件，约 2B 参数，官方称部署起点 ~8GB 显存 + KV cache）；另有官方量化：**GGUF**（llama.cpp，`IndexTeam/Index-Translate-2B-GGUF`）、**FP8/FP4**（vLLM）（上游 README 模型下载表）。
- 官方推理：vLLM serve（`vllm serve IndexTeam/Index-Translate-2B --max-model-len 32768`）+ OpenAI 兼容 client（`translate.py`）；另有免费公网 API `https://index-translate.bilibili.com/v1/chat/completions`（兼容 OpenAI 规范，model=`Index-Translate-35B-A3B`）。
- 关键评测（2B）：FLORES COMET-22 0.8655 / WMT26 Judge 60.26 / instTrans 质量 0.5391 + IFscore 0.7569 / MEME 0.6443，全方位优于 Hy-MT2-1.8B（0.8522 / 49.35 / 0.3181 / 0.4932 / 0.3643）。

## 2. 本项目现状（ Consult code）

| 层 | 现状 | 文件 |
|---|---|---|
| 翻译推理 | Candle 原生 Rust 推理，只吃 **GGUF**（`discover_gguf_files` / `is_gguf`），Hy-MT2 专用 prompt 与分块流水线 | `src-tauri/src/backend/settings.rs:833`、`engine.rs:730 translate_text` |
| 可下载清单 | 仅 Hy-MT2（1.8B/7B GGUF）+ PP-OCR，写死 `repo_id` + `file_specs` | `src-tauri/src/backend/model_download.rs:140 downloadable_models()`、`src/services/model-download-provider.ts:64` |
| 下载源 | ModelScope resolve / HF mirror + 浏览器 UA 防 403 | `model_download.rs:15-21` |
| 对外 API | 自有 OpenAI 兼容服务 `127.0.0.1:11438`（`hy-mt2:Chinese`），可被 Chrome 插件调用 | `docs/OPENAI_COMPAT_API.md` |

> 前提纠正：Hy-MT2 **不是 Qwen 系**，GGUF `general.architecture="hunyuan-dense"`（`model.rs:762`），metadata 全是 `hunyuan-dense.*` 前缀。看着像 Qwen 只是时代特征：GQA + RoPE 后 Q/K RMSNorm（QK-Norm，`model.rs:427`）+ SwiGLU，这三件套 Qwen3/GLM/Hunyuan 都有，不代表可直连加载。
+
## 3. Hy 模块复用清单
+
**能复用（约 70% 基础设施，不用重写）**：
- GGUF 读取：`Content::read` + arch 校验模式（`model.rs:1129`）+ `dtype_hint` 未知量化报错（`:42`）
- 量化矩阵：`fuse_quantized_rows`/`QMatMul`（`:688`）、`load_norm_weight` 反量化 RMSNorm 权重（`:671`）
- 基础件：`SimpleRmsNorm`（`:123`）、`precompute_freqs_cis` RoPE 表（`:729`）、`TokenizerFromGguf`（`:1153`）
- 运行时：`generation.rs` 贪心/采样解码 + penalty、`session.rs` 会话驱动/KV 状态、`forward_attn` KV cache 按需 2 倍扩容（`:467-504`）、`settings.rs` GGUF 自动发现（`:834`）
+
**不能复用（必须新写）**：
- `LayerWeights` 整层：Qwen3.5 的 18 个 linear-attention 层没有 q/k/v/o 投影，Hy 的 `attention_qk/attention_wv/attention_wo` 结构对不上
- `from_gguf` metadata 解析（`:752`）：`hunyuan-dense.*` 前缀全换成 `qwen3_5.*`，外加 `layer_types` 路由 + `linear_*` 参数 + mRoPE（`mrope_section=[11,11,10]`）
- 输出门控：Qwen3.5 `attn_output_gate=true`（Hy 无此门）
- prompt 模板：Hy 翻译指令 vs Index instTrans `【源文】/【约束要求】` + `enable_thinking=False`
+
## 4. 接入方案（纯 Candle，分三步走）

### Step 1：GGUF 取证（下载完成后，~0.5 天）
- `gguf_dump` / `Content::read` 列出 `general.architecture=qwen35`、`qwen35.*` metadata、全部 tensor 名 + dtype + shape。
- 确认：arch 字符串、linear 层 tensor 命名（q/k/v/o? `linear_conv1d`?）、tokenizer（`tokenizer.ggml.tokens`，vocab 248320）、rope/mrope（`rope_theta=10000000`、`mrope_section=[11,11,10]`）、`enable_thinking=False` 的 chat template。
- 输出 tensor 命名对照表，决定 Step 2 的 `from_gguf` 映射。

### Step 2：自研 `src-tauri/src/models/index/` loader
- `IndexSession::new(path, device)` → `Content::read` → arch 校验 → `ModelWeights::from_gguf`；provider 在 `src-tauri/src/models/mod.rs` 注册。
- GGUF 张量维度由 Candle reader 按 GGUF 规范逆序还原为 `[out, in]`，投影直接加载为 `QMatMul`，避免二次转置与全量 F32 复制；输入/输出嵌入共享。
- DeltaNet 按 Transformers `Qwen3_5GatedDeltaNet`：Q/K/V 为 6144 fused 投影，z/a/b 为独立投影；本地 config 的 key/value heads 均为 16，head dim 均为 128。
- Full attention：Q projection 同时输出 Q 与 gate（4096），K/V 各 512；Q/K RMSNorm 在 RoPE 前，partial RoPE 维度 64，GQA 8q/2kv，attention 输出乘 `sigmoid(gate)`。
- 层路由按本地 `layer_types` 的 18 linear + 6 full；TokenizerFromGguf 后应用本地 `tokenizer_config.json` ChatML 模板与 `generation_config.json` 双 EOS（248044、248046）。
### Step 3：下载清单 + 验证（~0.5 天）
- `model_download.rs:downloadable_models()` + `model-download-provider.ts:MODELSCOPE_DOWNLOADABLE_MODELS` 加 `index-translate-2b-q4` 条目：`repo_id=IndexTeam/Index-Translate-2B-GGUF`，文件名以实际仓库为准（Q4_K_M 优先）。
- 验证：中→英短句 + JSON 格式保持 + 术语表各 1 例；与 `translate_cases.jsonl` 官方样例对齐；与 Hy-MT2-1.8B 对比。

### 风险
- linear-attn 算子正确性是最大风险：以官方 `Qwen3_5GatedDeltaNet` 公式核对；真实权重只在 CUDA 用小上下文验证，不运行 CPU 模型测试。
- `head_dim=256` + GQA 在 CUDA 的 KV cache 显存占用需实测，默认 `--max-model-len 32768` 起步（对齐官方 2B/9B serving 预设），非 262144 全量。
## 5. 参考链接

- 中文 README：https://github.com/bilibili/Index-Translate/blob/main/README_zh.md
- 文本推理：https://github.com/bilibili/Index-Translate/blob/main/inference/llm/README_zh.md ｜提示词：https://github.com/bilibili/Index-Translate/blob/main/docs/prompts.md
- HF：https://huggingface.co/IndexTeam/Index-Translate-2B ｜GGUF：https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF ｜ModelScope：https://modelscope.cn/models/IndexTeam/Index-Translate-2B
- Candle 参考实现：`candle-transformers/src/models/quantized_qwen3.rs`（量化 GQA 前向 + KV cache）｜`granitemoehybrid.rs`（hybrid 层路由骨架）｜example：`candle-examples/examples/quantized-qwen3/main.rs`
- DeltaNet 实现参考：transformers `modeling_qwen3_next.py:Qwen3NextGatedDeltaNet`｜vLLM `qwen_gdn_linear_attn.py`｜`llama-gguf 0.14 docs: model::deltanet`｜fla `chunk_gated_delta_rule`｜HF Qwen3.5 文档（3:1 hybrid 说明）
