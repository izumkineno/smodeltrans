# Index-Trim：Index-Translate-2B 显存与速度（阶段一）

> Q4 权重显存 3.89G→约1.3G，无损消除单 token 前向开销，e2e 加基线计时

**Status**: Active
**Created**: 2026-10-05
**Owner**: pending approval（未批准执行前不得开工）

## Goal
RTX 4070 SUPER 上 Q4 模型整卡显存约 6G→约 2.5G；调用速度完成无损优化并打印基线；翻译质量不回归。

## Context
- 根因（已取证）：`index/model.rs::load_qmat` 全量反量化→F16（3.89G）；`session.rs` 每 token 每层约 73 次 `to_vec1` D2H 同步 + 逐 token prefill。
- 标准权重 Q4 为主（恢复文件名后验收），f16 备用；`from_qtensor` 对 f16 不省显存。
- 详见 `.omc/specs/deep-interview-index-perf.md`（ambiguity 18%）。

## Constraints
- AGENTS.md：严禁触发 candle-flash-attn 重编；只改 `models/index` 与测试。
- 阶段一仅无损手段；改计算路径需另立计划。
- Q4 三例质量断言保持全过。

## Tasks

| # | Task | Agent | Priority | Status | Dependencies |
|---|------|-------|----------|--------|--------------|
| 1 | 恢复 Q4 文件名（`.gguf1`→`.gguf`），确认 e2e 能定位权重 | executor | 1 | DONE | — |
| 2 | e2e 加基线计时（prefill/解码耗时、tok/s），质量断言不动 | executor | 1 | DONE | 1 |
| 3 | `load_qmat` 改 `from_qtensor`，人工看卡确认整卡约 2.5G | executor | 2 | DONE | 1 |
| 4 | norm 权重缓存消除每 token `to_vec1` 同步，记录前后对比 | executor | 2 | DONE | 2 |
| 5 | 固化显存断言进 e2e，更新 docs | executor | 3 | TODO | 3, 4 |
| 6 | P2b：hidden 常驻 GPU（层间残差/norm/FFN 全 tensor 化，仅 DeltaNet/GQA/argmax 过 CPU；量化统一喂 F16） | executor | 2 | DONE | 2 |

## Done When
- [ ] Q4 整卡约 2.5G（人工确认）
- [ ] e2e 三例全过，计时基线已打印
- [ ] `git diff --check` 通过，未触发禁令命令

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-05 | Q4 为主、f16 备用 | from_qtensor 对 f16 不省显存 |
| 2026-10-05 | 分阶段，先无损；速度数字基线后定 | 手段边界访谈决策 |
| 2026-10-05 | 显存先人工看卡、稳定后固化 | 测试内读显存不稳定，人工先行 |

## Progress Notes

- [2026-10-05] Plan created（deep-interview→plan 共识细化，pending approval）
- [2026-10-05] Autopilot approved；任务 1-4 代码完成（静态验证通过），待用户侧 CUDA e2e + 人工看卡；任务 5 待稳定后固化
- [2026-10-05] P2b（任务 6）代码完成：debug 下 0.22s/步主因是每步数百次 H2D/D2H，改为 GPU 常驻后每 linear 层 1 次 D2H+1 次 H2D、每 full 层同量；待用户 `--release` 跑 e2e 验证质量与计时
- [2026-10-05] P2c 代码完成：hidden 全程 F16（消约 300 cast/步，Hy 同仓契约）+ prefill 单 batch 前向（投影走 mmq，递推仍 CPU 逐 step）；`rms_norm_gpu` 改按行归一（batch 正确性）；待用户跑 e2e 验证质量与新计时
- [2026-10-05] 计划 A 落子：StepProfile 穿进 forward_batch/delta_step/full_step/ffn（SMODELTRANS_INDEX_PROFILE=1 开启，默认零开销），translate 末打印 proj/cpu/ffn/norm 四段；conv 权重提到 new_states 预取。待用户 release 跑 e2e 回填四段数字
- [2026-10-05] release 复测：总量 2.26s→1.78s；TTFT 330–385ms 不随 prompt 长度放大（batch prefill 生效）；decode 约合 53 tok/s（总量口径 15–27 系 prefill 摊薄）；e2e 加第 4 例 long（长输入 prefill 扩展性读数），待跑
- [2026-10-05] long 例首跑：TTFT 500ms（约 200 字 prompt），64 token 顶格跑满，decode 约 50 tok/s，总量 3.40s/102 tok，四例全过。decode 稳定 50–60 tok/s ≈ 上游 52.1 持平；TTFT 全 <1s。阶段二目标达成，待提交
- [2026-10-05] long 放宽到 256 后完整输出（121 tok 自然收尾），总量 4.61s/159 tok。用户追问继续优化：方向是 profiling 定点 + op hygiene 第二轮，MTP 为备选大牌
- [2026-10-05] MTP（env SMODELTRANS_INDEX_MTP=1，默认关）代码完成：model.rs MtpWeights+blk.24 加载（缺失则 None）；session.rs draft 前向 + 推测循环（快照回绕）+ 接受率打印。待用户跑 e2e 判定接受率（≥60% 立项 / <50% 撕票）与开关关闭 parity
 - [2026-10-05] release 实测定案：44s→2.26s（约 20x），debug CPU 递推循环是真凶（TTFT∝prompt 的数学已对上）。TTFT 366–598ms（<1s 达标）；decode 约 50 tok/s；总量 2.26s。100 tok/s 需继续抠 op/分配器，投入产出比待决策
 - [2026-10-05] MTP 撕票：接受率 86% 但总量 3.40s→4.04s 变慢（CPU-bound 下验证 batch≥单步成本，结构性无收益）；代码已全回退（loader/draft/循环/快照/env 门清除），parity 跑输出逐字一致、计时 3.46s/long 60.2 tok/s。纯 candle+CPU 递推架构即达天花板，阶段二结项
