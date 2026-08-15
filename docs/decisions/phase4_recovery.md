# 架构决策：Phase 4 语义恢复层实现（2026-08-15）

> 本文是 Phase 4 早期实现记录，后续架构已演进为「异步 mission-level
> replanning + compact LLM 决策」。最终口径与结果见
> `docs/decisions/phase4_summary.md`，本文保留作为历史证据。

## 决策

Phase 4 的本地 LLM 语义恢复层按「可替换后端 + 确定性兜底」实现，默认不依赖
大模型即可完整验证 R0/R1/R2 三层触发与安全边界。真正的本地 Qwen3-4B 4-bit
走 `MlxLmClient`（Apple Silicon 的 Metal 路径）；`TransformersQwenClient` 的
4-bit 仍依赖 `bitsandbytes`（CUDA），在该环境里会显式失败而不悄悄降级。

MLX 已在 `eai-swarm` 环境安装并验证 Metal 可用：
`mlx 0.32.0` + `mlx-lm 0.31.3`，默认设备 `Device(gpu, 0)`。

核心不变式：

> LLM / MARL / Predictor 都不可信，Recovery 只生成高层意图，最终命令永远
> 经过 Runtime Assurance 的 CBF 过滤后才进入执行器。

## 新增模块

```text
swarm/recovery/
  llm.py           LLMRecoveryClient + DeterministicRecoveryClient + TransformersQwenClient + MlxLmClient
  validator.py     RecoveryPlan schema + action whitelist + forbidden-field + arena/ttl 校验
  executor.py      HOLD/YIELD/REROUTE/REASSIGN/CHANGE_PRIORITY/ABORT/RETURN 的执行语义与 TTL
  orchestrator.py  R0/R1/R2 三层触发状态机 + latency budget/fallback + 指标计数
marllib/phase4_eval.py   R0/R1/R2 对比 harness
tests/test_recovery.py   17 个 Phase 4 单元测试
```

`MultiUAVEnv` 增加 `set_goal()`（仅用于语义恢复改写目标，不改既有训练/评测路径）。

## 三层触发语义

```text
R0  无预测器：runtime confirmation -> LLM 直接生成并执行
R1  预测器直接触发 LLM 执行（不做 runtime confirmation 门控）
R2  预测器 speculative planning + runtime confirmation（主方案）
    pre-alert 生成候选计划，不执行；
    runtime confirmation 确认后，若候选匹配则直接执行，否则现场生成并执行；
    候选超时（candidate_ttl）未确认则丢弃。
```

Runtime confirmation 条件（沿用 `predictor_architecture.md`）：

```text
rho < rho_warn 且 (g_bar > g_th 或 N_CBF(T) > n_th)
```

默认冻结值：

```text
rho_warn = 0.2
g_th = 0.3
n_th = 1
intervention_window_s = 0.5
latency_budget_s = 0.6
candidate_ttl_s = 2.0
confirmation_window_s = 1.0
```

## 动作白名单与执行语义

白名单即 `swarm/interfaces.py` 的 `RecoveryAction`：
`HOLD / YIELD / REROUTE / REASSIGN / CHANGE_PRIORITY / ABORT / RETURN`。

执行器只产生高层意图（velocity scale / goal override / abort / priority），
不直接下发 actuator：

```text
HOLD             velocity_scale = 0（TTL 后恢复）
YIELD            velocity_scale = 0.4（TTL 后恢复）
REROUTE          goal_override = waypoint（TTL 后回到 base goal）
REASSIGN         goal_override = waypoint（持久）
RETURN           goal_override = base_goal（持久）
CHANGE_PRIORITY  priority 改写
ABORT            aborted=True 且 velocity_scale = 0
```

## LLM 校验与兜底

所有 LLM 输出先过 `RecoveryValidator`：

1. `RecoveryPlan` pydantic schema（`extra="forbid"`）。
2. action 白名单（`RecoveryAction` Literal）。
3. 禁止字段：`d0 / cbf / safety_threshold / actuator_safety_limit` 等。
4. REROUTE/REASSIGN/RETURN 必须有 waypoint，且 waypoint 在 arena 内。
5. `0 < ttl_sec <= max_ttl_sec`；`keep_min_distance_m` 不低于硬安全下限。

LLM 超时（latency_budget）或输出非法时，`SemanticRecovery` 回退到
`DeterministicRecoveryClient` 的安全 REROUTE 计划，避免恢复被卡死。

## Phase 4 对比结果（randomized_8，20 seeds，训练 MARL seed1）

确定性后端（`plan_latency_s=0`），collision 均为 0：

```text
mode   mission_changes  unnecessary  completion  cbf_intervention_s  lead_time_s
R0     108              0            0.95        9.94                -
R1     1132             338          0.80        11.66               -
R2     106              0            0.95        9.99                0.175
```

R1 把预测直接当触发，产生 1132 次任务变更（其中 338 次未获 runtime 证据确认），
完成率反而从 0.95 掉到 0.80，CBF 介入时长从 9.94s 升到 11.66s。

R2 与 R0 的安全/性能基本一致（完成率 0.95，CBF 介入约 9.9s），但 R2 以
speculative planning 预热了 185 个候选、丢弃 166 个未确认候选、执行 106 次
必要任务变更，且平均提前 0.175s 准备。这直接支撑：

> Prediction suggests; Runtime evidence confirms.

## 延迟预算验证（确定性后端 + 模拟 0.3s LLM 生成）

```text
R0 recovery_latency_s_mean = 0.407s  (0.3s LLM + 0.1s 应用)
R2 recovery_latency_s_mean = 0.300s  (部分确认复用候选，只付 0.1s 应用)
R2 candidate_lead_time_s_mean = 0.086s
```

R2 在候选已预热时，确认时的恢复延迟不含 LLM 生成时间；R0 必须在确认现场等待
LLM。真实 4-bit Qwen3-4B 部署后应重测并核对 0.4-0.6s 预算；当前环境无 MLX/
bitsandbytes，因此未声称真实模型延迟达标。

## 真实 MLX 基准（2026-08-15，模型已复制到内置 SSD）

关闭 thinking（`enable_thinking=False`，统一走 chat template）后，用真实
`RiskEvent -> 短 JSON` 工作负载，同一已加载模型连续测 100 次（warmup 10）：

```text
max_tokens = 24
TTFT       P50 2.07s / P90 2.17s / P95 2.19s
generation P50 0.76s / P90 0.79s / P95 0.81s
total      P50 2.83s / P90 2.94s / P95 2.98s
valid JSON rate = 1.00
```

非 thinking 输出已经是干净 JSON（`{"action": ..., "agent": ...}`），但总延迟仍
约 2.8s，瓶颈在 TTFT（prefill + 首 token）而非生成。

**关键：speculative hiding 目前不成立。** randomized_8 的 R2 实测
`candidate_lead_time_s_mean = 0.175s`，远小于 2.8s 的 LLM 延迟。也就是说，冻结
的 H=1.5s 预测器给出的 pre-alert→confirmation 提前量只有约 0.18s，R2 无法把
LLM 生成藏进这个窗口。现在真正的约束是 pre-alert 提前量，而不只是模型大小。

## 结论与下一步

Phase 4 的架构、触发、校验、兜底和 R0/R1/R2 对比 harness 已完成并可离线复现；
MLX 后端 `MlxLmClient` 已就绪。

2026-08-15 已把 `mlx-community/Qwen3-4B-4bit`（约 2.1GB）经 hf-mirror +
aria2c 下载到移动硬盘（本机内置 SSD 另存一份用于基准）。`mlx_lm.load` 可加载。
真实非 thinking 总延迟约 2.8s（TTFT 约 2.1s），仍不满足 0.4-0.6s 预算；且
pre-alert 提前量约 0.175s，无法用 R2 speculative planning 隐藏该延迟。

下一步选择（需冻结一条新决策，不改 Phase 3 的 H=1.5s 除非另写决策）：
1. 接受约 2.8s，把 recovery 改为异步队列，指标从 T_confirm→action 改为
   candidate-ready→commit；
2. 换更小模型压 T_LLM，但 0.175s 提前量仍是绑定约束；
3. 重做 predictor 提前量（提高 H 或降低 pre-alert 阈值）以放大
   pre-alert→confirmation 窗口，再进入 Phase 5。
