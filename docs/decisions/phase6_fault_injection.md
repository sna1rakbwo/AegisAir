# Phase 6 —— 故障注入：本地冻结闸门

> 日期：2026-08-16
> 状态：三道本地闸门已跑通，进入 live PX4 前
> 前置：Phase 5 已达成 head-on / crossing / multi-UAV 的 `min_rho >= 0`
> 基线：111 个主测试 + 41 个 `px4_adapter` 测试全绿

## 1. 目标（来自 plan.md Phase 6）

注入并观察以下故障下系统的安全与降级行为：

- packet loss；
- stale telemetry；
- latency；
- perception noise；
- GCS / offboard loss；
- LLM timeout；
- invalid LLM command。

约定不变：不放松阈值、不加 seed 救失败假设；`Assume AI can fail`，Runtime
Assurance 保留最终硬安全。

## 2. 本阶段完成的本地闸门

### 2.1 闸门 A：adapter 本地 fail-closed 冻结

- 入口：`px4_adapter/p5/fault_scan.py` + `aggregate.py`
- 协议：`safedrones-px4-adapter-p5-local-v1`
- 故障：`NONE / COMMAND_LOSS_10/20/30 / COMMAND_LATENCY_100/1100MS /
  TELEMETRY_STALE / GCS_CONNECTION_LOST / OFFBOARD_LOSS`
- 种子：`1..10`，时长 6000ms，命令间隔 500ms
- 结果：90 episodes，`bypass_total = 0`，5 项 frozen stop rules 全过：

  - `bypass_total_zero`
  - `command_loss_monotonic`（10/20/30 接受数单调下降）
  - `command_latency_1100_expired`（TTL 超时命令全部 `command_expired`）
  - `telemetry_stale_detected`（`TELEMETRY_STALE` 与 `OFFBOARD_LOSS` 均触发
    `telemetry_stale`）
  - `gcs_lost_detected`（触发 `gcs_connection_lost`）

- fail-closed 首次动作延迟：
  - `GCS_CONNECTION_LOST`：0ms（立即）
  - `TELEMETRY_STALE` / `OFFBOARD_LOSS`：1500ms（`telemetry_stale_sec=1.5`）

结论：adapter 的本地 fail-closed 决策已冻结；任何被接受的命令不得绕过本地
限制（`bypass=0`）。

### 2.2 闸门 B：mission 级本地故障扫描（新增）

- 入口：`marllib/phase6_fault_scan.py`
- 协议：`safedrones-aegisair-phase6-local-v1`
- 种子：`1..10`，`max_steps=200`，`dt=0.05`，sampled-data `gamma=0.1`
- safety/适配器故障用 `head_on`；LLM 故障用 `priority_conflict`（`change_step=0`
  保证确定性触发 replanner）

覆盖的 mission 级故障：

- `PERCEPTION_NOISE`：对 RA 观测到的共享状态（位置/速度）注入高斯噪声
  （`pos=0.2m`, `vel=0.1m/s`），真实动力学不扰动。
- `STALE_TELEMETRY`：遥测时间戳回退 3000ms，同时把 AoI=3s 传入 RA。
- `COMMAND_LATENCY`：命令时间戳回退 1100ms（`ttl=0.5s`）。
- `LLM_TIMEOUT`：慢后端（`plan_latency_s=0.3`）+ `replan_timeout_s=0.05`。
- `INVALID_LLM_COMMAND`：后端返回非法动作 `{"action":"HOVER"}`。

结果（60 episodes，6 项 stop rules 全过）：

| 故障 | collision | completed | perceived min_rho | fail-closed / fallback 证据 |
| --- | --- | --- | --- | --- |
| NONE | 0 | 10/10 | 0.542 | - |
| PERCEPTION_NOISE | 0 | 10/10 | -0.440 ~ 0.497 | - |
| STALE_TELEMETRY | 0 | 0/10 | -0.605 | `telemetry_stale` 4000 |
| COMMAND_LATENCY | 0 | 10/10 | 0.542 | `command_expired` 3080 |
| LLM_TIMEOUT | 0 | 0/10 | 0.465 | `llm_timeouts` 10, fallback 10 |
| INVALID_LLM_COMMAND | 0 | 0/10 | 0.458 | `llm_schema_invalid` 10, fallback 10 |

说明：

- `completed=0` 的 LLM 故障来自 `priority_conflict` 场景本身（非故障时
  `ASYNC` 也会在 200 步内不完成），本闸门只验证 fallback 被提交，不验证
  该场景的使命完成率。
- `STALE_TELEMETRY` 下 `completed=0`：3s AoI 使 `d_safe` 显著增大，RA 极保守，
  使命停滞；真实 PX4 中 adapter 会 fail-closed（LAND），本闸门未闭环该动作。

### 2.3 闸门 C：SharedStateEstimator 本地闸门（新增）

- 入口：`marllib/phase6_estimator_scan.py`
- 协议：`safedrones-aegisair-phase6-estimator-local-v1`
- 架构依据：`docs/decisions/phase6_shared_state_estimator.md`
- 实现：`swarm/estimation.py`（`SharedStateEstimator` + `EstimatedState`）；
  RA 用 covariance 在连线方向的投影替代固定 `perception_sigma`。

配置：`NONE / DELAY_300MS / DELAY_2000MS / DROPOUT_30 / COV_HIGH`，
种子 `1..10`，`head_on`，sampled-data `gamma=0.1`，`max_steps=300`。

结果（50 episodes，5 项 stop rules 全过）：

| config | collision | completed | perceived min_rho | mean age ms | mean cov m2 |
| --- | --- | --- | --- | --- | --- |
| NONE | 0 | 10/10 | 0.542 | 0 | 0.010 |
| DELAY_300MS | 0 | 10/10 | 0.591 | 289 | 0.010 |
| DELAY_2000MS | 0 | 0/10 | -0.705 | 1733 | 0.010 |
| DROPOUT_30 | 0 | 10/10 | 0.042 ~ 0.504 | 21 | 0.0104 |
| COV_HIGH | 0 | 0/10 | 0.230 | 0 | 0.250 |

观察（与架构决策一致）：

- delay 只抬 age（`M_comm`），covariance 保持 base；
- dropout 抬 covariance（hold + 增长）；
- 高 covariance 使 `min_rho` 下降且 300 步内使命不完成（RA 更保守，fail-safe）；
- 2s delay 下 perceived `min_rho` 为负，但 ground-truth `collision=0`。

## 3. 冻结的 stop rules（闸门 B）

```text
no_collision                所有 episode collision == false
baseline_min_rho_ge_0       NONE 的 perceived min_rho >= 0
stale_telemetry_fail_closed STALE_TELEMETRY 出现 telemetry_stale 拒绝
command_latency_fail_closed COMMAND_LATENCY 出现 command_expired 拒绝
llm_timeout_fallback        LLM_TIMEOUT llm_timeouts>0 且 fallback>0
invalid_llm_fallback        INVALID_LLM_COMMAND llm_schema_invalid>0 且 fallback>0
```

### 3.1 冻结的 stop rules（闸门 C）

```text
no_collision                 所有 episode collision == false
baseline_min_rho_ge_0        NONE 的 perceived min_rho >= 0
delay_increases_age          DELAY_300MS mean_age > NONE mean_age
dropout_increases_covariance DROPOUT_30 mean_cov > NONE mean_cov
high_covariance_reduces_min_rho COV_HIGH min_rho < NONE min_rho
```

## 4. 声明边界（不要过度声称）

这三道闸门只验证**确定性决策路径**：

- 闸门 A 验证 adapter 的 fail-closed **决策**，不验证 PX4 飞行安全；
- 闸门 B 验证 RA + adapter **决策**和 recovery-layer **fallback 路径**，
  **不闭环** adapter fail-closed 后的飞行器动作。
- 闸门 C 验证 `SharedStateEstimator -> RA` 的**决策路径**，不验证真实
  PX4 telemetry 的 delay/dropout 分布，也不闭环飞行器动作。

`min_rho` 是 RA 在**感知状态**上算出的归一化安全裕度；`collision` 是环境
**真实状态**。因此 `PERCEPTION_NOISE` / `STALE_TELEMETRY` 下 perceived
`min_rho` 可以为负，而真实 `collision` 保持为 0。二者不能混为一谈。

Phase 5 的「`min_rho >= 0`」只适用于无故障基线；故障下的硬安全指标是真实
`collision`，以及 fail-closed / fallback 是否被触发。

## 5. Go/No-Go：进入 live PX4 故障注入

Go，但需先处理内存压力：

- 本机 16GB，Phase 5 已知 4 机 PX4 偶发 `Killed: 9` 或单实例挂死；
- live 前先释放内存，按最低成本顺序：单机 packet loss / stale / latency →
  4 机 sampled-data + SEQUENTIAL_PASS 故障注入；
- live 结果同样落到
  `/Volumes/Expansion/aegisair_phase6_20260816/`。

## 6. 下一步

1. ~~把 `SharedStateEstimator` 接入 live 路径~~（已完成：`phase5_runner.py`
   新增 `--estimator` 系列参数，MQTT telemetry -> estimator -> RA）。
2. 单机 live 故障注入（packet loss / stale telemetry / latency），复用
   `px4_adapter/p5/live_command_fault_scan.py` 的 frozen 协议。
3. 4 机 sampled-data + SEQUENTIAL_PASS 下注入故障，观察
   `min_rho`、真实 `collision`、完成率，以及 LLM timeout / invalid command
   是否仍被 validator + fallback 兜住。
4. 若 live 结果与本地闸门不一致，回查 estimator / adapter 闭环路径，不放松阈值。
