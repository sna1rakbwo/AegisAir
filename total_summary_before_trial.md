# AegisAir 全系统总结（Before Trial）

> 日期：2026-08-17
> 状态：Phase 0–6 结束，实验场景搭建完成，进入 Phase 7 正式实验前
> 仓库：`/Users/lijiajun/Documents/drone/AegisAir`（private GitHub）
> 基线：117 个主测试 + 41 个 `px4_adapter` 测试全绿

## 1. 定位与核心假设

AegisAir = **Adaptive Runtime Assurance for Intelligent Multi-UAV Systems**。

核心假设（贯穿 Phase 0–6，从未放松）：

```text
LLM / MARL / Predictor 都不可信
            ↓
独立的 Risk-Adaptive Runtime Assurance 保留最终硬安全
            ↓
预测安全裕度 + 语义恢复，在硬安全前提下尽量恢复任务性能
```

即：**AI 的失败 / 不完美，不应自动变成安全失败**。

## 2. 冻结后的系统架构

```text
Mission / User
   ↓ 高层目标 / 恢复计划
Local LLM Commander（RuleMissionPlanner / MlxLmClient）
   ↓
MARL Nominal Pilot（MAPPO checkpoint，可换 rule go-to-goal）
   ↓ u_nom
SharedStateEstimator（covariance + AoI；raw telemetry → EstimatedState）
   ↓
Risk-Adaptive Runtime Assurance
   ├── Dynamic Safety Boundary d_safe = d0 + M_dyn + M_perc + M_comm
   ├── Normalized Safety Margin ρ
   ├── Safety-Margin Degradation g
   ├── CV / CPA Predictive Monitor
   └── Sampled-Data Acceleration Barrier（gamma=0.1，PX4 lag-aware）
   ↓ v_safe（velocity setpoint）
PX4 Offboard → PX4 SITL / Gazebo
   ↑
   └── Structured RiskEvent → AsyncMissionReplanner（validator + fallback）
```

最终控制方案（Phase 5/6 冻结）：

```text
sampled-data acceleration barrier（gamma=0.1）
+ SEQUENTIAL_PASS（mission-aware right-of-way）
+ safe holding-point semantics（YIELD/HOLD 先撤后停）
+ PX4 velocity-mode offboard
+ PX4 lag-aware prediction（tau_px4=0.2）
+ AoI state propagation
```

## 3. Phase 0–6 成果总览

| Phase | 内容 | 结果 |
| --- | --- | --- |
| 0 | 冻结跨层接口 | `swarm/interfaces.py` 7 个 pydantic 模型，`extra=forbid` |
| 1 | 轻量多机环境 + MAPPO | `MultiUAVEnv`；训练 single/head-on/perpendicular/diagonal/randomized_2/4/8 |
| 2 | Runtime Assurance | velocity-CBF / HOCBF / sampled-data 三模式；动态安全边界、ρ、g |
| 3 | 预测监视器 | CV/CPA + 标定；predicted margin / TTSB |
| 4 | 语义恢复 | `AsyncMissionReplanner` + `RecoveryValidator` + rule fallback |
| 5 | PX4/Gazebo 闭环 | sampled-data + SEQUENTIAL_PASS；head-on/crossing/4 机 `min_rho>=0` |
| 6 | 故障注入 | SharedStateEstimator；packet loss/stale/latency/perception/LLM 故障 |

## 4. 关键架构决策与三个结构修复

决策文档都在 `docs/decisions/`：

- `hocbf_acceleration_aware.md`：HOCBF → sampled-data 演变、SEQUENTIAL_PASS、PX4 结果。
- `llm_async_mission_replanning.md`：System-1/System-2 分层。
- `phase6_shared_state_estimator.md`：raw telemetry → estimator → RA。
- `phase7_robustness_plan.md`：MAPPO 越界诊断 + 修复 + Phase 7 robustness 设计。

### 4.1 三个结构修复（Phase 6 后期，修复前禁止冻结）

1. **safe holding-point semantics**
   - 修 YIELD/HOLD “原地停车”会制造“移动 UAV 撞向静止 yielding UAV”的危险配置。
   - 现在 YIELD/HOLD 先撤到 swarm 质心外的 holding point，到点后再停。
   - `swarm/safety.py:safe_holding_point` + SEQUENTIAL_PASS + `executor.apply_plan`。

2. **PX4 lag-aware sampled-data prediction**
   - barrier 不再假设速度命令立即实现；改为一阶速度跟踪
     `alpha = 1 - exp(-dt/tau_px4)`。
   - QP 预测 `v_next = v + alpha*a*dt`、`r_next = r + v*dt + 0.5*alpha*a_rel*dt²`。

3. **AoI state propagation**
   - 把 stale telemetry 用 `p̂ = p + v*age` 传播到当前控制时刻再进 RA；
   - `M_comm` 继续保留 uncertainty margin，职责分开。

### 4.2 live 复测证据（修复前 vs 修复后）

| | min_rho | min_distance | cbf_events |
| --- | --- | --- | --- |
| 修复前 | -0.508 | 0.498 m | 326 |
| 修复后 | **+0.564** | **1.523 m** | **0** |

修复后 `cbf_events=0`：危险配置在 mission 协调层被 safe holding-point 消解，
CBF 无需硬刹车。这正好支撑论文分层叙事（协调语义 + RA，而非单靠 CBF 兜底）。

## 5. 仿真 / 实验环境

| 组件 | 值 |
| --- | --- |
| Python | `/opt/anaconda3/envs/eai-swarm/bin/python`（conda `eai-swarm`，3.11） |
| PX4 SITL | `PX4-Autopilot/build/px4_sitl_default/bin/px4`（arm64） |
| Gazebo | Harmonic（`gz sim -s` headless / `gz sim -g` GUI） |
| Docker | `aegisair-px4-bridge:humble`、`safety-margin-gate0-ros2:humble` |
| MQTT | `amqtt`（`config/amqtt.yml`，`0.0.0.0:1883`） |
| LLM | `MlxLmClient` + 本地 Qwen3-4B-4bit（MLX） |
| 结果盘 | `/Volumes/Expansion/aegisair_phase6_20260816/` |

live 4 机 runbook 见 `PHASE6_HANDOFF.md` §5。

## 6. 验证证据链（Before Trial 已完成的证据）

### 6.1 本地冻结闸门（确定性，无 PX4）

- 闸门 A：adapter 本地 fail-closed（`px4_adapter/p5/fault_scan.py`）
  - 90 episodes，`bypass=0`，5 项 stop rules 全过。
- 闸门 B：mission 级本地故障（`phase6_fault_scan.py`）
  - 60 episodes，6 项 stop rules 全过。
- 闸门 C：SharedStateEstimator 本地（`phase6_estimator_scan.py`）
  - 50 episodes，5 项 stop rules 全过。

### 6.2 live 单机

- command loss（10/20/30%）单调、1100ms latency 100% `command_expired` fail-closed。

### 6.3 live 4 机

- baseline / stale 500ms / dropout 30% / LLM timeout / LLM invalid：全部无碰撞。
- LLM timeout：4 trigger → 4 `llm_timeouts` → 4 fallback。
- LLM invalid：4 trigger → 4 `llm_schema_invalid` → 4 fallback。

### 6.4 MAPPO nominal pilot

- `MappoPilot` 已接入 sim 与 live；`randomized_4` 可作为 aggressive imperfect
  policy 的 stress test。

## 7. 当前状态（Before Trial）

- Runtime Assurance 已按 4.1 完成三个结构修复，**建议冻结，不再做结构升级**。
- MAPPO、SharedStateEstimator、SEQUENTIAL_PASS、safe holding、lag-aware barrier、
  AoI propagation 均已接入 sim/live。
- 158 个测试全绿；git 干净，已 push 到 private repo。

## 8. Claim Boundary（什么已证明 / 什么还没证明）

**已证明：**

- 确定性决策路径：adapter fail-closed、validator+fallback、estimator→RA。
- live 闭环在注入 packet loss / latency / stale / dropout / LLM timeout /
  invalid 下无碰撞。
- 结构修复后 aggressive MAPPO 在 4 机 live 从 `min_rho=-0.51` 回到 `+0.56`。

**尚未证明（Phase 7 正式实验才做）：**

- 多 seed / 多 episode 统计（20–50 episodes/condition、mean+CI）。
- nominal policy 分层（rule / weak / medium / strong MAPPO）× RA on/off。
- OOD、swarm scaling（2/4/8）、LLM baseline、failure envelope。
- 无法保证“对任意 MARL 绝对安全”；RA 的 safety guarantee 只覆盖
  **feasible / well-modeled envelope**。

## 9. 进入 Phase 7 的准备度

Phase 7 按 `docs/decisions/phase7_robustness_plan.md` 执行，冻结对象：

- 独立变量：nominal policy（rule / weak / medium / strong MAPPO）× RA（off/on）。
- 因变量：collision rate、boundary violation rate、min_rho、completion、
  intervention magnitude、QP feasible、saturation、constraint residual、
  tracking error、deadlock/replanning rate。
- 统计：固定 evaluation seed set（与 training seed 分开）、20–50
  episodes/condition、mean + uncertainty interval。

**Go/No-Go：Go** —— 环境、代码、冻结闸门、live 验证、结构修复均已就绪，
可以开始跑正式 robustness 实验。
