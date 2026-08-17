# AegisAir 交接文档（2026-08-15）

> 给下一个对话的完整上下文。新对话开始后先读这份文档，再读 plan.md 和
> docs/decisions/。

## 1. 项目定位

AegisAir = Adaptive Runtime Assurance for Intelligent Multi-UAV Systems。

核心假设：LLM、MARL、Predictor 都不可信，只有确定性 Runtime Assurance
（动态安全边界 + 归一化裕度 + time-varying CBF）负责硬安全。

论文目标：证明 autonomy quality 与 hard runtime safety 可以解耦；通过预测安全
裕度与语义恢复，在保持硬安全的前提下恢复任务性能。

## 2. 仓库与数据位置

* 代码仓库：/Users/lijiajun/Documents/drone/AegisAir
* 权威计划：plan.md（v1.0）
* 数据/结果（移动硬盘）：/Volumes/Expansion/safedrones_marllib_vec/
* GitHub（private）：https://github.com/EthelFord1/AegisAir-Adaptive-Runtime-Assurance-for-Intelligent-Multi-UAV-Systems
* 旧原型（已分离，别改）：/Users/lijiajun/Documents/drone/SafeDrones

## 3. 环境

* Python：/opt/anaconda3/envs/eai-swarm/bin/python（Conda env eai-swarm）
* 依赖：torch 2.13 + numpy 2.4 + pydantic 2 + paho-mqtt + opencv（无 gymnasium）
* 运行测试：cd /Users/lijiajun/Documents/drone/AegisAir && /opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s tests
* 当前 52 个测试全过

## 4. 代码结构

```text
swarm/
  interfaces.py       Phase 0 冻结接口（pydantic，7 个 schema）
  geometry.py         向量运算
  safety.py           DroneSnapshot / 旧 Safety Gate
  ego.py              EgoStateStore
  perception.py       双目感知
  stereo.py           双目几何
  ra/                 Phase 2+3 Runtime Assurance
    margins.py        RuntimeAssuranceParams + 动态安全边界
    margin.py         归一化 rho + 退化 g（EMA）
    cbf.py            time-varying CBF 速度过滤
    predictor.py      PredictiveMonitor（CV/CA/CPA/不确定性/TTSB/reliability）
    runtime_assurance.py  编排器
  recovery/            Phase 4 语义恢复层
    llm.py              LLMRecoveryClient + DeterministicRecoveryClient + TransformersQwenClient
    validator.py        RecoveryPlan schema + whitelist + 禁止字段/arena/ttl 校验
    executor.py         7 个恢复动作执行语义 + TTL
    orchestrator.py     R0/R1/R2 三层触发 + latency budget/fallback + 指标
marllib/              Phase 1 轻量 MARL
  config.py           场景 + 奖励配置 + 课程
  reward.py           奖励函数
  envs/multi_uav.py   单环境
  envs/vectorized.py  向量化环境
  policies/mappo.py   MAPPO
  train_vec.py        向量化训练入口（支持 CUDA）
  run_sweep.py        批量 sweep 驱动
  eval_sweep.py       评测全部 checkpoint
  eval_ra.py          RA vs baseline
  eval_predictor.py   提前量评测
  eval_predictor_full.py  离线冲突检测
  phase3_calibrate.py 预测误差校准
  phase3_eval.py      margin/阈值/可靠性/可行性
  phase4_eval.py      R0/R1/R2 语义恢复对比
  sweep_horizon.py    horizon sweep
  selective_trigger.py  persistence/CPA/event-level
  ablate_predictor.py P0-P4 ablation
px4_adapter/          PX4 SITL/Gazebo 适配器
schemas/              Phase 0 JSON Schema
examples/             Phase 0 示例
docs/interfaces/      冻结接口规范
docs/decisions/       架构决策文档
tests/                78 个测试
```

## 5. 当前进度（Phase 0-4 完成，Phase 5 下一步）

### Phase 0 冻结接口（完成）

7 个跨层接口：Telemetry / Observation / MarlAction / RiskEvent / RecoveryPlan /
SafetyDecision / LogEvent。定义在 swarm/interfaces.py。

### Phase 1 轻量 MAPPO（完成）

* 向量化环境 + MAPPO（minibatch 256, epochs 5）。
* 课程：single / head_on / perpendicular / diagonal / randomized 2/4/8。
* 5 seeds 全部训练完（数据在移动硬盘）。
* 结果：单机/2 机 0-1% 碰撞；randomized_4 18% 碰撞、randomized_8 59% 碰撞。
* 这证明 MARL 在 dense 场景不可信，需要 Runtime Assurance。

### Phase 2 Runtime Assurance（完成）

* swarm/ra/：动态安全边界 + 归一化 rho + 退化 g + CBF/QP 过滤。
* 结果：randomized_4 碰撞 18% 到 0%、完成率 78% 到 96%；randomized_8 碰撞 59%
  到 1%、完成率 38% 到 85%。

### Phase 3 预测监视器（完成，已冻结，不再碰）

* Filtered CA + CPA + 不确定性 + TTSB + reliability score。
* 冻结配置：Filtered CA, Q=0, rho_th=0, H=1.5s, K=1。
* untouched test（seeds 15-24）：precision 49.5%, recall 54.9%, FPR 1.0%,
  margin MAE 1.77, TTSB MAE 0.18s。
* 关键负面结论：persistence/CPA 只能把 precision 提到约 44% 且 recall 掉到 19%；
  event-level 反而更低；prediction error heavy-tailed；confidence 不可校准。

### Phase 4 本地 LLM Async Mission Replanning（完成）

* 新增 `swarm/recovery/`：可替换 LLM 后端 + schema/whitelist/forbidden-field 校验
  + 7 个恢复动作执行语义 + R0/R1/R2 三层触发编排器 + latency budget 超时兜底。
* 默认 `DeterministicRecoveryClient` 离线可复现；真实 4-bit 走 `MlxLmClient`
  （MLX 已装：mlx 0.32.0 + mlx-lm 0.31.3，Metal 可用）；`TransformersQwenClient`
  的 4-bit 仍需 bitsandbytes（CUDA），本机显式失败不假装量化。
* `marllib/phase4_eval.py` 输出 4 个核心指标：recovery latency、
  unnecessary mission changes、CBF intervention duration、mission completion time。
* randomized_8（20 seeds，训练 MARL seed1）：R1 产生 1132 次任务变更 / 338 次
  无谓变更、完成率 0.80；R0/R2 完成率 0.95、CBF 介入约 9.9s，R2 预热候选平均
  提前 0.175s。详见 `docs/decisions/phase4_recovery.md`。
* 2026-08-15 架构重定位：LLM 定位为异步 Semantic Mission Manager，由
  Mission Validity Monitor 触发（E_safety route deviation / E_mission 任务
  变化 / E_coord 等待退化）。详见
  `docs/decisions/llm_async_mission_replanning.md`。
* 真实 `Qwen3-4B-4bit`（MLX，非 thinking）100 次基准：valid JSON 1.00，
  TTFT P50 2.07s / generation P50 0.76s / total P50 2.83s（短 JSON）。
  完整 `RecoveryPlan` 单次生成约 23s（含 prefill + 约 300 token 输出），
  因此真实 LLM 是「慢但异步」，不要求赶在单次碰撞前完成。
* LLM 输出改为「compact 高层决策」→ `MissionDecision` → 确定性展开为
  `RecoveryPlan`，单次生成降到约 2.8s；并加入四层校验（syntactic / schema /
  semantic / execution acceptance）。
* 三组 mission 场景 harness：`priority_conflict` / `corridor_blocked` /
  `drone_failure`，用 `RuleMissionPlanner`（LLM 的确定性替身）+ 异步
  replanning 已跑通，见 `marllib/phase4_eval_missions.py`。
* 真实 Qwen 10 seeds × 三场景（real-time）跑通：LLM 30 次调用四层校验全过、
  fallback=0。`drone_failure` 关键任务完成率 0→1.0；`corridor_blocked`
  zone_cross 1.0→0.0；`priority_conflict` 高优先级到达步数 92→86.6、
  CBF 介入 72→66.2。结果写入
  `/Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json`。
* Phase 4 完整总结：`docs/decisions/phase4_summary.md`。

## 6. 关键架构决策（必读）

当前顶层定位（2026-08-15，见
docs/decisions/llm_async_mission_replanning.md）：

```text
CBF / Runtime Assurance = immediate safety control（同步，当前这一刻不能撞）
LLM = asynchronous mission-level replanning（为什么反复冲突、谁让路、谁改航路）
Protect now -> Understand later -> Replan future
```

LLM 不抢方向盘，LLM 改路线图。LLM 触发是 Mission Validity Monitor：
`E_replan = E_safety（route deviation）OR E_mission（任务变化）
OR E_coord（等待/效率退化）`。

见 docs/decisions/predictor_architecture.md：

Predictor 降级为 Pre-alert，不再是 LLM 任务级触发器。

三层架构：

```text
Level 1 - Predictive Pre-alert（H=1.5, K=1）
    预热语义恢复上下文 / 生成候选恢复计划，不修改任务
Level 2 - Runtime Confirmation（rho 小于 rho_warn 且 (g_bar 大于 g_th 或 N_CBF 大于 N_th)）
    确认后才允许 LLM 改任务
Level 3 - Hard CBF（独立兜底）
```

格言：Prediction suggests; Runtime evidence confirms.

## 7. 下一步 Phase 5（PX4/Gazebo 闭环）

把 Phase 4 的异步 mission-level replanning 接进 PX4 SITL/Gazebo：

1. telemetry → observation → MARL inference → Runtime Assurance → 异步 replanning →
   PX4 Offboard 全闭环。
2. 用已下载的 Qwen3-4B-4bit 验证 repeated CBF interventions / recurrent
   conflict rate / mission completion time / path efficiency 是否改善。
3. 复用 Phase 4 的 R0/R1/R2 harness 在 Gazebo 场景复测。

## 8. 重要约定

1. 文档用中文写。
2. 代码/文档放仓库，训练数据/checkpoint/结果放移动硬盘。
3. 每次大方向调整，在 docs/decisions/ 写带日期的决策文档。
4. 数据分离：calibration seeds 0-9，validation 10-14，final test 15-24。
5. 不放松阈值/加种子去救失败假设；不把仿真说成真实飞行。
6. Assume AI can fail：LLM/MARL/Predictor 都不可信，只有 Runtime Assurance 硬安全。

## 9. Git 状态

本地有 4 个 commit 还没 push（网络时好时坏）：

```text
0682dc8 Rename prediction_confidence to reliability_score; add horizon sweep
a40e664 Freeze Phase 3 predictor: H=1.5, reliability_score rename, final untouched test
72c66fe Add selective proactive trigger experiments
d156913 Document architecture decision: predictor as pre-alert
```

网络恢复后执行：cd /Users/lijiajun/Documents/drone/AegisAir && git push origin main

## 10. 外部结果文档（移动硬盘）

* SWEEP_RESULTS.md：Phase 1 课程结果
* PHASE2_RESULTS.md：Runtime Assurance 结果
* PHASE3_SUMMARY.md / PHASE3_FINAL_SUMMARY.md / PHASE3_FREEZE.md：预测器总结
* eval_summary.json：sweep 聚合
* prediction_calibration.json：预测误差校准
