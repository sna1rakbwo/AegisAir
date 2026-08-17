# Phase 5 交接文档（2026-08-16）

> 给下一个对话的完整上下文。新对话先读本文件，再读
> `docs/decisions/phase4_summary.md` 和 `plan.md`。

## 1. 项目与仓库

项目：AegisAir = Adaptive Runtime Assurance for Intelligent Multi-UAV Systems。

代码仓库：`/Users/lijiajun/Documents/drone/AegisAir`（private GitHub，未 push）。

权威计划：`plan.md`。

## 2. 一句话状态

Phase 4（本地 LLM 异步 mission-level replanning）**实现已结束、机制验证通过**；
但还差论文级统计/failure-injection 才算正式冻结。Phase 5 是把它接进 PX4 SITL/Gazebo
闭环。

## 3. 环境与路径

- Python：`/opt/anaconda3/envs/eai-swarm/bin/python`（conda env `eai-swarm`）
- MLX：`mlx 0.32.0` + `mlx-lm 0.31.3`，Metal GPU 可用（Apple M4 16GB）
- 模型（移动硬盘）：`/Volumes/Expansion/safedrones_marllib_vec/models/Qwen3-4B-4bit`
- SSD 基准副本：`$HOME/.cache/aegisair-qwen3-4b-4bit-bench`
- 结果目录：`/Volumes/Expansion/safedrones_marllib_vec/`

## 4. Phase 4 已完成的实现

- 异步 Semantic Mission Manager：`swarm/recovery/async_replanner.py`
- Mission Validity Monitor：`E_safety(route deviation) / E_mission(外部任务变化) /
  E_coord(等待/效率退化)`
- compact LLM 决策 `MissionDecision` → 确定性展开 `RecoveryPlan`：
  `swarm/recovery/decision.py`
- LLM 后端：`MlxLmClient`（真实 Qwen）、`RuleMissionPlanner`（确定性 fallback）
- 执行语义：HOLD / YIELD / REROUTE / REASSIGN / CHANGE_PRIORITY / ABORT / RETURN
- 四层校验：syntactic / schema / semantic / execution acceptance
- 真实 Qwen 三场景 harness：`marllib/phase4_eval_missions.py`

## 5. Phase 4 验证结果

真实 Qwen3-4B-4bit，10 seeds × 三场景（real-time），结果文件：
`/Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json`

| 场景 | 指标 | CBF_ONLY | ASYNC (Qwen) |
|---|---|---|---|
| priority_conflict | 高优先级到达步数 | 92.0 | 86.6 |
| | CBF 介入事件 | 72.0 | 66.2 |
| corridor_blocked | zone_cross_rate | 1.0 | 0.0 |
| drone_failure | 关键任务完成率 | 0.0 | 1.0 |

LLM 30 次调用四层校验全过、0 fallback、单次 replan 延迟约 2.6–3.0s。

## 6. Phase 4 还没做完（建议 Phase 5 前收尾）

1. 用 final test seeds 15–24 跑三场景，出 P50/P90/P95（现在只有 10 seeds 的
   mechanism 验证）。
2. 补一个显式 failure-injection 测试：LLM 输出非法/恶意时，最终命令仍被
   Runtime Assurance/CBF 过滤，不突破 Safety Gate。
3. 在 dense 场景（randomized_8）补一轮 CBF_ONLY vs ASYNC，证明「降低 persistent
   CBF intervention」这一完成标准（`marllib/phase4_eval_replan.py`）。

## 7. Phase 5 目标（PX4/Gazebo 闭环）

来自 `plan.md` Phase 5：

1. telemetry → observation → MARL inference → Runtime Assurance → 异步 replanning
   → PX4 Offboard 全闭环。
2. 用已下载的 Qwen3-4B-4bit 验证 repeated CBF interventions / recurrent conflict /
   mission completion / path efficiency 是否改善。
3. 复用 Phase 4 的 harness 在 Gazebo 场景复测。

## 8. 关键架构决策（必读）

- `docs/decisions/phase4_summary.md`：Phase 4 最终汇总。
- `docs/decisions/llm_async_mission_replanning.md`：LLM = 异步 mission-level
  replanning；CBF = immediate safety。
- `docs/decisions/predictor_architecture.md`：Predictor 只做 Pre-alert。

核心句：

```text
CBF preserves safety; the LLM preserves mission intent under changing constraints.
Protect now -> Understand later -> Replan future.
```

## 9. 冻结接口

`swarm/interfaces.py`：

- 新增 `MissionDecision`（compact LLM 决策）。
- `RiskEvent.event` 新增 `MISSION_PLAN_INVALIDATED`、`REPEATED_ROUTE_CONFLICT`。
- `RiskCause` 新增 `ROUTE_DEVIATION`、`MISSION_CHANGE`、
  `COORDINATION_DEGRADATION`、`REPEATED_CONFLICT`。
- `RiskEvent` 可选 `mission_priority`。

任何新消息格式必须冻结到 `swarm/interfaces.py` 并加测试。

## 10. 可复现命令

```bash
# 测试（85 个）
cd /Users/lijiajun/Documents/drone/AegisAir
/opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s tests

# 三场景真实 Qwen（real-time）
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase4_eval_missions.py \
  --scenarios priority_conflict corridor_blocked drone_failure \
  --seeds 10 --max-steps 120 --modes CBF_ONLY ASYNC \
  --llm qwen --qwen-model "$HOME/.cache/aegisair-qwen3-4b-4bit-bench" \
  --qwen-max-tokens 48 --real-time \
  --output /Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json
```

## 11. Git 状态（未提交）

工作区有大量 Phase 4 改动尚未 commit：

- 修改：`HANDOFF.md`、`plan.md`、`swarm/interfaces.py`、
  `marllib/envs/multi_uav.py`、`tests/test_interfaces.py`
- 新增：`swarm/recovery/`（含 `async_replanner.py`、`decision.py`、`llm.py` 等）、
  `marllib/phase4_eval.py`、`marllib/phase4_eval_replan.py`、
  `marllib/phase4_eval_missions.py`、`scripts/benchmark_qwen_recovery.py`、
  `docs/decisions/phase4_summary.md`、`docs/decisions/llm_async_mission_replanning.md`、
  `docs/decisions/phase4_recovery.md`、`tests/test_recovery.py`、
  `tests/test_async_replanner.py`、`tests/test_decision.py`、`tests/test_rule_planner.py`

建议新对话前先 `git add` + commit，避免上下文丢失。网络恢复后 `git push`。

## 12. 重要约定

1. 文档中文；代码/文档放仓库，训练数据/checkpoint/结果/模型放移动硬盘。
2. 不放松阈值/加种子救失败假设；不把仿真说成真实飞行。
3. Assume AI can fail：LLM / MARL / Predictor 都不可信，只有 Runtime Assurance
   保留最终硬安全。

## 13. 新对话第一步建议
先把git push上去

phase4的实验以后再弄
先弄好 Phase 5 的 PX4/Gazebo 闭环。
