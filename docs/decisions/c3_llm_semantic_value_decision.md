# C3 LLM 语义价值探索：负结果收尾决策

日期：2026-08-20

## 1. 问题

冻结 C3 v1 已证明 fail-closed 语义恢复架构有效，但三个场景 rule 都是 0.1 s 最优，
本地 LLM 无可增值空间。本轮探索的问题是：能否构造「规则启发式次优/盲区」的场景，
让本地 Qwen3-4B 带来可测量的语义决策价值，从而支撑「LLM 语义恢复优于 rule」的 claim。

## 2. 探索路径与结果

### 2.1 异步队列（fallback-first）smoke —— 正向基础设施，但不证明 LLM 增值

把恢复承诺与 LLM 延迟解耦（触发当拍先提交 rule fallback，LLM 仅在后台前瞻），消除了
冻结 C3 v1 里 priority-conflict 因等待慢 LLM 造成的安全退化（min rho 负转正、任务恢复）。
这是正向的基础设施改动，但 LLM 计划仍几乎不被采纳，不能作为 LLM 增值的证据。

### 2.2 重分配——算术场景 `reassign_3` —— 负结果

构造 greedy「最近机接管」次优的 3 机 3 目标：rule 总路径 13.8 m，最优（转派非最近机）
7.9 m，证明 rule 确实次优。但真实 Qwen 即使被明确告知「最小化总路径」仍选 greedy
（agent=1，5/5）。根因是全局代价最小化本质是距离算术，恰是 4B 模型短板。

### 2.3 重分配——优先级场景 `priority_reassign` —— 端到端负结果

构造「最近机是 critical、不得转派」的定性约束，rule 的最近机启发式无视优先级（3/3
`priority_violation=True`），是真实盲区。对照探针：

- 聚焦 prompt（只问别碰 critical 机）：Qwen 5/5 选对（agent=2）。
- 完整 mission-manager prompt（含 fail_drone/block_corridor/priority_change/死锁等全部
  指令，并把优先级约束标为 `HARD CONSTRAINT`）：Qwen 5/5 选 greedy（agent=1）。

即 4B 模型在聚焦指令下能做，但在复杂多场景 prompt 下守不住这条具体约束，退化为就近。

## 3. 结论

1. 本地 Qwen3-4B 在完整语义恢复 prompt 下不能稳定遵循具体语义约束，与冻结 C3 v1 一致：
   LLM 无稳定任务收益。
2. `Assume AI can fail` 得到进一步支持：LLM 的语义价值脆弱，复杂上下文下回到最简单默认
   启发式；rule fallback + validator + RA 仍是可靠安全骨架。
3. 不得在论文中写「LLM 在重分配/优先级场景优于 rule」。

## 4. 可写边界

作为 limitation / future work 可写：LLM 在聚焦指令下可遵循定性约束，但在完整多场景
prompt 下失效；若要用 LLM 做语义恢复，需拆 prompt 结构或换更强模型，并重新预注册。

## 5. 保留产物（证据与基础设施）

- `swarm/recovery/async_replanner.py`：`immediate_fallback`（异步队列）+ `initial_priorities`
- `marllib/phase5_runner.py`：完成判据计入 `goal_override`（正确性修复）、`reassign_3` /
  `priority_reassign` 场景、`priority_violation` 指标
- `swarm/recovery/decision.py`：fail_drone 重分配长 TTL（静态目标不因 LLM 延迟过期）
- `swarm/recovery/llm.py`：REASSIGN 优先级约束（即便显式写出，4B 仍失效，作为证据保留）
- smoke/probe 脚本：`run_c3_async_queue_smoke.py`、`run_c3_reassign_smoke.py`、
  `run_c3_priority_smoke.py`、`probe_qwen_*.py`
- 协议草案：`AEGISAIR_C3_LLM_REASSIGNMENT_PROTOCOL.md`
