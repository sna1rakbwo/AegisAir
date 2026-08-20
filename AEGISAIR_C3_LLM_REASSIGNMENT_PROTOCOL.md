# AegisAir C3 扩展：LLM 语义重分配实验协议（草案）

状态：**已收尾为负结果，不再冻结执行。** smoke 门证明本地 Qwen3-4B 无法在完整
mission-manager prompt 下稳定优于 rule（算术与优先级两个场景均选 greedy）。详见
`docs/decisions/c3_llm_semantic_value_decision.md`。本文保留为探索记录与 future work
的预注册框架。

## 1. 背景与假设

冻结 C3 v1 已经证明 fail-closed 语义恢复架构有效（确定性规则恢复 + validator +
fallback + RA 否决，safety bypass = 0）。但它的三个场景（drone failure / blocked
corridor / priority conflict）里，`RuleMissionPlanner` 都是 0.1 s 内的无歧义最优解，
本地 LLM 没有可增值的空间，因此只能写诚实的负结果。

本实验换一个**规则确实次优**的场景，检验本地 LLM 是否能在任务级决策质量上带来可
测量的收益。

核心假设：在「失效后任务重分配」这种需要**全局权衡**的场景，`RuleMissionPlanner`
的 greedy 规则（“最近健康机接管”）次优；LLM 若能做全局语义推理，可以选出代价更低的
分配。

关键设计决策：**把“决策质量”和“延迟”解耦。** 重分配的目标是一个静态坐标，不是动态
冲突，2.8 s 的 LLM 延迟不构成障碍。R2 使用异步提交 + 足够长的计划有效期，使 LLM 的
计划不会被“过期”误杀——这样测的是“LLM 会不会选得更好”，而不是“LLM 够不够快”。

## 2. 冻结对象

正式运行前冻结：场景几何、seed manifest、LLM 模型与推理参数、RA 参数（复用 C3 的
exact-ZOH 配置）、指标定义、统计方法、停止规则。

## 3. 场景定义（候选几何 + 次优性）

场景名 `reassign_3`：3 机 3 目标，drone 0 在 step 0 失效，其目标 `G0` 成为必须被覆盖
的 orphan critical goal。默认 `speed_limit=1.5 m/s`。

| drone | start | 原始 goal |
|---|---:|---:|
| 0（失效） | (-4, 1) | G0 = (2, 0) |
| 1 | (0, 0) | G1 = (3, 0) |
| 2 | (0, 3) | G2 = (10, 3) |

失效后，`RuleMissionPlanner` 挑「离 G0 最近的健康机」重分配，并放弃该机自身目标；
另一机继续自身目标。两种分配的**总路径长度**（单位 m）：

| 分配 | 重分配机 | 各机终点 | 总路径 |
|---|---|---|---:|
| greedy（rule） | drone 1（到 G0 距离 2.0） | 1→G0、2→G2 | 2.0 + 10.0 = **12.0** |
| 全局最优 | drone 2（到 G0 距离 3.606） | 1→G1、2→G0 | 3.0 + 3.606 = **6.606** |

greedy 比最优慢约 1.82×（多 5.4 m）。原因是 greedy 只看到“谁离 G0 近”，忽略了
“该机自身目标是否近、被重分配后机会成本是否高”：drone 1 自身目标就在附近（3.0），
重分配它是昂贵的；drone 2 本就要走 10.0，重分配它几乎无额外代价。

> 该几何是**最小机制测试**：刻意让 greedy 与最优的差距大而清晰，以隔离“全局推理”这一
> 单一因素。若通过，再扩展到更接近真实的 N 机 N 目标配置。

## 4. 条件

| 条件 | 含义 |
|---|---|
| R0 | RA only，无恢复（orphan 无人覆盖，作为失效基线） |
| R1 | RA + `RuleMissionPlanner`（greedy 最近机） |
| R2 | RA + 本地 Qwen + validator + rule fallback（异步提交 + 长 TTL，LLM 可指定任意健康机 REASSIGN） |

## 5. 指标

- 主指标：**总路径长度** `path_length_m`（所有健康机完成其最终目标后的累计路程）。
- 次指标：总完成时间（makespan）、orphan 覆盖（G0 是否被到达）、碰撞率、`min_rho`。
- 安全门：所有条件碰撞 = 0；`safety_bypass` = 0。

## 6. 统计

- 30 个 paired seed，R1 与 R2 使用相同 seed 与扰动流。
- 主比较：R2 − R1 的总路径长度配对差，95% bootstrap CI。
- Go 判据：R2 总路径长度显著低于 R1（配对 CI 不跨 0），且安全门全部满足。

## 7. 停止规则

1. 先跑最小 smoke：确认 R1 相对最优确实次优（总路径 ≈ 12.0 vs 6.6），且 R2 至少能
   产生一次被采纳的、指向非最近机的 REASSIGN。
2. smoke 通过 → 冻结 → 全量 30-seed。
3. 全量若不满足 Go 判据 → **冻结负结果**，不调 LLM 参数、不改场景救假设；只记录
   “本地 4B 模型在该重分配场景未表现出可测的全局收益”。

## 8. claim 边界

可写：在冻结的 `reassign_3` 场景与 paired seed 下，本地 LLM 的全局重分配相对 greedy
rule 降低总路径长度（配对照，95% CI）。

不可写：LLM 通用更强、LLM 实时避碰、任意/大规模场景的收益、真实硬件任务分配。任何
超出 `reassign_3` 的表述都视为未经验证。

## 9. 实现缺口（冻结前必须关闭）

1. **完成判据要计入 goal_override**：当前 `run_sim_episode` 的 `reached_base` 只比对
   `base_goals`，被重分配机的最终目标是 override 后的 G0，会被误判为“未完成”。需要
   让完成判据按每机当前有效目标计算，并加单测。
2. **R2 长 TTL 语义**：确认 validator 的 stale 判定用足够大的有效期（≥ LLM 延迟），
   避免静态重分配计划被误判过期。
3. **LLM 输出 schema**：确认 `MissionDecision` 能表达「对非最近机 REASSIGN」且
   `expand_decision` 正确生成带指定 `drone` 与 `waypoint` 的 `REASSIGN`。
4. **场景注册**：`_scenario("reassign_3")` 返回上述几何，`failed_drone=0`、
   `change_step=0`，并正确设置 orphan 覆盖判定。

## 10. 执行顺序（least-cost）

1. 关闭 §9 缺口 + 单测；
2. 最小 smoke（少量 seed）：验证 rule 次优 + LLM 可被采纳；
3. smoke 通过 → 冻结协议 → 全量 30-seed；
4. 统计、按 §7 决定 Go/No-Go、写中文决策文档。
