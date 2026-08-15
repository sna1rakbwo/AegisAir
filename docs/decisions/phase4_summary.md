# Phase 4 总结（2026-08-16）

> 本文汇总 Phase 4 的最终定位、实现、接口冻结、真实 LLM 基准与三场景结果，
> 并给出论文可 claim 的边界。历史演进见
> `docs/decisions/predictor_architecture.md`、`phase4_recovery.md` 与
> `llm_async_mission_replanning.md`。

## 1. 最终定位

LLM 不是实时避碰，也不是窄义的「repeated CBF recovery」，而是：

```text
CBF / Runtime Assurance  = immediate safety control（同步）
Semantic Mission Manager = asynchronous mission-level replanning（异步）

CBF preserves safety; the LLM preserves mission intent under changing constraints.
Protect now -> Understand later -> Replan future.
```

LLM 不抢方向盘，LLM 改路线图。

## 2. 触发：Mission Validity Monitor

```text
E_replan = E_safety(route deviation) OR E_mission(任务/优先级/环境变化)
           OR E_coord(等待/效率退化)
```

单次 imminent collision 只交给 CBF，不唤醒 LLM。

## 3. 冻结接口（swarm/interfaces.py）

- `RiskEvent.event` 新增 `MISSION_PLAN_INVALIDATED`、`REPEATED_ROUTE_CONFLICT`。
- `RiskCause` 新增 `ROUTE_DEVIATION`、`MISSION_CHANGE`、
  `COORDINATION_DEGRADATION`、`REPEATED_CONFLICT`。
- `RiskEvent` 新增可选 `mission_priority`。
- 新增 `MissionDecision`：compact LLM 高层决策
  `{action, agent?, high?, low?}`。

## 4. LLM 输出协议

LLM 只输出 compact 决策，确定性层展开为完整 `RecoveryPlan`：

```text
LLM JSON (MissionDecision)
    -> schema + semantic validator（agent ∈ active agents）
    -> RecoveryPlan encoder
    -> RecoveryValidator
    -> executor（仍受 CBF 最终否决）
```

四层校验：syntactic / schema / semantic / execution acceptance。任一失败回退
`RuleMissionPlanner`。

## 5. 本地 LLM 部署

- 环境：`eai-swarm`，已装 `mlx 0.32.0` + `mlx-lm 0.31.3`，Metal GPU 可用。
- 模型：`mlx-community/Qwen3-4B-4bit`，约 2.1GB，经 hf-mirror + aria2c
  下载到移动硬盘；SSD 基准副本在
  `/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench`。

## 6. 延迟基准

| 输出 | 延迟 | 说明 |
|---|---|---|
| 短 JSON（max_tokens 24） | total P50 2.83s | TTFT 2.07s / gen 0.76s，valid JSON 1.00 |
| 完整 RecoveryPlan（约 300 token） | 23.3s | 首次 schema 出错（int command_id、[] constraints） |
| compact MissionDecision | 2.77s | 输出 `{"action":"REASSIGN","agent":1}`，四层校验全过 |

结论：让 LLM 自己写完整 schema 既慢又易错；compact 决策 + 确定性展开是正确的
边界。

## 7. 真实 Qwen 三场景结果（10 seeds，real-time）

结果文件：`/Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json`

| 场景 | 指标 | CBF_ONLY | ASYNC (Qwen) |
|---|---|---|---|
| priority_conflict | 高优先级到达步数 | 92.0 | 86.6 |
| | CBF 介入事件 | 72.0 | 66.2 |
| corridor_blocked | zone_cross_rate | 1.0 | 0.0 |
| drone_failure | 关键任务完成率 | 0.0 | 1.0 |

LLM 侧：30 次调用，syntactic/schema/semantic/execution 全部 0 失败，0 timeout，
0 fallback，单次 replan 延迟约 2.6–3.0s。

## 8. Claim 边界

- 2.8s 异步 LLM 对长时程 mission-level 变化有效：失效重分配、航路重规划。
- 对短时程优先级冲突收益温和，不 claim「LLM 实时避碰」。
- LLM 语义推理通常正确，但 schema 细节会出错，因此 strict validator +
  deterministic fallback 是必需，不是可选项。

## 9. 可复现命令

```bash
# 短 JSON 延迟基准
/opt/anaconda3/envs/eai-swarm/bin/python scripts/benchmark_qwen_recovery.py \
  --model "$HOME/.cache/aegisair-qwen3-4b-4bit-bench" --n 100 --warmup 10 --max-tokens 24

# 三场景 mission 对比（真实 Qwen，real-time）
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase4_eval_missions.py \
  --scenarios priority_conflict corridor_blocked drone_failure \
  --seeds 10 --max-steps 120 --modes CBF_ONLY ASYNC \
  --llm qwen --qwen-model "$HOME/.cache/aegisair-qwen3-4b-4bit-bench" \
  --qwen-max-tokens 48 --real-time \
  --output /Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json
```

## 10. 测试

`python -m unittest discover -s tests`：85 个测试全过。
