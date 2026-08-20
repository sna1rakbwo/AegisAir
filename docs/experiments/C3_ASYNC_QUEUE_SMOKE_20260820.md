# C3 异步队列（立即回退）Smoke 记录

日期：2026-08-20  
性质：探索性 smoke，不是冻结实验，不覆盖 `minimum_c3_v1_recovery_result.md` 的结论。

## 问题

冻结 C3 v1 的 R2 在 `priority_conflict` 中，本地 Qwen 推理约 2.8 s，超过
`RecoveryPlan` 1 s 有效期，导致计划过期被拒后回退 rule；但回退前系统已同步
等待约 2.56 s，期间安全裕度被压到 `rho=-0.536`。本 smoke 验证：改成「异步队列」
——触发恢复时不等待 LLM、立即提交确定性 rule 回退——能否消除这一安全退化。

## 改动

`AsyncMissionReplanner` / `ReplanConfig` 新增 opt-in 开关
`immediate_fallback`，默认 `False`（保持冻结行为不变）。开启时，`_submit` 在触发
恢复后立即提交 rule fallback，LLM 仍在后台生成、仅作为后续触发的前瞻候选。

- `swarm/recovery/async_replanner.py`：`ReplanConfig.immediate_fallback` + 立即提交路径
- `marllib/phase5_runner.py`：`run_sim_episode(..., immediate_fallback=...)` 透传
- `tests/test_async_replanner.py`：新增单测（145 项测试全过）
- `marllib/run_c3_async_queue_smoke.py`：本 smoke 复现脚本

## Smoke 设置

- 场景：`priority_conflict`，3 个 seed，`max_steps=150`，`real_time=True`，`execution_tau_s=0.2`
- LLM 替身：`DeterministicRecoveryClient(plan_latency_s=2.8)`（复现本地 Qwen 约 2.8 s 延迟，不加载真实模型）
- fallback：`RuleMissionPlanner`

## 结果

| 条件 | min_rho(3 seed 最差) | recovery_step | 碰撞 | 完成 |
| --- | ---: | ---: | ---: | ---: |
| `immediate_fallback=False`（冻结行为） | -0.036 | 21 / 21 / 22 | 0/3 | 0/3 |
| `immediate_fallback=True`（异步队列） | +0.816 | 0 / 0 / 0 | 0/3 | 3/3 |

## 结论与边界

1. 异步队列把回退从「等待约 2.1 s（本 smoke 的确定性替身）」提前到触发当拍，`min_rho`
   由负回正（-0.036 → +0.816），且 `priority_conflict` 从 0/3 完成恢复到 3/3。
2. 这与冻结 C3 v1 的方向一致：R1（rule 即时执行）本来就 `rho=+0.857`、100% 完成；
   异步队列让 R2 不再因等待慢 LLM 而比 R1 更差。
3. 本 smoke 不能宣称「本地 LLM 带来额外任务收益」：异步队列只是消除了 LLM 延迟的
   拖累，并没有让 LLM 计划被采纳。若要让 LLM 真正增值，还需单独解决 pre-alert 提前量
   （0.175 s）远小于 LLM 延迟（2.8 s）的矛盾。
4. 冻结 C3 v1 结论原样保留；本改动为独立探索，若要转成正式实验需另行冻结协议并全量
   30-seed 复跑。
