# MAPPO policy-quality 分层 Smoke 记录

- 日期：2026-08-20
- 协议：`aegisair-policy-quality-smoke-v1`
- 状态：完成；可进入 30-seed validation，不进入 final statistics。

## 冻结对象

- 场景：四机 `multi_uav`。
- Checkpoint：`randomized_4` 的 training seed `1..5`；每个 final checkpoint 的
  SHA-256 记录在原始 JSON 中。
- 独立 evaluation seed：`101..105`，与 training seed 不重叠。
- 质量层：相同 deterministic checkpoint actor 的动作幅度缩放
  `pi*q`，其中 `q∈{0.2,0.5,0.8,1.0}`。
- 条件：S0（不施加 RA，仅观测 margin）与 S6（exact-ZOH interval RA）。
- 每个 `(training seed, eval seed, q, condition)` 运行一次；总计
  `5×5×4×2=200` 条 episode，每条最长 200 steps。

该质量层是一个受控 nominal-authority degradation，**不是**四组独立训练策略，
不得外推为“任意低质量 MARL”的完整行为模型。

## 原始数据

- 代码提交：`7f6dc71cfb806367fcd25d0b397d3fa00323ea74`
- 路径：`/Volumes/Expansion/Aegis/aegisair_policy_quality_smoke_20260820_7f6dc71/policy_quality_smoke.json`
- SHA-256：`756c9348e39f52fdebd714c03c7f8fa5b48835e06a0dc1a8a362083fdfe12821`

## 合并结果

| 质量层 q | 条件 | 碰撞率 | 完成率 | mean min rho |
| ---: | --- | ---: | ---: | ---: |
| 0.2 | S0 | 40% | 0% | -0.669 |
| 0.2 | S6 | 0% | 0% | 0.061 |
| 0.5 | S0 | 40% | 60% | -0.657 |
| 0.5 | S6 | 0% | 20% | 0.057 |
| 0.8 | S0 | 80% | 20% | -0.743 |
| 0.8 | S6 | 0% | 20% | 0.025 |
| 1.0 | S0 | 80% | 20% | -0.774 |
| 1.0 | S6 | 0% | 40% | 0.069 |

## 判断

本 smoke 表明 S6 在这组预定四机条件中能阻止 S0 所出现的碰撞，但任务完成率
没有稳定提升，且当前样本量不足以做显著性或 effect-size 结论。

下一步固定相同 checkpoint、质量层和控制器，不更改阈值或场景，扩展至独立
evaluation seed `101..130` 的 30-seed validation；报告 paired collision /
completion difference、minimum rho 与 intervention burden。若任务完成率仍无改善，
仅声称 safety robustness，不声称 mission recovery 收益。
