# S0/S1/S2/S3/S6 全链路 Smoke 记录

- 日期：2026-08-20
- 协议：`aegisair-baseline-smoke-v1`
- 状态：通过接口与全链路 smoke；不进入 validation 或 final statistics。

## 目的和冻结范围

本次完成正式计划第 3 步的最小接口闭合，并执行第 4 步 smoke。控制条件为：

- S0：无 safety filter，仅观测并记录 RA 裕度；
- S1：静态速度 CBF-QP；
- S2：acceleration HOCBF-QP；
- S3：sampled-data/ZOCBF-style reference，`tau_px4=0`；
- S6：exact-ZOH execution-consistent RA，`tau_hat=0.7 s`，区间 `[0.53, 1.76] s`。

每个条件使用相同的 seed `1..5`，分别运行二机 `head_on` 与四机
`multi_uav`，每条 episode 最多 120 步；总计 50 条 episode。所有条件经过
同一 `MultiUAVEnv -> RA -> adapter command validation -> environment` 链路。

## 原始数据

- 运行时基线提交：`cd1186e54762f6c4032c59377c4c522b2b6bcc0c`；本 smoke runner
  在产物校验后被纳入后续提交。运行时脚本 SHA-256：
  `ded1a2d13a7711c6e2b9ed1aac34acd9e27f334a615c599f4ab3660a73cda737`。
- 路径：`/Volumes/Expansion/Aegis/aegisair_baseline_smoke_20260820_cd1186e/paired_baseline_smoke.json`
- SHA-256：`78f7570d1f39ec79b8912c9473dd2c0e09cba9e769a9843b2b12fddfb60d9c7e`

## 结果

| 场景 | 条件 | 碰撞率 | 完成率 | 最低 rho | 平均 CBF 介入 | adapter 拒绝 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| head_on | S0 | 0/5 | 5/5 | -0.479 | 0.0 | 0 |
| head_on | S1 | 0/5 | 5/5 | 0.508 | 76.0 | 0 |
| head_on | S2 | 0/5 | 5/5 | 0.805 | 110.0 | 0 |
| head_on | S3 | 0/5 | 0/5 | 2.216 | 158.0 | 0 |
| head_on | S6 | 0/5 | 5/5 | 0.342 | 78.0 | 0 |
| multi_uav | S0 | 5/5 | 0/5 | -0.871 | 0.0 | 0 |
| multi_uav | S1 | 0/5 | 5/5 | -0.047 | 282.0 | 0 |
| multi_uav | S2 | 0/5 | 0/5 | 0.001 | 480.0 | 0 |
| multi_uav | S3 | 0/5 | 0/5 | 0.731 | 480.0 | 0 |
| multi_uav | S6 | 0/5 | 0/5 | 0.078 | 472.0 | 0 |

## 判读与边界

通过：每个预定条件都有独立控制器实例、5 个 paired seed、统一输出字段与完整
原始 episode 记录；adapter rejection 均为 0。S0 在 `multi_uav` 的 5/5 碰撞也
确认无安全过滤基线具有预期风险。

不作正式效力结论：样本量仅为 smoke；S3/S6 在部分场景的 0 完成率、S1 的负
`rho`、以及各方法间的任务差异必须在后续 validation 中以冻结 checkpoint、
更长 horizon、paired bootstrap 和 failure audit 解释。不得据此筛选方法、调参或
宣称 S6 优于其他 baseline。

## 后续顺序

第 3、4 步完成后，下一项是第 5 步：5 个独立 MAPPO training seed 与 policy-quality
分层；final test seed 必须独立于训练 seed。
