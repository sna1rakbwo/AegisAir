# 最小版本 C2 v2：exact-ZOH 执行一致性结果

> ⚠️ **参数勘误（2026-08-21）**：本结果使用的 `tau_px4=0.2 s` / `tau_ctrl=0.2 s` 是
> 错误的。Phase 7 阶跃识别实测的 velocity-tracking tau 是 `τ_hat=0.7 s`（经验区间
> `[0.53, 1.76] s`），`tau_ctrl < 0.1 s`。0.2 与 0.7 差 3.5 倍，且本结果 plant 与
> 控制器同为 0.2（自证），故在真实 PX4 上不可复现；正确值应以 `tau_px4=0.7`、
> `tau_ctrl=0.1` 重跑。本文保留为历史记录，不用于最终结论。

- 日期：2026-08-20
- 代码提交：`1d31eaf`
- 状态：确定性执行矩阵与 100 点 intersample 审计已完成；旧 C2 v1 的 robust-`tau`
  旁路审计仍原样保留，不能替代本结果。

## 修正后的实验对象

此前 lightweight runner 把 controller 的 `tau_px4` 传给 RA，却仍使用瞬时
`env.step` 作为真实 plant。因此它只能比较不同预测器，不能检验执行一致性。

v2 新增精确一阶 ZOH plant：

```text
v(k+1) = v(k) + alpha * (u - v(k))
p(k+1) = p(k) + v(k)*dt + beta*(u-v(k))
```

其中 `tau=0.2 s`、`tau_ctrl=0.2 s`、`dt=0.05 s`；加速度上限仍按场景的
`3.0 m/s²` 在 plant 内执行。三种 controller 面对的是同一个 lagged plant 和相同
nominal 输入：

| 条件 | controller 内部执行模型 |
| --- | --- |
| E0 | 瞬时执行假设，`tau_px4=0` |
| E1 | 历史 `a_nom=k_v(v_nom-v)` + trapezoidal 近似 |
| E2 | AegisAir nominal exact-ZOH projected barrier，`tau_px4=0.2 s` |

这恢复了综合计划中已经冻结的 exact-ZOH 加速度投影公式；没有更改阈值、seed、
场景几何或协调策略。

## 2 机 high-closing 配对矩阵

20 个 paired seeds（101--120），每 episode 120 steps；每个控制周期以 100 个内部
点重建 exact-ZOH 轨迹。

| 条件 | collision | endpoint min rho | intersample min rho | intersample min distance | QP infeasible steps / 120 | QP P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0/20 | +3.149 | +2.348 | 3.299 m | 0 | 23.94 ms |
| E1 | 0/20 | +0.798 | **-0.0012** | 0.958 m | 94 | 12.55 ms |
| E2 | 0/20 | +1.038 | **+0.383** | 1.318 m | 99 | 23.55 ms |

E2 相对历史近似 E1 的关键差异发生在控制周期内部：E1 的 projected barrier
`-0.00115`、squared barrier `-0.00220` 均越界；E2 分别保持 `+0.3647` 与
`+0.8283`。因此 E2 并非只是在 endpoint 上掩盖 intersample 风险。

E0 在同一真实 lagged plant 上停得更远，因而数值裕度最大；它是过于保守的
瞬时假设反事实，不是执行一致性的优势。C2 的可辨识主结论是 E2 相对 E1：在相同
实际执行滞后下，exact-ZOH 将 dense audit 从负裕度恢复为正裕度，并提高最小物理
间距约 0.36 m。

## 4 机关键审计

`multi_uav` 场景，E2，10 seeds（201--210），120 steps、每周期 100 内部点：

| 指标 | 结果 |
| --- | ---: |
| collision | 0/10 |
| endpoint min rho | +1.404 |
| intersample min rho | +0.301 |
| intersample min distance | 1.203 m |
| projected / squared barrier 最小值 | +0.278 / +0.592 |
| QP infeasible steps / 120 | 119 |
| QP P99 | 97.46 ms |

该场景不启用 `SEQUENTIAL_PASS`，所以 completion=0 是对称穿越的安全死锁，不能
作为 C2 任务指标；任务恢复归 C3。QP 不可行步和 hard-brake fallback 已单列，
没有从统计中删除。

## 可写结论与边界

可写：在已识别的 `tau=0.2 s` 一阶执行模型、这组 2/4 机 deterministic 场景和
100 点 intersample audit 范围内，exact-ZOH projected barrier 比历史启发式执行
近似保持更大的物理分离，并消除了该 high-closing 测试中的 intersample boundary
violation。

不可写：连续时间全局保证、任意 PX4 时间常数保证、或在真实飞控上的 20 Hz deadline
保证。已有 PX4 SITL/Gazebo RUN/STOP held-out 阶跃日志只提供 `tau` 的 empirical
coverage；其目录和边界见 `minimum_c2_execution_consistency_decision.md`。

## 可复现物

- 2机结果：`/Volumes/Expansion/Aegis/aegisair_c2_v8_20260820/c2_high_closing_paired20.json`
  - SHA-256：`636c4fe9f3fed0397ed813d903a79605d0f476b59e70228ed3ff43f706567a5f`
- 4机审计：`/Volumes/Expansion/Aegis/aegisair_c2_v8_20260820/c2_multi_uav_e2_audit10.json`
  - SHA-256：`7d3da6a7342472d405677bf4f3344daeff20cfac417dc844897334b54da0ec97`

两份 artifact 保存了每步开始状态、命令、加速度限幅后的有效命令与 intersample
summary，可在不读取 git 工作区的情况下重建审计。
