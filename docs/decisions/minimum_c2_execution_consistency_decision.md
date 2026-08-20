# 最小版本 C2 决策：执行一致性验证未通过

- 日期：2026-08-20
- 执行基线：C1 v2 完成后的提交 `e0fbfe9`。
- 状态：**C2 No-Go**。保留 execution model 与 one-step feasible-QP 证据，但不宣称 exact-ZOH 在 episode 级优于 nominal execution model。

## 已核验的执行标定

PX4 SITL/Gazebo 单机阶跃原始日志保存在
`/Volumes/Expansion/Aegis/aegisair_phase7_step_response_20260817/`。目标速度为
0.5、1.0、1.5 m/s，每个速度三次；r1/r2 用于 identification，r3 是 held-out。
RUN 与 STOP 的 held-out 一阶时间常数均落在经验区间 `[0.53, 1.76] s` 内。
这是本批命令下的 empirical coverage，不是全局确定性 bound。

## 本轮冻结复验

使用相同的 two-UAV high-closing-speed 机制场景：hidden `tau_true` 在
`[0.53, 1.76] s` 内 paired 抽样，nominal 模型取 `tau_hat=0.7 s`，exact-ZOH
模型对整个经验区间做 vertex-robust QP。初始距离为 6、8、10 m，各 50 paired
episode，每条最多 240 steps。

- 原始 episode：`/Volumes/Expansion/Aegis/aegisair_c2_v1_20260820/two_uav_execution_comparison.json`
  - SHA-256：`9b18b8c6d08d93d79ab6b786a3eabc100867fbfbe92743aabc141963d50a8b2f`
- one-step 100×100 tau-grid 审计：`/Volumes/Expansion/Aegis/aegisair_c2_v1_20260820/one_step_feasibility_audit.json`
  - SHA-256：`60cb3ed046f332e5d317b4258d697223ef1e23976ee89df47892c331759082cd`
- feasible-QP 与 fallback 重放：
  `feasible_qp_replay.json`（`9fbbc669…`）和 `infeasible_fallback_replay.json`（`f1f59341…`），均位于同一目录。

## 结果

| 初始距离 | nominal tau：碰撞 / 边界违规 | exact-ZOH interval：碰撞 / 边界违规 |
| ---: | ---: | ---: |
| 6 m | 100% / 100% | 100% / 100% |
| 8 m | 100% / 100% | 100% / 100% |
| 10 m | 66% / 80% | 66% / 80% |

在 one-step 扫描中，interval 会改变 225 个 activation case 的安全决策；其中 44 个
robust QP 可行，且 44/44 通过 100×100 tau-grid 的 projected barrier 与 squared-h
检查。其余 181 个 robust QP 不可行，进入 deterministic hard-brake fallback。

这两类结果必须分开：前者只支持“在可行一步，interval 影响安全动作”；后者不能被
包装成 QP guarantee。20-step 重放进一步显示，起点的一步可行并不推出后续 recursive
feasibility：下一步即可转为 infeasible/fallback，且随后出现 `rho < 0`。

## 决策

1. **执行模型标定：通过 coverage 检查。** exact-ZOH 的经验 tau 区间有 held-out 阶跃支持。
2. **C2 主效力：不通过。** exact-ZOH interval 没有在 high-closing-speed episode 中优于 nominal tau；两者均大量碰撞或边界违规。
3. **不扩大 Gazebo 统计来掩盖失败。** 主比较已失败，且当前缺陷是 recursive feasibility 与 fallback 后闭环可达性，不是增加样本能解决的随机不确定性。因而不把 endpoint 结果夸大为连续时间保证，也不声称 C2 通过。

后续可继续 C3：在 C1 已证实安全的本机新鲜/peer 陈旧观测架构上，评估 SEQUENTIAL_PASS 的任务恢复；C2 的 No-Go 作为执行模型的限制性结果保留。
