# 最小版本 C2 v1 审计：非主比较，不能用于最终结论

- 日期：2026-08-20
- 状态：已撤回为 C2 主实验的解释。

正式 C2 v2 的执行一致性矩阵与 dense audit 见
`docs/decisions/minimum_c2_v2_exact_zoh_result.md`；本文件只保留 v1 旁路审计的
来历和边界。

## 原因

这轮运行比较的是 nominal-\(\tau\) 与 robust-\(\tau\) interval 的 episode-level
压力测试。它没有包含 C2 所需的三种执行模型：ideal/instantaneous、历史启发式和
nominal exact-ZOH。因此其碰撞率不能回答 exact-ZOH 是否优于 ideal/heuristic。

此前将它写为 C2 No-Go 是错误的解释，不是实验结论。

## 保留的旁路审计

- 阶跃 held-out：PX4 SITL/Gazebo 的 0.5、1.0、1.5 m/s RUN/STOP 原始日志位于
  `/Volumes/Expansion/Aegis/aegisair_phase7_step_response_20260817/`；r3 的拟合
  tau 落在经验区间 `[0.53, 1.76] s` 内。这是 empirical coverage，不是全局 bound。
- robust-\(\tau\) one-step grid：
  `/Volumes/Expansion/Aegis/aegisair_c2_v1_20260820/one_step_feasibility_audit.json`
  （SHA-256 `60cb3ed046f332e5d317b4258d697223ef1e23976ee89df47892c331759082cd`）。
  225 个 activation case 中，44 个 robust QP 可行且通过 100×100 tau-grid 审计；
  181 个不可行并进入 hard-brake fallback。

这些结果只作为 C2 的参数不确定性与 fallback 边界审计；它们不代替主比较。

## 正确的后续实验

正式 C2 将在统一的 exact-ZOH 实际执行 plant 下重跑：

1. E0：ideal/instantaneous execution assumption；
2. E1：历史启发式执行近似；
3. E2：AegisAir nominal exact-ZOH execution model；
4. 对每个控制周期重建至少 100 个内部点，并单独报告 feasible-QP 与 fallback。

只有这组统一 plant、统一 nominal 输入的对比才用于 C2 最终结论。
