# 最小版本 C1 v2 决策：一致观测架构下的安全包络

- 日期：2026-08-20
- 协议：`aegisair-minimum-c1-v2-observation-consistent`。
- 样本：50 个 paired seed（201--250）× 4 个场景 × 4 个消融 = 800 条 episode。
- 核心架构：每架无人机的本机状态保持新鲜；peer 状态经 300 ms 延迟估计器进入 nominal pilot 和本机 RA。二者使用同一可见信息。

## 产物与审计

- 原始 episode：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v6_20260820_observation_consistent/c1_minimum_ablation.json`
  - SHA-256：`c6e21d278661121b126106192112deaec0341a111d0e3ead8a6d9789585ede53`
- paired bootstrap（10,000 draws）：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v6_20260820_observation_consistent/c1_paired_bootstrap.json`
  - SHA-256：`04aae69354c34f3b5849c7570fad2a3f01ba346bfdb5202100dc2f4248b2e2ce`

v4 的全局陈旧状态结果原样保留，不能被覆盖：它揭示了把本机状态也延迟会造成架构性失败。
本次 v2 仅改变观测接口，不改变 C1 的 seed、阈值、协调策略或安全 margin 公式。v5 曾将无估计器的
raw snapshot 时间戳 `0` 误作陈旧 peer；已修复且 v5 不用于任何结论。

## 结果

| 场景与比较 | 完整包络结果 |
| --- | --- |
| randomized：完整 vs 固定 | 碰撞 6% → 0%；boundary violation 26% → 4%；平均最小几何距离 0.826 → 0.929 m（paired +0.104 m，95% CI 0.066, 0.144）。完成率 78% → 74%。 |
| dense：完整 vs 固定 | 物理碰撞均为 0%；boundary violation 100% → 0%；最小几何距离 0.280 → 0.610 m（paired +0.330 m）。 |
| dropout：完整 vs 固定 | 碰撞 100% → 0%；最小几何距离 0.239 → 0.629 m（paired +0.390 m，95% CI 0.353, 0.426）。 |
| dropout：完整 vs 去 perception | 碰撞 100% → 0%；最小几何距离 +0.390 m（95% CI 0.352, 0.425）。感知 margin 具有独立作用。 |
| 300 ms telemetry：完整 vs 固定 | 碰撞 100% → 0%；最小几何距离 0.241 → 0.760 m（paired +0.519 m）。 |
| 300 ms telemetry：完整 vs 去 AoI | 碰撞均为 0%；最小几何距离 0.288 → 0.760 m（paired +0.472 m）。AoI margin 提供实质安全余量。 |

`rho` 相对于各自的 `d_safe` 归一化，不能跨消融作为物理距离比较。强故障场景仍有
`rho < 0`，即系统虽未物理碰撞但未保持其保守安全边界；该事实必须报告。

## 决策与主张边界

1. **C1 通过。** 在一致观测接口下，完整包络相对固定距离及关键消融改善相应安全指标；它不是只靠停住降低碰撞，因为几何最小间距系统性提高。
2. **架构结论。** v4 的 100% telemetry collision 是“将本机状态延迟”而不是 margin 大小造成的闭环不一致；本机新鲜、peer 陈旧的分布式接口修复了该失败。
3. **不作过度主张。** dense、dropout、telemetry 三个压力场景的 CBF-only completion 为 0%，这是对称冲突下的安全死锁；C1 不将完成率列为 Gate，任务恢复由 C3 评估。`rho < 0` 也禁止宣称连续时间硬安全保证。

下一阶段进入 C2：验证 exact-ZOH 执行模型、intersample 审计，以及 QP 与 hard-brake fallback 的边界。
