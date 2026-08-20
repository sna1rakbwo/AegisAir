# 最小版本 C1 决策：感知包络部分通过，真实陈旧遥测子线失败

- 日期：2026-08-20
- 协议：`aegisair-minimum-c1-v1`；50 个 paired seed，4 场景 × 4 消融 = 800 条 episode。
- 正式产物使用校正后的 v4；没有改阈值、seed 或协调策略。
- 样本：4 个四机场景 × 4 个消融 × 50 paired seed = 800 条 episode。

## 产物

- 原始 episode：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v4_20260820_stale_state/c1_minimum_ablation.json`
  - SHA-256：`7f64621bd76030fb74ac604626b21f45a8a1d722670892d986e6d50dabc2ee48`
- Paired bootstrap（10,000 draws）：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v4_20260820_stale_state/c1_paired_bootstrap.json`
  - SHA-256：`6098e896afe34a24bbfdb88d65dd5d3e4a793d588d349a8c2b0166c3cf54dab2`

## 校正与比较边界

- 固定距离：仅 `d0`；完整包络：dynamics、perception、AoI 三类 margin；去 perception：`beta=0`；去 AoI：communication margin 为零。
- 干净快照缺协方差时曾回退为 `perception_sigma=0.1`，凭空增加 0.424 m 常数 margin；估计器协方差也没有匹配注入的 0.2 m 噪声，且模拟路径漏接 dead-reckon。三处均已校正并由单测覆盖。
- v3 的 `telemetry_delay` 只增加 AoI 数字、仍向 RA 泄露当前真值。v4 改用 `estimator_delay_ms=300`，RA 消费真实陈旧估计并前推到控制时刻。v2、v3 仅保留审计，不能用于结论。

## 校正后结果

| 场景与比较 | 安全结果 |
| --- | --- |
| randomized：完整 vs 固定 | 碰撞 6% → 0%；boundary violation 26% → 4%；平均最小几何距离 0.826 → 0.929 m（配对 +0.104 m，95% CI 0.066, 0.144）。 |
| dense：完整 vs 固定 | 碰撞均为 0；boundary violation 100% → 0%；距离 0.280 → 0.610 m。CBF-only 对称交叉的安全死锁，完成率不是 C1 Gate。 |
| dropout：完整 vs 固定 | 碰撞 90% → 44%（配对 -46pp，95% CI -64, -26）；距离 0.231 → 0.402 m（+0.171 m，0.116, 0.230）。 |
| dropout：完整 vs 去 perception | 碰撞 90% → 44%（-46pp，-62, -28）；距离 +0.163 m（0.109, 0.222），表明 calibrated perception margin 有独立安全贡献。 |
| 真实 300 ms stale telemetry：完整 vs 固定/去 AoI | 四组碰撞、boundary violation 均为 100%。完整包络距离 0.227 m（固定 0.191 m；去 AoI 0.243 m），没有可靠安全收益。 |

`rho` 相对于各自 `d_safe` 归一化，不能跨消融直接解释为物理间距；跨条件以碰撞率和
几何最小距离为主。C1 不将 completion 作为 Gate：任务完成/死锁恢复属于 C3 的协调
评估，但 completion 仍保留在原始日志供后续任务实验使用。

## 决策

1. **感知与动态包络子贡献：通过。** 在 randomized、dense 和 30% dropout 中，完整包络提高物理分离；dropout 下碰撞相对固定距离及去 perception 各降低 46pp。
2. **强 dropout 的可靠安全保证：不通过。** 完整包络仍有 44% 碰撞和 100% 自身 boundary violation，不能声称硬安全保证。
3. **真实陈旧遥测子线：不通过。** 修复真值泄漏后，300 ms delay 下所有 C1 条件均碰撞；full 与去 AoI 的比较也没有显示 AoI 项在该几何中带来收益。

允许的写作结论是：校准后的完整包络在随机、稠密和感知 dropout 条件下具有可重复的
物理分离与部分碰撞降低收益；它不能覆盖强 dropout，也不能在真实 300 ms 陈旧遥测下
保证安全。C2 应针对执行/陈旧状态一致性解释并验证这一残余；不得删除或弱化这条失败记录。
