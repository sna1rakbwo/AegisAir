# 最小版本 C1 决策：部分支持，不通过主贡献 Gate

- 日期：2026-08-20
- 协议：`aegisair-minimum-c1-v1`，带几何距离日志的审计版本 v2
- 代码提交：`81be89f26991cbf088e9de10dd1a50148d0762e8`
- 样本：4 个四机场景 × 4 个消融 × 50 paired seed = 800 条 episode。

## 产物

- 原始 episode：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v2_20260820_81be89f/c1_minimum_ablation.json`
  - SHA-256：`aaeb889a121a1a387d07b92e62aa7213228fd078ec107d7b1cf2826fe64d84e2`
- Paired bootstrap（10,000 draws）：`/Volumes/Expansion/Aegis/aegisair_minimum_c1_v2_20260820_81be89f/c1_paired_bootstrap.json`
  - SHA-256：`4a3f49ab936fb5003d3e6081abb66657ea2e70e13581440bd5943bf9a0ce0f89`

## 预注册比较

- 固定距离：仅 `d0`，不含 dynamics/perception/AoI margin；
- 完整包络：全部三类 margin；
- 去 perception：`beta=0`；
- 去 AoI：communication margin 为零。

所有条件使用相同场景、seed、sampled-data RA、QP 参数和 fallback；只改变
`d_safe` 中的 margin 项。

## 关键结果

| 场景与比较 | 主要结果 |
| --- | --- |
| randomized：完整 vs 固定 | 碰撞差 -6pp（95% CI -14, 0）；最小几何距离 +0.313 m（0.256, 0.376）；完成率 -32pp（-46, -18）。 |
| dense：完整 vs 固定 | boundary violation 差 -100pp；最小几何距离 +0.716 m；两组完成率均为 0。 |
| perception/dropout：完整 vs 固定 | 碰撞差 -36pp（-52, -20）；最小几何距离 +0.064 m（0.034, 0.095）；完成率 -8pp（-16, -2）。 |
| perception/dropout：完整 vs 去 perception | 碰撞差 -34pp（-48, -20）；最小几何距离 +0.055 m（0.028, 0.084）；完成率 -10pp（-20, -2）。 |
| telemetry delay：完整 vs 去 AoI | 碰撞差 0pp；最小几何距离 +0.604 m；完整包络的自身 `rho` 违规率更高，因为安全边界被抬高。 |

`rho` 是相对于各自安全边界的归一化量，不能跨“完整 / 去 margin”条件直接作为
物理距离收益比较；跨条件的主要安全证据应使用碰撞和最小几何距离。

## Gate 决策

**C1 主贡献 Gate：不通过。**

完整包络在 dense、dropout 与 delay 条件中提高物理间隔，且 perception margin
在 dropout 下有清晰的碰撞降低证据；这支持它作为保守安全表示。可是任务完成率
在 randomized/dropout 下显著下降，dense/delay 均未完成；因此尚不能证明收益并非
主要来自保守介入/停滞。dropout 下完整包络仍有 56% 碰撞和 100% 自身 boundary
violation，不能声称它已在该故障下可靠保证边界安全。

后续写作只能陈述：在本轻量四机测试中，完整包络扩大了几何分离并降低了部分
碰撞风险；其任务-安全权衡和严重 dropout 下的残余失效必须原样报告。不得声称
普适 C1 性能优势，不得通过追加控制器、阈值或 seed 挽救该 Gate。
