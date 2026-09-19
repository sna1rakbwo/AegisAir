# 论文实验实现复审（2026-09-20）

## 范围与原则

本审查不改动任何冻结 manifest、种子、场景、指标定义或历史结果。它检查论文当前主张所依赖的实现、封存 summary、轨迹哈希和由轨迹重算的逐步指标。

## 已确认的实现问题

### P1：Dykstra 终止条件曾可过早停止

`swarm/ra/sota_cbf.py`、`swarm/ra/hocbf.py` 的主投影，以及 `swarm/ra/pcbf.py` 的 Gate-0 终端投影此前仅用相邻两轮的原始投影点差值判断收敛。Dykstra 的校正项仍可变化，而原始点在中间若干轮暂时不变；此时停止会返回可行但非最小距离的投影。

一个二维、四半空间的确定性反例中，旧停止规则在第 2 轮停止，输出与枚举 QP 最优点的最大坐标差为约 0.094；继续迭代后在第 10 轮收敛。修复要求原始点变化和所有校正项变化均不超过同一容差。此前已发现的“每轮从名义点重置”问题也保留修复。

这不是“可行集合为空”的错误报告；它可以改变所选动作、干预和轨迹，并且在某些输入上可与回退逻辑共同改变实验结果。

## 修复验证

- 两个确定性回归反例加入 `tests/test_sota_cbf.py`：保留迭代点，以及校正项尚在变化时不得停止。
- 对 200 个随机构造、保证可行的二维/四维 box--halfspace QP，以活动约束枚举的欧氏投影为 oracle 核对；最大坐标误差为 `4.31e-9`。
- `conda run -n eai-swarm python -m unittest discover -s tests`：256 项通过。
- 同一封存 seed `v4val01` 在修复后重跑 Hybrid、Reactive PB 和 PB-CBF。三条均 mission complete、无 collision、selected-QP infeasible 为 0，说明该小样本未出现安全门槛翻转；轨迹指标存在差异，不能据此断言旧数值不受影响。

复跑数据位于：

- `/Volumes/Expansion/Aegis/c1_dykstra_projection_patch_smoke_20260920_convergence_hybrid`
- `/Volumes/Expansion/Aegis/c1_dykstra_projection_patch_smoke_20260920_convergence_v1`

## 封存结果一致性核对

对下列论文主线 campaign 的每一条 summary，检查 trajectory SHA-256、逐步最小 `rho` 与 summary 的 `min_rho`、逐步 `feasible=false` 计数与 summary 的 `qp_infeasible_steps`。共 360 条 episode，均未发现不一致。

| Campaign | 条件数 | 结果 |
| --- | ---: | --- |
| C1 主比较 | 80 | 通过 |
| C1 reserve/prediction 消融 | 80 | 通过 |
| C1 matched PCBF 比较 | 50 | 通过 |
| C2 execution sensitivity | 30 | 通过 |
| C3 mission recovery | 60 | 通过 |
| recoverability admission | 60 | 通过 |

独立重跑的分析还确认：C1 消融为 80/80 完整且 Go；C2 为 30/30 完整且 Go；四机 GroupSlot 为 20/20 Go。论文图表生成器重新读取并哈希核验了 150 条 external-comparison 轨迹和 20 条四机轨迹，重算的 PCBF 配对统计和四机时延与正文一致。

## 对投稿证据的结论

封存日志和论文表格之间没有发现统计、计数或文件完整性错误；这支持“历史记录被正确汇总”。但这些检查不能证明历史可执行代码没有上述 Dykstra 终止缺陷。补充材料已经说明当前源码不是每个历史 campaign 的精确快照，因此当前无法仅凭日志判定缺陷是否进入所有历史运行。

在获得历史源码哈希或按原冻结配置完成受影响 campaign 的重跑前，不应把 C1、C2、C3 与四机实验的具体效应量视为已完成修复后的最终数值。论文可以保留已验证的边界性表述和历史结果来源，但投稿前应把该问题作为 P1 处理：固定全部配置、种子和停止规则，重跑受影响的主线条件并重新生成所有统计表；不得调参或选择性替换结果。

## 未发现的同类问题

当前 HOCBF 的 robust sampled-data 和 one-step sampled-data 投影执行固定轮数，不使用该过早终止条件。PCBF PX4 外部比较实际使用的是有限候选的欧氏距离/终端恢复适配；`pcbf.py` 中受修复影响的 projected-subgradient Gate-0 参考路径并非该 PX4 比较的主求解路径。
