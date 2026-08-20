# 陈旧观测一致性诊断 v1：D3 修复 300 ms 延迟碰撞

- 日期：2026-08-20
- 协议：`aegisair-stale-observation-diagnosis-v1`
- 样本：3 场景 × D0--D3 × 50 paired seed = 600 条 episode。
- 原始产物：`/Volumes/Expansion/Aegis/aegisair_stale_observation_diagnosis_v1_20260820/d0_d3.json`
  - SHA-256：`4b51247e797ca90d40bdeb298f2205244154f18a8ebfc73dfe5def0254d83dca`

## 问题与边界

C1 v4 的 300 ms 陈旧遥测将所有无人机（包括本机）送入延迟估计器；同时名义
go-to-goal 使用当前环境位置。该结果保留不变。本诊断是新协议，目标是区分：
碰撞是否来自名义/RA 观测不一致，还是 300 ms peer 状态本身不可恢复。

不使用本地 LiDAR/视觉，不增加 `M_comm`，不修改 C1 v4。

## 冻结的 D0--D3

| 组别 | nominal 可见状态 | RA 可见状态 |
| --- | --- | --- |
| D0 | 当前共享状态 | 当前共享状态 |
| D1 | 当前状态 | 所有状态均为 300 ms 陈旧估计（旧实现） |
| D2 | 所有状态均为同一 300 ms 陈旧估计 | 同一 300 ms 陈旧估计 |
| D3 | 本机新鲜；普通 go-to-goal 不读取 peer | 每架无人机独立：本机新鲜、peer 为 300 ms 陈旧并 dead-reckon |

RA、exact-ZOH 参数、场景、seed 和 CBF-only 模式相同。D3 的每架无人机都有独立
RA tracker；本机状态为 covariance-zero 的当前机载状态，peer 保留估计器的时间戳、
协方差和 AoI。

## 结果

| 场景 | D0 碰撞 | D1 碰撞 | D2 碰撞 | D3 碰撞 | D3 平均最小间距 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 直线对向 | 0% | 0% | 0% | 0% | 1.488 m |
| crossing | 0% | 100% | 100% | 0% | 1.090 m |
| 稠密四机 | 0% | 100% | 100% | 0% | 0.760 m |

D2 没有改善 D1，说明仅让 nominal 与 RA 一起使用“所有状态均陈旧”的接口不够。
D3 将 crossing/dense 的碰撞从 100% 降至 0%，同时 CBF intervention 恢复到与 D0
同量级。这将根因定位为：**本机状态被错误地陈旧化，导致安全闭环以过时的自身几何
和速度工作**，而不是 scalar communication margin 不足。

crossing/dense 的 completion 仍为 0：这是 CBF-only 对称交叉的协调死锁，不计作
安全修复失败；任务恢复仍属于 C3。

## 决策

后续含通信延迟的即时避碰实验必须使用 D3 观测架构：每个本机 RA 读取新鲜本机状态，
只对 peer 使用延迟/丢包/协方差估计；对应 nominal 与 RA 使用同一套本机可见信息。
不得再把本机与 peer 状态一同延迟，也不得把 D1/D2 的 100% 碰撞归因于 `M_comm`
不足。

这只验证 300 ms、此轻量模型和三个冻结场景。若未来 D3 在更高延迟或转向不确定性
下失败，再新建协议评估 peer reachable-set 与 degraded mode；abort 必须计入任务失败。
