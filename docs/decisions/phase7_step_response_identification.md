# Phase 7 —— PX4 execution identification checkpoint

> 日期：2026-08-17
> 状态：Step A 首轮完成；经验包络可用于下一轮验证，不是 deterministic bound。

## 1. 数据与协议

使用 `marllib/phase5_step_response.py`，单机 PX4 SITL + Gazebo + MQTT adapter，
目标速度 `vx ∈ {0.5, 1.0, 1.5}` m/s；每个目标速度 3 次，参数固定为
`settle=3 s, run=6 s, stop=6 s`。原始 JSONL 保存在：

`/Volumes/Expansion/aegisair_phase7_step_response_20260817/`

对每条 RUN/STOP 曲线分别在前 3 s 拟合一阶指数；后段振荡不被强行解释为一阶
模型。`r1/r2` 作为 identification，`r3` 作为 held-out。

## 2. 结果

| 目标 vx | RUN tau (r1/r2/r3) | STOP tau (r1/r2/r3) |
| ---: | --- | --- |
| 0.5 | 0.755, 0.736, 0.708 s | 0.727, 0.710, 0.727 s |
| 1.0 | 0.678, 0.698, 0.680 s | 0.733, 0.731, 0.708 s |
| 1.5 | 0.666, 0.757, 0.690 s | 1.461, 0.708, 0.762 s |

`vx=1.5/r1` 的 STOP 长尾是明显异常/振荡现象；它不能被删除，也不能用窄
一阶区间掩盖。其余曲线的拟合 RMSE 随速度升高而增加，说明单一一阶模型只是
execution approximation。

## 3. 当前经验包络与 held-out 检查

基于 `r1/r2` 的拟合值并加约 20% margin，暂定下一轮实验使用：

`T_empirical = [0.53, 1.76] s`

这不是 PX4 的 deterministic guarantee。3 条 held-out `r3` 的 RUN/STOP 拟合值
均落入该包络，包括 `vx=1.5/r3` 的 STOP `0.762 s`。该结论只证明本轮阶跃
样本的 coverage，不能外推到其他场景、负向速度或不同飞控参数。

## 4. lightweight robust-QP 检查

在 `randomized_4/seed1`、`sequential-pass`、AoI=11 ms、`gamma=0.1`、
`tau_ctrl=0.2` 下：

| execution setting | first rho < 0 |
| --- | --- |
| nominal `tau_px4=0.7` | `t=11.05 s`, `rho=-0.0023` |
| robust `tau_px4=0.7, T=[0.53,1.76]` | `t=11.05 s`, `rho=-0.0023` |
| out-of-envelope `tau_px4=2.0` | 300 steps 内未越界，末点 `rho=0.5371` |

因此本轮没有显示 robust interval 能消除该固定轨迹的浅越界，也没有构造出
可靠的 out-of-envelope guarantee breakdown。下一步应扩大 held-out 场景/速度
并记录 interval 是否触发不同的 safe action；在此之前不升级 safety claim。
