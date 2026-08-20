# Phase 7 —— two-UAV tau-uncertainty activation

> 日期：2026-08-17
> 目的：隔离验证 PX4 execution interval 是否能改变 one-step safety decision。

## 1. 固定实验对象

使用 `marllib/robust_activation_search.py`，不启动 PX4、不跑四机 episode。状态为
一架继续前进、另一架实际仍有速度但 nominal command 为零的 moving-versus-yielding
配置。扫描 `d=d_safe+delta`、`delta∈[0.05,1.0] m`、`v_cl∈[0.2,1.4] m/s`，
`dt=0.05 s`、`gamma=0.1`、`T=[0.53,1.76] s`、`tau_hat=0.7 s`。

当前 QP 使用 projected barrier，因此 activation 判据明确写成：

```text
G_nominal = G_projected(tau_hat, tau_hat; D_next_nominal)
G_robust  = min G_projected(tau_i, tau_j; D_next_robust)
```

`D_next_robust` 是四个 tau rectangle vertices 的最大下一拍安全边界；这一步修正
了原实现只 robustify `beta`、却仍使用 nominal `s_next` 的缺口。

## 2. one-step search 结果

共找到 225 个 activation case：

```text
G_nominal >= 0
G_robust  < 0
delta_u > 0
```

代表性 case（该 case 的 robust QP 可行）：

```text
delta = 0.45536 m
v_cl = 0.90 m/s
yielding actual speed = 0.10 m/s
G_nominal_projected = +1.0e-6
G_robust_projected   = -1.6267e-4
Delta u = 2.1745 m/s^2
```

nominal acceleration 为 `{0:[0,0], 1:[-0.2,0]}`；robust acceleration 为
`{0:[-2.0,0], 1:[1.9745,0]}`，两者均在 QP 中 feasible。

原始搜索输出保存在：

`/Volumes/Expansion/aegisair_phase7_step_response_20260817/robust_activation_search.json`

## 3. dense-grid 审计、可行性与 claim boundary

对全部 225 个 case 额外计算 `50×50=2500` 个 tau 点：

```text
robust feasible cases       = 44
robust infeasible cases     = 181
projected dense gate passed = 44 / 44 feasible cases
squared-h dense gate passed = 44 / 44 feasible cases
```

这 44 个 feasible case 的 projected margin 最差仅为数值误差量级
`-1.1e-16`，exact squared-h 全部非负。其余 181 个在 conservative
`D_next` 下本身不可行，返回 hard-brake fallback；这不是 dense audit 失败，
但也不能把它们纳入 Proposition 的 feasible-step guarantee。

因此当前证据支持的是：

> identified tau interval changes the projected one-step safety decision and
> produces a different feasible acceleration.

它还不支持：

> robust QP alone guarantees the original squared-distance h inequality under
> all tau values.

原因是 Proposition 必须显式带上 one-step robust QP 的 feasibility assumption；
它不是对 infeasible fallback 的保证。20-step controlled replay 也显示 knife-edge
case 会进入 infeasible/hard-brake 区域；不能把它报告成完整 episode safety
improvement。

## 4. 可复现实验

```bash
python marllib/robust_activation_search.py \
  --tau-min 0.53 --tau-max 1.76 --tau-hat 0.7 \
  --dense-points 50 --top-k 225 \
  --out /Volumes/Expansion/aegisair_phase7_step_response_20260817/robust_activation_search_all.json
```

固定 case 的 controlled replay：

```bash
python marllib/robust_activation_episode.py \
  --search /Volumes/Expansion/aegisair_phase7_step_response_20260817/robust_activation_search.json \
  --case-index 0 --steps 20 \
  --out /Volumes/Expansion/aegisair_phase7_step_response_20260817/robust_activation_episode_20.json
```

## 5. E3：two-UAV episode-level benchmark

随后按 `d_start={6,8,10} m`、每个距离 50 个 episode、同一 episode 的 hidden
`tau_true∼U[0.53,1.76]` 配对比较 nominal `tau_hat=0.7` 与 robust interval。
结果文件：

`/Volumes/Expansion/aegisair_phase7_step_response_20260817/robust_episode_benchmark_50.json`

| d_start | controller | violation | collision | mean min rho | QP infeasible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 m | nominal | 22% | 0% | 1.658 | 6.68% |
| 6 m | robust | 22% | 0% | 1.662 | 6.85% |
| 8 m | nominal | 0% | 0% | 3.171 | 0.80% |
| 8 m | robust | 0% | 0% | 3.172 | 0.87% |
| 10 m | nominal | 0% | 0% | 4.766 | 0% |
| 10 m | robust | 0% | 0% | 4.766 | 0% |

这是一项 null result：在这个一维 head-on / goal-swap 设定中，robust interval
没有带来可辨识的 episode-level safety improvement。`completion=0` 是因为两个
UAV 被要求在一维轨迹上互换位置，而控制器没有横向绕行自由度；该 completion
指标不适合被解释为算法失败或成功。下一步若要研究 efficacy，应改为二维、有
可行绕行路径的任务，并保持当前 one-step activation gate 不变。

## 6. 二维近共线 crossing 复验

新增 `near_crossing`：目标横向偏移仅 0.5 m，仍保留二维控制自由度；先做
10 episodes/距离的试跑。结果同样没有显示 robust efficacy：

| d_start | controller | violation | collision | mean min rho | QP infeasible |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6 m | nominal | 100% | 100% | -0.916 | 22.9% |
| 6 m | robust | 100% | 100% | -0.919 | 23.3% |
| 8 m | nominal | 70% | 60% | -0.434 | 15.1% |
| 8 m | robust | 70% | 60% | -0.430 | 15.4% |
| 10 m | nominal | 50% | 50% | +0.022 | 11.1% |
| 10 m | robust | 50% | 50% | +0.018 | 11.3% |

这组结果表明当前问题已进入 recursive feasibility / viability 边界；robust
interval 不是主要瓶颈。暂不扩到 50 episodes，也不回 PX4，下一步应先设计
带显式横向避让策略或更早 guard 的二维任务。

## 7. 显式 lateral feasibility guard 试验

初版 `guarded_crossing` 在每个 closing timestep 永久叠加 ±0.8 m/s 横向偏置，
因此 guard 实际接管了 nominal mission controller；12 s 内 completion=0，不能
用来评价 τ uncertainty。

现已改为有限状态协议：

```text
NORMAL -> SEPARATE -> RECOVER -> NORMAL
```

进入条件为 `d <= 4.0 m` 且 closing speed `> 0.1 m/s`；释放条件为
`d >= 4.5 m` 或 closing speed 不再超过阈值，并连续保持 3 个 timestep。`SEPARATE`
期间才施加 ±0.8 m/s 横向偏置，`RECOVER` 期间不再施加偏置。

小规模 smoke（5 episodes/distance，600 steps，`y_goal=±0.5 m`）结果：

| d_start | controller | violation | collision | QP infeasible | completion | guard activations |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 6 m | nominal/robust | 0% | 0% | 0% | 60% | 1.0/episode |
| 8 m | nominal/robust | 0% | 0% | 0% | 100% | 1.0/episode |

该结果只证明 guard 已经会释放且不再永久阻塞任务；nominal 与 robust 仍完全
相同，不能宣称 episode-level robust efficacy。下一步是基于 one-step activation
轨迹向前 rollback 1--3 s，构造“可行但 execution uncertainty 足以改变决策”的
受控 episode。

## 8. rollback 受控 episode 的第一轮结果

新增 `marllib/robust_sensitive_rollback.py`，从 one-step activation case 向后
rollback 1/2/3 s，并沿用相同的 right-of-way / yielding 命令。第一轮 smoke
（5 paired episodes/rollback）全部进入 violation/infeasible 区域：

| rollback | nominal violation | robust violation | nominal QP infeasible | robust QP infeasible |
| ---: | ---: | ---: | ---: | ---: |
| 1 s | 100% | 100% | 23.9% | 24.6% |
| 2 s | 100% | 100% | 21.3% | 21.9% |
| 3 s | 100% | 100% | 20.3% | 20.8% |

这不是 robust efficacy 证据，而是一个有效的设计否证：one-step robust-feasible
并不自动意味着把该 snapshot 简单 rollback 后就是 episode-level viable。当前
rollback 协议仍把两机初始化在过近的 crossing corridor，且没有可行的横向
recovery policy。因此暂不扩大 episode 数量，也不回 PX4；下一步应先冻结一个
带显式二维绕行/release policy 的 viable corridor，再重新测试 τ interval 是否
改变 intervention onset 或 realized minimum rho。

## 9. 首轮二维 recovery corridor 复验

在 rollback runner 中加入了有限状态横向 recovery：3.5 m 触发、4.0 m 释放、
±0.8 m/s 横向偏置。结果显示 3 s rollback 已能达到 `violation=0`，但 15--40 s
窗口内 completion 仍为 0；1--2 s rollback 仍会进入 violation/infeasible 区域。
因此这个 recovery policy 目前只能作为 safety corridor，尚未成为可完成的
mission corridor，不能拿来声称 robust efficacy。

当前 Go/No-Go：对 PX4 和四机实验继续保持 No-Go。下一步需要补一个明确的
goal-handoff/rejoin 状态（绕行完成后释放横向偏置并重新分配目标），先让二维
corridor 满足 `completion>0` 且 `violation=0`，再测 nominal-vs-robust 差异。

随后加入了显式 `SEPARATE -> CROSS -> REJOIN` waypoint handoff。3 s rollback、
5 paired episodes、30 s horizon 下安全性为 `violation=0`、`QP infeasible=0`，
但 completion 仍为 0，且 nominal/robust 完全相同。因此 viable-corridor 的
最后一关尚未通过；按预设 stop condition，不再继续添加新的 guard 或扩大实验。
