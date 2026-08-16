# Acceleration-Aware HOCBF Runtime Assurance（2026-08-16）

## 1. 决策

把 Phase 4/5 的一阶 velocity-CBF 升级为 **acceleration-aware HOCBF**。控制变量
从 `v_safe` 改为 `a_safe`，并把所有 pairwise 约束放进**同一个集中式 QP**，不再
逐机 sequential projection。

触发原因：4 机 PX4/Gazebo 2v2 交叉在 velocity-CBF 下 `min_rho < 0`
（No-Go），且控制率从 10Hz 提到 20Hz 仍为负，说明不是采样频率问题，而是
一阶 velocity 控制面对 PX4 二阶动力学时可行性不足。

## 2. 二阶模型

\[
\dot p_i=v_i,\qquad \dot v_i=a_i
\]

pair 相对量：

\[
r=p_i-p_j,\qquad v=v_i-v_j,\qquad a_{ij}=a_i-a_j
\]

## 3. Barrier（sample-and-hold `d_safe`）

每个 control step 先重新算 `s=d_safe(t)`，但单次 QP solve 内把 `s` 视为常数
（`dot s = ddot s = 0`）：

\[
h=|r|^2-s^2,\qquad
\dot h=2r^Tv,\qquad
\ddot h=2|v|^2+2r^T(a_i-a_j)
\]

二阶 HOCBF：

\[
\boxed{
2|v|^2+2r^T(a_i-a_j)
+2(k_1+k_2)r^Tv
+k_1k_2(|r|^2-s^2)\ge0
}
\]

整理成对 `a_i,a_j` 线性的 pairwise constraint：

\[
\boxed{
2r^T(a_i-a_j)
\ge
-2|v|^2
-2(k_1+k_2)r^Tv
-k_1k_2(|r|^2-s^2)
}
\]

## 4. 集中式 QP

把所有 UAV 的加速度堆成一个向量 `A=[a_1,a_2,...]`，一次性最小化：

\[
\min_A \sum_i |a_i-a_i^{nom}|^2
\quad\text{s.t.}\quad
\psi_{2,ij}\ge0,\ -a_{max}\le a_i\le a_{max}
\]

其中：

\[
a_i^{nom}=\mathrm{clip}\left(k_v(v_i^{MARL}-v_i),\ -a_{max},\ a_{max}\right)
\]

输出 velocity setpoint：

\[
v_{cmd}=v_{actual}+a_{safe}\Delta t,\qquad |v_{cmd}|\le v_{max}
\]

四机最多 6 条 pairwise constraints，仍是凸 QP。当前无 scipy/cvxopt/osqp，
所以用 numpy 实现 Dykstra 交替投影（box + half-spaces）。

## 5. 实现

- `swarm/ra/hocbf.py`：pairwise constraint + 集中式 QP。
- `swarm/ra/runtime_assurance.py`：`use_hocbf` 模式，`a_nom → QP → a_safe →
  v_cmd`；记录 `qp_solve_count` / `qp_infeasible_count`。
- `marllib/phase5_runner.py`：`--hocbf`、`--hocbf-k1/k2`、`--a-max`、`--kv`。
- `tests/test_hocbf.py`：4 机约束满足、加速度 box、无 pair 退化。

## 6. 轻量 sim 首验

| 场景 | velocity-CBF | HOCBF (k1=k2=1, a_max=2, kv=2) |
| --- | --- | --- |
| 2 UAV head-on | 完成，0 碰撞，cbf_events 76 | 完成，0 碰撞，cbf_events 110 |
| 4 UAV crossing | 完成，0 碰撞，cbf_events 282 | 未完成，0 碰撞，cbf_events 644 |

结论：

- E1 2 机 HOCBF 安全且完成；
- E2 4 机 HOCBF 安全（0 碰撞）但**未到 goal**；`(k1,k2)` 越大越安全但越容易
  卡死，越小越可能短时跌破 `rho=0`。

`(k1,k2)` 对 4 机 `min_rho` 的实测（修复 Dykstra 收敛后）：

```text
k1,k2     min_rho       cbf_events  qp_infeasible
0.5,0.5   -0.1109       790         0
1,1       -0.0207       644         0
2,2       2.4e-16       800         0
```

说明对称 4 机交叉处在 safety-completion 边界：要 `rho>=0` 就会接近 deadlock。
这是集中式 QP 在该对称几何下的真实现象，不是求解器错误。

## 6.1 deadlock 处理尝试（未解决）

在 `k1=k2=2` 上加确定性侧向分离 bias（stall 后按 id 奇偶给横向速度），能
完成 mission，但 `min_rho` 仍小幅为负（约 -0.011~-0.019）。

优先级 yield（stall 时让 id 2/3 停车）同样不能同时满足
`rho>=0` 和完成：

```text
scheme            min_rho    completed
lateral bias      -0.0115    True
priority yield    -0.0086    False
```

结论：对称 4 机同时穿越在 `d_safe≈1.5m` 下，当前 HOCBF 无 slack 时无法既安全
又完成；需要的是 **barrier slack（显式声明软化）**，或预先排序/错峰通过，
而不是简单侧向/让行。

## 6.2 LLM deadlock resolution（Semantic Mission Recovery）

把死锁交给高层 LLM：`COORDINATION_DEGRADATION`（无 progress）触发
`AsyncMissionReplanner`，rule 版 LLM 输出确定性 `YIELD + REROUTE`（按 id 奇偶
选侧向方向），HOCBF 继续在线过滤。轻量 sim 结果：

```text
mode                    min_rho    collision  completed  triggers
HOCBF k=3 + ASYNC rule  -0.0066    False      True       4
```

即 LLM 死锁解除后 mission 完成、0 碰撞；`min_rho=-0.0066` 仍有一个很小（0.66%）
的离散化缺口。相比纯 HOCBF 的 deadlock，以及侧向/让行策略，这是当前最接近
`rho>=0` 且完成的结果。

小 `dt` 复测（HOCBF + ASYNC rule，固定总仿真时长）：

```text
dt     k=2.5       k=3.0       k=3.5
0.1    -0.0163     -0.0066     -0.0140
0.05   -0.0027     -0.0029     -0.0031
0.02   -0.0073     -0.0038     -0.0035
```

`dt=0.05` 把缺口压到约 0.27%，但 `dt=0.02` 不再单调改善；剩余缺口不是单纯
`dt`，还包含离散时间 barrier / LLM recovery maneuver 的残差。

## 6.3 冻结：discrete-time implementation tolerance

**不上离散时间 CBF。** 把采样实现的小幅安全裕度违反记为
`epsilon_impl = 3e-3`，并明确写为 limitation：

> The sampled implementation permits a small empirical safety-margin
> tolerance (`epsilon_impl ≈ 3e-3`), while maintaining zero collisions in
> the evaluated scenarios.  We therefore do not claim exact continuous-time
> invariance in the sampled implementation.

冻结评估配置：

```text
dt = 0.05 s
k1 = k2 = 3.0
a_max = 2.0 m/s^2
kv = 2.0
```

对应结果：`collision=0`、`mission completed`、`min_rho≈-0.0029`。

## 6.4 CBF vs HOCBF 首轮统计（N=5，dt=0.05）

```text
scenario   metric              velocity-CBF   HOCBF(+ASYNC rule for 4UAV)
head_on    collision_rate      0.0            0.0
           completion_rate     1.0            1.0
           mean_mission_steps  138.0          127.0
           mean_cbf_events     154.0          92.0
           min_rho             0.599          0.348
           violation_rate      0.0            0.0

multi_uav  collision_rate      0.0            0.0
           completion_rate     1.0            1.0
           mean_mission_steps  170.0          357.0
           mean_cbf_events     512.0          996.0
           min_rho             -0.0103        -0.0029
           violation_rate      1.0            1.0
```

解读：

- 2 机：HOCBF 更快、干预更少，且 `rho>0`，说明它在松场景用安全裕度更高效。
- 4 机：HOCBF 把最小安全裕度违反从 1.03% 压到 0.29%（3.5×），代价是 mission
  时间变长、干预变多；LLM 死锁解除让完成率保持 1.0。

## 6.5 imperfect MARL（nominal noise=0.6，N=10，dt=0.05）

```text
scenario   metric      velocity-CBF  HOCBF(+ASYNC rule for 4UAV)
head_on    min_rho     0.606         0.277
           cbf_events  206.2         88.2
multi_uav  min_rho     -0.0008       -0.0014
           violation   0.1           0.1
           cbf_events  507.6         331.8
```

所有场景 0 碰撞、完成率 1.0。HOCBF 在噪声 nominal 下显著降低干预次数
（head_on 88 vs 206，multi_uav 332 vs 508），安全裕度保持同级。

## 6.6 PX4/Gazebo 2 机 head-on（HOCBF live）

velocity-mode offboard + HOCBF（k1=k2=3，20Hz）：

```text
min_rho = 0.2779  (> 0)
min_distance_m = 1.7739
cbf_events = 24
```

即 HOCBF 在真实 PX4/Gazebo 闭环中保持 `rho>0`。这是第一次 HOCBF 的 live 安全
通过；4 机 PX4 版本与 `epsilon_impl` 放大观察留作后续。

## 6.7 4 机 PX4 HOCBF —— No-Go

4 机 2v2 交叉在 PX4/Gazebo（velocity-mode + HOCBF k=3、20Hz、ASYNC rule）：

```text
min_rho = -0.8072
min_distance_m = 0.3073
cbf_events = 375
triggers = 4, plans_committed = 4, mission_changes = 4
```

`min_rho` 从 sim 的 `-0.003` 放大到 `-0.807`，且最小中心距 0.31 m 接近碰撞
（drone 4/5 在中心附近几乎相撞，drone 4 最终落到地面 z=-0.25）。这已经不能
用 `epsilon_impl≈3e-3` 的离散化 tolerance 解释。

按之前约定：PX4 4 机出现 near collision 时，应重新考虑 discrete-time CBF。

## 7. 下一步

1. 把 N 提到 10，补 control effort（`mean |a_safe|`）与 4 机真实 Qwen 对比。
2. imperfect MARL 下的 HOCBF 保护。
3. 4 机 PX4 已出现 near collision，进入 **discrete-time CBF / higher-fidelity
   live controller** 阶段：先诊断是 QP 截止频率不足、velocity-mode 跟踪、还是
   deadlock recovery 时机问题，再决定升级方向。

## 8. 诊断（4 机 PX4 near-collision）

从 4 机 PX4 轨迹测到的关键量：

```text
HOCBF QP solve time      mean 20 ms（20Hz deadline 50ms 内）
telemetry AoI            mean 0.029 s, max 0.07 s
实际控制 loop dt         mean 0.128 s => 7.8 Hz（请求 20Hz）
velocity tracking error  mean 0.285 m/s, p90 0.364 m/s
first rho<0              t=1.43 s
worst rho                -0.807 @ t=5.2 s, min distance 0.31 m
```

结论：

1. QP 求解不是瓶颈。
2. AoI 也不大。
3. **实际控制率只有 7.8Hz**（每步 MQTT 发布/遥测 + 50ms sleep 叠加），把
   discrete-time HOCBF 的积分步长从 0.05s 拉成 ~0.128s。
4. **PX4 velocity tracking 有 ~0.29 m/s 滞后**，而 HOCBF 假设
   `v_dot = a_safe` 的理想双积分器，实际 `v_actual` 跟不上 `v_cmd`。

因此 near-collision 是“控制率不足 + 速度跟踪滞后未建模”共同导致，不是单纯
采样容差。

## 9. 修复优先级

1. 先修控制率：把 live loop 改成固定定时器/减少每步 MQTT 开销，实测回 20Hz。
2. 把实测 velocity-tracking lag 以 `tau_ctrl≈0.2s` 形式并入 `M_dyn`，或加
   velocity-tracking compensator。
3. 若仍失败，上 discrete-time CBF，用 PX4 速度跟踪模型离散化 barrier。

## 10. 修复 1+2 后复测（仍 No-Go）

固定速率 deadline + `tau_ctrl=0.2` 后：

```text
actual loop dt   = 0.050 s => 20.0 Hz（原 7.8 Hz）
tracking error   = mean 0.102 m/s, p90 0.141（原 0.285 / 0.364）
min_rho          = -0.7598（原 -0.8072，基本未改善）
min_distance_m   = 0.375（仍 near collision）
```

控制率和跟踪误差都修好了，但 `min_rho` 仍约 `-0.76`。说明 remaining root
cause 不是控制率，而是**连续时间 HOCBF + sample-and-hold d_safe + velocity-mode
跟踪**在真实 4 机 PX4 上不成立。按诊断进入第 3 项：discrete-time CBF。

## 8. 主张边界

当前只证明 HOCBF 已实现且 2 机 sim 安全完成；4 机 sim 尚未解决 deadlock，
更未接 PX4/Gazebo。不能声称四机 HOCBF 安全或 mission 完成。
