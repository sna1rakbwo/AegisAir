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

## 7. 下一步

1. 关闭剩余约 0.27% 缺口：离散时间 CBF 或显式记录 `epsilon` violation
   （不声称连续时间严格保证）。
2. 用真实 `MlxLmClient` 替换 rule 版复测，确认 LLM 能产出合法 YIELD/REROUTE。
3. 轻量 sim 的 CBF vs HOCBF 对比表（min_rho / violation / collision /
   intervention / control effort / mission time）。
4. 通过后再接 PX4 velocity-mode。

## 8. 主张边界

当前只证明 HOCBF 已实现且 2 机 sim 安全完成；4 机 sim 尚未解决 deadlock，
更未接 PX4/Gazebo。不能声称四机 HOCBF 安全或 mission 完成。
