# Phase 5 live PX4/Gazebo smoke — 2026-08-16

## 结论（更新）

本机（Apple Silicon）已把 Phase 4 的 `RuntimeAssurance` + `AsyncMissionReplanner`
接入真实 PX4 SITL/Gazebo/ROS 2 闭环。首次 1 cm 的“接近碰撞”是坐标帧测量
错误，不是真实碰撞；修正共享坐标系并切换到 velocity-mode Offboard 后，
单 seed head-on 安全门通过。

## 复用的现有运行时

- PX4 SITL v1.16 本机二进制 + Gazebo Sim 8.14；
- `safety_margin_scheduler_gate0` 的 `launch_multi_sitl_pose.sh`；
- `safety-margin-gate0-ros2:humble`（ROS 2 Humble + px4_msgs + uXRCE agent）。

## 新增（本仓库）

- `px4_adapter/docker/Dockerfile` + `run_adapter.sh`：容器内同跑
  `MicroXRCEAgent` 与 1..N 个 `px4_adapter/node.py`，支持每 instance 的
  `ORIGIN_OFFSET_<id>`。
- `config/amqtt.yml`：broker 绑定 `0.0.0.0`，容器经 `host.docker.internal`
  访问 host broker。
- `px4_adapter` 新增 `velocity` 动作：`TrajectorySetpoint.velocity` 有效、
  position 为 NaN，走 PX4 velocity-mode Offboard。
- `marllib/phase5_runner.py --mqtt`：arm/takeoff → RA → async replanner →
  `velocity` setpoint → land，并支持 `--trajectory` 轨迹日志。

## 关键修复（P0 / P1）

### P0 共享坐标系

`vehicle_local_position` 是每台 PX4 各自的局部坐标（EKF 原点在各自 spawn 点），
所以不共享原点时两台无人机都从 ~0 开始，pairwise CBF 的距离计算没有意义。

修复：给每个 adapter instance 传 `--origin-offset-ned`：

```text
instance 2: -3,0,0
instance 3:  3,0,0
```

对应 `launch_multi_sitl_pose.sh` 的默认 spawn pose
`2=0,-3,0.5;3=0,3,0.5`。修正后 telemetry 显示 drone 2 x≈-3、drone 3 x≈+3。

### P1 velocity-mode Offboard

RA 输出 2D 安全速度，直接编码为 FLU `velocity` 命令；adapter 转成 PX4 NED
`TrajectorySetpoint.velocity`，`position` 设 NaN。垂向由 runner 里一个简单
高度保持 P 控制器给 `vz`。不再把安全速度积分成 0.5 s 位置目标。

## 已通过的闸门

1. 单机只读遥测：PASS。
2. 单机控制 `arm→takeoff→move_to→land`：PASS。
3. 双机 head-on（ASYNC + rule，单 seed）：PASS（安全判据 `min rho >= 0`）。

双机修正后结果：

```text
min_rho        = 0.321622   # > 0
min_distance_m = 2.1354
cbf_events     = 66
final_x:       drone2=3.989, drone3=-3.952
```

## 10 次 frozen head-on 重复（P4）

同一 frozen 场景（spawn ±3、goal ∓4、ASYNC + rule）重复 10 次，每次先
`reset-to-start` 再穿越：

```text
rep  min_rho    min_distance_m  cbf_events
1    0.367311   2.1363          61
2    0.622190   1.9987          84
3    0.644867   1.9924          84
4    0.614095   2.0148          84
5    0.595035   2.0014          82
6    0.603528   1.9760          82
7    0.440230   2.0596          80
8    0.419085   2.0784          80
9    0.413527   2.0365          80
10   0.414691   2.0611          81
```

`min_rho_all = 0.367311 > 0`，且 10/10 次两机都到达 goal。该 frozen gate 通过。
轨迹：`/Volumes/Expansion/aegisair_phase5_20260816/headon10_rep*.jsonl`。

## 多 seed 横向抖动 head-on（10 seed）

`--multi-seed 10`，每 seed 用 `random.Random(seed).uniform(-0.8, 0.8)` 生成
goal 横向偏移（drone2→(4, a)，drone3→(-4, -a)），spawn/reset 仍固定 ±3：

```text
seed lateral   min_rho   min_distance_m  cbf_events
1    -0.5850   0.303710  2.2632          61
2     0.7297   0.445893  2.4214          68
3    -0.4193   0.536637  1.8458          88
4    -0.4223   0.595349  1.8458          90
5     0.1966   0.311847  1.4425          108
6     0.4693   0.610254  1.8870          86
7    -0.2819   0.409041  1.6331          98
8    -0.4373   0.624450  1.8406          88
9    -0.0592   0.125655  1.2055          138
10    0.1142   0.216771  1.2983          122
```

`min_rho_all = 0.125655 > 0`，10/10 seed 通过。最接近正对头的 seed 9
（lateral=-0.059）也保持 `rho>0`。轨迹：
`/Volumes/Expansion/aegisair_phase5_20260816/headon_seeds_seed*.jsonl`。

## 垂直交叉 crossing（10 次重复）

`--starts '2=-3,0,2.5;3=0,-3,2.5'`、`--goals '2=3,0,2.5;3=0,3,2.5'`，
ASYNC + rule，10 次重复：

```text
rep min_rho   min_distance_m  cbf_events
1   0.531066  3.1218          42
2   0.525001  3.0289          42
3   0.486357  3.0612          43
4   0.548477  3.0520          44
5   0.546623  3.0776          43
6   0.540687  3.0185          42
7   0.520154  3.0545          42
8   0.527330  3.0918          42
9   0.519849  3.0396          42
10  0.536043  3.0422          43
```

`min_rho_all = 0.486357 > 0`，10/10 通过，两机都到达各自 goal。轨迹：
`/Volumes/Expansion/aegisair_phase5_20260816/crossing10_rep*.jsonl`。

## 四机 2v2 交叉（multi-UAV）—— No-Go

4 台 PX4（instance 2/3/4/5）两两对头交叉，ASYNC + rule。首 3 次重复：

```text
rep min_rho     min_distance_m  cbf_events
1   -0.025376   1.0677          354
2   -0.069093   0.9997          464
3   -0.046738   1.0188          427
```

`min_rho < 0`，即存在 `d < d_safe` 的瞬间，安全判据**未通过**。这是 4 机场景
首次把当前一阶 velocity CBF + 顺序投影压到失败：每机有 6 个 pairwise 约束、
4 机同时投影，顺序投影无法在实时动态下维持 `rho >= 0`。

按既有优先级，下一步不是无限加大 `d0`，而是：
1. P2 控制率 20/50Hz 先测；
2. P3 用 PX4 实测 `a_eff/tau_ctrl` 回填 `M_dyn`；
3. 仍失败则升级 HOCBF / acceleration-aware barrier。

## 剩余（还不能说正式冻结）

- head-on 与 crossing 已过；4 机 multi-UAV 未过（当前一阶 CBF 的边界）。
- 仍需 P2 控制率扫描、P3 多速度制动标定，必要时升级 HOCBF。
- 尚未做 P2 控制率扫描（10/20/50 Hz）。
- P3 已做第一次速度阶跃（`marllib/phase5_step_response.py`），实测
  `v0≈1.56 m/s`、`d_brake≈0.59 m`、`a_eff≈2.06 m/s²`、`tau_ctrl < 0.1 s`；
  与默认 `M_dyn` 的 `a_eff=2.0 m/s²` 基本一致，`tau_r=0.3 s` 反而更保守。
  但仍需多种速度多次重复取分位数后回填。

## 下一步

1. 把 spawn/offset/goal 加上 per-seed 横向抖动，跑多 seed 的 head-on 统计。
2. P2：把 adapter/runner 控制率提到 20/50 Hz 复测。
3. P3 补速度 0.5/1.0/1.5 m/s 各多次，取 `a_eff=P10(|a_decel|)` 回填 `M_dyn`。
4. 若多 seed 仍失败，再升级 HOCBF，不无限加大 `d0`。

## 主张边界

这是 SITL 仿真证据，不是真实飞行；当前只能声称“PX4/Gazebo 控制链路与
RA/replanner 已闭环，且单 seed head-on 满足 `min rho >= 0`”，不能声称真实
硬件安全或正式统计结论。
