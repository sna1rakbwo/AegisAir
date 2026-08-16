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

## 剩余（还不能说正式冻结）

- 这是 1 seed 冒烟，不是 10-seed 统计闸门。
- 尚未做 P2 控制率扫描（10/20/50 Hz）。
- P3 已做第一次速度阶跃（`marllib/phase5_step_response.py`），实测
  `v0≈1.56 m/s`、`d_brake≈0.59 m`、`a_eff≈2.06 m/s²`、`tau_ctrl < 0.1 s`；
  与默认 `M_dyn` 的 `a_eff=2.0 m/s²` 基本一致，`tau_r=0.3 s` 反而更保守。
  但仍需多种速度多次重复取分位数后回填。

## 下一步

1. 同一 frozen head-on 场景重复 10 次，出 `min_rho` 分布与
   `min_t rho(t) >= 0` 统计（每次从 spawn 重新起飞，或加 reset-to-start）。
2. P2：把 adapter/runner 控制率提到 20/50 Hz 复测。
3. P3 补速度 0.5/1.0/1.5 m/s 各多次，取 `a_eff=P10(|a_decel|)` 回填 `M_dyn`。
4. 若 10 次仍失败，再升级 HOCBF，不无限加大 `d0`。

## 主张边界

这是 SITL 仿真证据，不是真实飞行；当前只能声称“PX4/Gazebo 控制链路与
RA/replanner 已闭环，且单 seed head-on 满足 `min rho >= 0`”，不能声称真实
硬件安全或正式统计结论。
