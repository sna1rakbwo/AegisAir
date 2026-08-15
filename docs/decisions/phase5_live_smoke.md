# Phase 5 live PX4/Gazebo smoke — 2026-08-16

## 结果

本机（Apple Silicon）已把 Phase 4 的 `RuntimeAssurance` + `AsyncMissionReplanner`
接入真实 PX4 SITL/Gazebo/ROS 2 闭环，并完成首次 live 冒烟。

复用的现有运行时：

- PX4 SITL v1.16 本机二进制 + Gazebo Sim 8.14；
- `safety_margin_scheduler_gate0` 的 `launch_isolated_sitl.sh` /
  `launch_multi_sitl_pose.sh`；
- `safety-margin-gate0-ros2:humble`（ROS 2 Humble + px4_msgs + uXRCE agent）。

新增（本仓库）：

- `px4_adapter/docker/Dockerfile` + `run_adapter.sh`：容器内同跑
  `MicroXRCEAgent` 与 1..N 个 `px4_adapter/node.py`；
- `config/amqtt.yml`：broker 绑定 `0.0.0.0`，让容器经 `host.docker.internal`
  访问 host broker；
- `marllib/phase5_runner.py --mqtt`：arm/takeoff → RA → async replanner →
  `move_to` → land 的 live 控制器。

## 已通过的闸门

1. 单机只读遥测：PASS。`swarm/drone/2/telemetry`（FLU）持续出数，坐标与
   NED/FLU 转换正确。
2. 单机控制：PASS。`arm` → `takeoff` → `move_to` → `land` 全链路，adapter
   本地安全门返回 `accepted=true`，PX4 进入 offboard（nav_state 14），高度
   到 ~2.5 m，位置随 `move_to` 移动。
3. 双机闭环：机械链路 PASS。两台 PX4（instance 2/3）在 ASYNC+rule 模式下
   都到达各自 goal，RA 有 10 次 CBF 介入。

## 未通过/待修

双机 head-on 的安全结果**未通过**：

```text
cbf_events = 10
min_distance_m = 0.0087   # 期望 > d_safe(约 3.65 m)
```

即闭环机械上工作，但两机对头穿越时最小中心距只有约 1 cm，等于接近碰撞。
虽然两机最终都到达 goal（未在 Gazebo 中坠落），但该结果不能作为安全 Go。

## 根因假设（待验证）

轻量 sim 的 CBF 假设无人机瞬间跟踪 RA 输出的 2D 速度；live 回路里该速度被
编码为 0.5 s 前瞻的 `move_to` 位置目标，再由 PX4 位置控制器跟踪，叠加
telemetry/命令/MQTT/ROS 延迟后，CBF 的瞬时速度投影无法维持实时分离。

已做的一项修正：把实测 telemetry 年龄作为 AoI 喂进
`dynamic_safety_boundary`（`M_comm`）。本次数据（0.0087 m）显示单靠该 AoI
修正仍不足，不能因此宣布安全。

## 下一步（同一冻结 head-on 闸门复测，不放松判定）

1. 记录完整轨迹，确认 min distance 是真实接近而非 telemetry 时间错位。
2. 工程修复候选（按成本递增）：
   - 缩短 `COMMAND_HORIZON_S` 并提高控制率；
   - 让 adapter 走 velocity-mode offboard，直接跟踪 RA 安全速度；
   - 在 live 层把 `d0`/`a_eff` 按 PX4 实测制动能力重新标定（只加安全裕度，
     不放松碰撞判定）。
3. 修复后用同一种子/同场景重跑，仍以 min distance > d_safe 为冻结标准。

## 主张边界

这是 SITL 仿真证据，不是真实飞行；当前只能声称“PX4/Gazebo 控制链路与
RA/replanner 已机械闭环”，不能声称双机对头安全。
