# PX4/Gazebo C2 多 seed 复位修复交接文档

日期：2026-08-21
状态：**待修（阻塞项）**。论文必须有 PX4 闭环仿真，C2 的 E0/E1/E2 对比必须在真实 PX4
执行回路上跑多 seed；当前单集闭环已通，但多 seed 复位不可靠，导致 E0/E1/E2 无法做
配对比较。

## 1. 为什么必须修

- 论文 claim 需要“PX4 SITL/Gazebo 闭环”证据，不能只靠轻量仿真。
- C2 的核心是“执行一致性”（exact-ZOH vs 启发式 vs 瞬时），这一比较在轻量 exact-ZOH
  plant 里是自证（循环论证），必须在真实 PX4 执行回路上验证。
- 多 seed 是配对统计的前提；复位不可靠 = seed 起点不一致 = 比较无效。

## 2. 现状

**已通**：

- 完整 PX4 SITL/Gazebo/MQTT/RA 闭环端到端跑通。
- 单集 E2 正对头穿越：`min_distance ≈ 0.33–0.79 m`、0 碰撞、CBF 干预正常。
- E0/E1/E2 三种执行模型都能在真实 PX4 上运行（`--execution-model` 已接进 live 路径）。

**不通**：

- 多 seed 的 `reset_starts` 复位不可靠，seed 之间起点漂移，E0/E1/E2 无法配对比较。

## 3. 环境与启动流程（可复现）

### 3.1 两个已踩过的坑

1. **docker CLI 软链断**：`/usr/local/bin/docker` 指向 App Translocation 临时路径，报
   `command not found`。用全路径 `/Applications/Docker.app/Contents/Resources/bin/docker`
   绕过，或 `sudo ln -sf /Applications/Docker.app/Contents/Resources/bin/docker /usr/local/bin/docker`。
2. **`host.docker.internal` 解析成 IPv6**（`fdc4:f303:9324::254`），broker 只监听 IPv4
   `0.0.0.0:1883`，容器内连不上。adapter 的 `--mqtt-host` 用宿主机 LAN IP
   `192.168.1.20`（`ipconfig getifaddr en0` 确认）。

### 3.2 启动顺序（四个长驻组件）

```bash
# 1) MQTT broker（AegisAir 目录下）
/opt/anaconda3/envs/eai-swarm/bin/amqtt -c config/amqtt.yml

# 2) Gazebo + 2 个 PX4 SITL（阻塞，另开终端）
cd '/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0'
POSES='2=0,-3,0.5;3=0,3,0.5' bash scripts/launch_multi_sitl_pose.sh S1 2,3

# 3) GCS heartbeat（保 preflight）
/opt/anaconda3/envs/eai-swarm/bin/python \
  /Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0/scripts/gcs_heartbeat.py \
  --port 18572 --duration-s 3600   # 18573 再起一个

# 4) adapter 容器（velocity mode 20Hz）
DOCKER=/Applications/Docker.app/Contents/Resources/bin/docker
$DOCKER run -d --name aegisair-adapters \
  -p 127.0.0.1:8889:8889/udp -e XRCE_UDP_PORT=8889 \
  -e ADAPTER_INSTANCES='2 3' \
  -e ADAPTER_ARGS='--no-read-only --mqtt-host 192.168.1.20 --mqtt-port 1883 --control-rate-hz 20 --telemetry-rate-hz 20' \
  -e ORIGIN_OFFSET_2='-3,0,0' -e ORIGIN_OFFSET_3='3,0,0' \
  aegisair-px4-bridge:humble
```

改 `px4_adapter` 代码后需重建镜像：

```bash
$DOCKER build -t aegisair-px4-bridge:humble -f px4_adapter/docker/Dockerfile .
```

## 4. 坐标约定（关键，别改错）

- 每台 PX4 的 `vehicle_local_position` 是**各自本地 NED**（原点在各自 spawn 点）。
- adapter 的 `build_state` 把 telemetry 变成**共享坐标系**：`position = local_ned + origin_offset_ned`，
  再 `px4_ned_to_flu` 发到 MQTT（`swarm/drone/{id}/telemetry`，`source_frame="FLU"`）。
- `ORIGIN_OFFSET`：drone 2 = `-3,0,0`，drone 3 = `3,0,0`；spawn `POSES='2=0,-3,0.5;3=0,3,0.5'`
  （Gazebo +Y = FLU +X，所以 Gazebo `(0,-3)` = 共享 FLU `(-3,0)`）。
- 命令 `swarm/drone/{id}/command` 是 **FLU**：`velocity` 只做 `flu_to_px4_ned`（符号翻转，
  平移不变）；`move_to` 的位置目标要再减 `origin_offset`（已修，见 §5）。

## 5. 已修复的根因

`px4_adapter/node.py` 的 `_convert_target` 之前只做 FLU→NED 符号翻转，没有减
`origin_offset_ned`，导致共享 FLU 位置目标被当成本地 NED，偏了 `origin_offset`。
这解释了最早的“奇/偶 seed 交替”现象。修复（commit `529f501`）抽出 `flu_to_local_ned`
并加单测。这是多 seed 复位的**第一个** bug，但不是唯一一个。

## 6. 三种复位方案的记录与结果

### 6.1 move_to（origin_offset 修复后）

- 位置目标现在正确（`flu_to_local_ned`）。
- 残留问题：`move_to` 是 position-mode，复位后切到 crossing 的 velocity-mode，存在模式
  切换 + 残速，seed 间仍有漂移。

### 6.2 velocity + settle

- 复位改用 velocity（与 crossing 同模式，无模式切换），复位后 settle 1.5s。
- 结果：E2 一致了（`0.33–1.86 m`），但 E0/E1 发散（部分 seed `min_distance` 到 7–19 m）。
- 未提交（已被 teleport 覆盖）。

### 6.3 teleport（gz service set_pose，当前工作区未提交）

- `gz service -s /world/s1_single_obstacle/set_pose` 直接瞬移 Gazebo 模型。
- 命令本身返回 `data: true`，但 teleport 后无人机**不穿越**（`min_distance≈5.98 m`、
  `cbf_events=0`）——teleport 打断了 PX4 的 offboard/EKF 状态，crossing 的 velocity
  指令失效。

## 7. 修复方向（按优先级）

### 7.1 teleport 破坏 offboard/EKF 状态（最可能）

teleport 只是挪 Gazebo 模型，PX4 侧的 offboard 模式 / EKF 状态没跟着重置。下一步先诊断：

1. teleport 后订阅 `swarm/drone/{id}/telemetry`，看 `nav_state` / `armed` / `failsafe` /
   `connection_lost` 是否还正常（预期 `nav_state=14` offboard、`armed=true`）。
2. 若 offboard 被破坏，尝试：teleport 后重新 `arm` + `takeoff`；或找 PX4 原生
   teleport 命令（`vehicle_command` 的 EKF reset / `MAV_CMD_DO_SET_HOME` / 直接走
   MAVLink `SET_POSITION_TARGET_GLOBAL_INT`），而不是只挪 Gazebo 模型。
3. 检查 adapter 是否需要发一个“重新进 offboard”的 `DO_SET_MODE`。

### 7.2 E0/E1（tau_px4=0 / legacy_trapezoidal）发散

velocity+settle 下 E2 一致、E0/E1 发散，怀疑 `tau_px4=0` 时 sampled-data QP 不可行或
输出饱和。诊断：跑 E0 时在 `run_mqtt_loop` 打日志看 `results[i].feasible`、
`a_safe`、`vel_saturated`，确认是否 QP 不可行导致 RA 输出异常。

### 7.3 场景太激进（正对头 min_distance 到 0.22 m）

`--lateral 0` 正对头太狠（最接近碰撞半径 0.25 m）。考虑用 `--lateral 0.3–0.5` 给点横向
余量，让 E0/E1/E2 的差异体现在“分离距离”而不是“碰不碰”。

### 7.4 兜底：每执行模型重启 SITL

如果复位始终不可靠，最稳但最慢的做法是每个 `(model, seed)` 都重启 SITL 再跑单集。代价是
30 次 × 约 30s 启动 = 15 分钟以上，且要自动化。

## 8. 关键文件

- `marllib/phase5_runner.py`：`run_mqtt_loop`（arm/takeoff/reset/crossing）、`_teleport_drone`
  （当前未提交的 teleport 尝试）、`--execution-model`（已提交）。
- `marllib/run_c2_gazebo.py`：E0/E1/E2 多 seed driver（`--lateral` / `--models` / `--out`）。
- `px4_adapter/node.py`：`_convert_target`（已修）、`_convert_velocity`、`control_tick`。
- `px4_adapter/px4_codec.py`：`build_control_plan`、`flu_to_px4_ned`、`flu_to_local_ned`。
- gate0：`launch_multi_sitl_pose.sh`（PX4_GZ_MODEL_POSE spawn）、`launch_multi_sitl.sh`
  （Gazebo set_pose service 分离，可参考其 teleport 用法）、`gcs_heartbeat.py`。

## 9. 验证标准（修好后必须满足）

`run_c2_gazebo.py --seeds 10 --lateral 0`（或 0.3）跑 E0/E1/E2：

- 10/10 seed 都是**有效穿越**（`cbf_events > 0`，`min_distance` 稳定在一个合理区间，无
  7 m+ 发散、无 0.25 m 以下碰撞）；
- 三模型 0 碰撞；
- 同一 seed 起点一致，E0/E1/E2 可直接配对比较 `min_rho` / `min_distance`。

## 10. 当前代码/提交状态

- 已提交：`529f501`（origin_offset 修复）、`8e03007`（E0/E1/E2 driver + `--execution-model`）、
  `b9f77b2`（重定位 + 扩规模协议）。
- 未提交：`marllib/phase5_runner.py` 里的 teleport 尝试（§6.3，不通）。建议从已提交的
  move_to 基线开始修，teleport 这版不要提交。
