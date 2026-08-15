# Phase 5 —— PX4/Gazebo 闭环计划

> 日期：2026-08-16
> 状态：Plan + 第一道冻结闸门（本地确定性闭环）
> 前置：Phase 4 已实现并 push；85 个主测试 + 34 个 px4_adapter 测试全绿。

## 1. 目标

把 AegisAir 的 System-1/System-2 栈接入 PX4 SITL/Gazebo，实现：

```text
PX4 telemetry -> observation -> nominal pilot (规则/MARL) ->
Runtime Assurance (CBF) -> async mission replanning -> PX4 Offboard
```

完成标准来自 `plan.md` Phase 5：

1. telemetry → observation；
2. MARL inference；
3. velocity setpoint；
4. Runtime Assurance；
5. PX4 Offboard；
6. head-on；
7. crossing；
8. multi-UAV。

**完成标准：完整 closed loop 工作。**

## 2. 现状与复用

- PX4/Gazebo 运行时已在外部目录验证过（`px4_migration_summary.md`、
  `/Volumes/Expansion/safety_margin_scheduler_gate0_runtime/`）。
- Docker 镜像已存在：
  - `px4io/px4-sitl-gazebo:v1.18.0-beta1-arm64`
  - `safety-margin-gate0-ros2:humble`（含 `px4_msgs` + Micro-XRCE-DDS Agent）
- `px4_adapter/` 已含 ROS 2 节点、MQTT/PX4 codec、本地 fail-closed 状态机。
- Phase 4 的 `RuntimeAssurance`、`AsyncMissionReplanner` 已在轻量环境验证。

## 3. 冻结接口（Phase 5 不改）

- 遥测：`swarm/drone/{id}/telemetry`，FLU，字段见
  `encode_safedrones_telemetry`。
- 命令：`swarm/drone/{id}/command` 或 `px4/{id}/command`，动作白名单
  `arm/takeoff/move_to/land/rtl/hold`，`source_frame` 显式。
- 坐标：FLU（x 前、y 左、z 上）；适配器负责 FLU↔NED。
- 本地安全限制：`px4_adapter/config.yaml` 的 `safety.*`（ttl、速度、高度、
  距离、fail-closed 动作）。

任何新消息格式必须先冻结到 `swarm/interfaces.py`（swarm 层）或
`px4_adapter/mqtt_codec.py`（适配器层）并加测试。

## 3.1 复用现有 PX4/Gazebo 运行时（不要重建）

已存在的可复用资产：

```text
PX4_ROOT = /Users/lijiajun/Documents/ChatGPT/无人机论文尝试/PX4-Autopilot
GATE0    = /Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0
```

- PX4 SITL v1.16 二进制（本机 arm64）：`PX4_ROOT/build/px4_sitl_default/bin/px4`
- Gazebo Harmonic（本机）：`gz`
- 启动/停止脚本：
  - `GATE0/scripts/launch_multi_sitl_pose.sh S1 2,3`
  - `GATE0/scripts/stop_multi_sitl.sh <state_dir>`
- ROS 2 Humble + uXRCE bridge 镜像：`safety-margin-gate0-ros2:humble`
  - `GATE0/scripts/start_bridge_agent.sh`（`XRCE_UDP_PORT=8889`）
- MQTT：`SafeDrones/scripts/dev_broker.py`（amqtt）或任意 mosquitto。
- 适配器：本仓库 `px4_adapter/node.py`（ROS 2 节点，MQTT↔PX4，本地 fail-closed）。

Live 闭环编排（顺序启动，逐步验证）：

```bash
# 1) MQTT broker
cd /Users/lijiajun/Documents/drone/SafeDrones
/opt/anaconda3/envs/eai-swarm/bin/python scripts/dev_broker.py --config config/amqtt.yml

# 2) Gazebo + 2 PX4（native，脚本阻塞）
"$GATE0/scripts/launch_multi_sitl_pose.sh" S1 2,3

# 3) ROS2/uXRCE bridge（Docker）
"$GATE0/scripts/start_bridge_agent.sh"

# 4) 每个 instance 跑 px4_adapter（在 bridge 容器内；需要 rclpy+px4_msgs）
#    注意：容器内需同时有 MicroXRCEAgent 与 adapter node，且能访问 host MQTT。
#    该容器编排需要 live 验证后固化。

# 5) AegisAir Phase 5 控制器（host）
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase5_runner.py \
  --scenario head_on --mode ASYNC --llm rule --mqtt
```

步骤 4 的容器编排（MicroXRCEAgent + adapter 同容器）尚未在本次 live 验证；
第一步先走单机只读遥测，确认 `swarm/drone/{id}/telemetry` 再放开
`--read-only`。

## 4. 第一道闸门（本次执行，最低成本）

**不启动 PX4/Gazebo**，用轻量 `MultiUAVEnv` 作为确定性 telemetry 源，把
`RuntimeAssurance` + `AsyncMissionReplanner` 的 safe velocity 编码成适配器
命令，并用真实 `px4_adapter` codec + `LocalSafetyState` 解码/校验。

验证对象：

- telemetry → `DroneSnapshot`；
- nominal pilot（`go-to-goal` 规则速度，作为 MARL 推理的占位）；
- `RuntimeAssurance.filter`；
- `AsyncMissionReplanner.step`；
- safe velocity → FLU `move_to` 命令 → 适配器 codec → 本地安全门。

### 4.1 冻结对象

- 场景：`head_on`（双机对头穿越）、`priority_conflict`、
  `corridor_blocked`、`drone_failure`。
- 模式：`CBF_ONLY`、`ASYNC`。
- LLM 后端：`rule`（`RuleMissionPlanner`，确定性）；`qwen` 留作第二阶段。
- 巡航高度：`2.5 m`；命令外推 horizon：`0.5 s`；ttl：`0.5 s`。
- 每场景种子：`1..10`（本闸门先 1 seed 冒烟，正式统计用 10）。

### 4.2 统计与停止规则

每 episode 指标：`collision`、`completed`、`completion_steps`、
`cbf_events`、`zone_crossed`、`critical_reached`、`high_reached_step`、
`emitted_commands`。

停止规则：

- 任何编码命令被 `LocalSafetyState` 拒绝且原因不是“自身越限”的，判定为
  `ENGINEERING_INVALID`（命令路径错误）。
- 不放松阈值、不加种子、不把轻量 sim 结果当作 PX4 飞行证据。

### 4.3 主张边界

本闸门只验证“telemetry → RA → replanner → 适配器命令”的确定性集成路径。
它不证明 PX4 飞行安全、碰撞避免统计或端到端故障统计。

## 5. 后续闸门（按成本递增）

1. **单机 PX4 只读遥测**：`px4_adapter/node.py` `control.read_only=true`，
   确认 `swarm/drone/{id}/telemetry` 与 `DroneSnapshot` 解析。
2. **单机 PX4 闭环**：`arm → takeoff → move_to`，RA 的 safe velocity 经适配器
   到 Offboard；验证一条直线航线。
3. **双机 head-on/crossing**：CBF_ONLY vs ASYNC，复用本 runner 的 `--mqtt` 路径。
4. **Phase 4 三场景复测**：priority_conflict / corridor_blocked / drone_failure，
   在 Gazebo 中用 Qwen 或 rule planner。

## 6. 可复现命令

```bash
cd /Users/lijiajun/Documents/drone/AegisAir

# 本地确定性闭环（第一道闸门）
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase5_runner.py \
  --scenario head_on --mode ASYNC --llm rule --seeds 1 --max-steps 120

# 测试
/opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s tests
```
