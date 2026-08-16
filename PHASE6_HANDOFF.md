# Phase 6 交接文档（2026-08-16）

> 新对话先读本文件，再按需读 `docs/decisions/hocbf_acceleration_aware.md`。

## 1. 一句话状态

Phase 5（PX4/Gazebo 闭环）核心已达成：head-on / crossing / 4 机 multi-UAV
三种场景都满足 `min_rho >= 0`。最终方案是：

```text
sampled-data acceleration barrier（gamma=0.1）
+ SEQUENTIAL_PASS（LLM/rule 决定通行顺序）
+ PX4 velocity-mode offboard
```

现在进入 **Phase 6：Fault Injection**。

## 2. 仓库与环境

- 仓库：`/Users/lijiajun/Documents/drone/AegisAir`（private GitHub，已 push，
  当前 `main` 干净，HEAD=`2ad9703`）。
- Python：`/opt/anaconda3/envs/eai-swarm/bin/python`（conda env `eai-swarm`）。
- PX4 SITL（本机 arm64）：
  `/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/PX4-Autopilot/build/px4_sitl_default/bin/px4`
- Gazebo Harmonic：`/opt/homebrew/bin/gz`（`gz sim -s` headless，`gz sim -g` GUI）。
- 启动脚本：
  `/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0/scripts/`
  - `launch_isolated_sitl.sh S1`
  - `launch_multi_sitl_pose.sh S1 2,3,4,5`
  - `stop_multi_sitl.sh <state_dir>`
  - `gcs_heartbeat.py --port <18572..18575>`
- Docker 镜像：
  - `aegisair-px4-bridge:humble`（已把 `px4_adapter` 打进镜像，避免 macOS 只读卷
    并发读死锁）
  - `safety-margin-gate0-ros2:humble`（ROS 2 Humble + px4_msgs + uXRCE agent）
- MQTT：`amqtt`（`config/amqtt.yml`，绑定 `0.0.0.0:1883`）。
- Qwen 模型：
  `/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench`
  （或 `/Volumes/Expansion/safedrones_marllib_vec/models/Qwen3-4B-4bit`）。
- 外部盘结果：`/Volumes/Expansion/aegisair_phase5_20260816/`。

## 3. 关键文件

- 核心安全层：
  - `swarm/ra/runtime_assurance.py`（velocity-CBF / HOCBF / sampled-data 三模式）
  - `swarm/ra/hocbf.py`（集中式 QP：`solve_acceleration_qp`、
    `solve_sampled_data_qp`）
  - `swarm/ra/margins.py`（`d_safe`，含 `tau_ctrl`）
- 语义恢复层：
  - `swarm/recovery/async_replanner.py`（异步 mission replanner）
  - `swarm/recovery/llm.py`（`MlxLmClient` / `RuleMissionPlanner`）
  - `swarm/recovery/decision.py`（compact decision → RecoveryPlan）
- Phase 5 实验入口：
  - `marllib/phase5_runner.py`（`--sim` / `--mqtt`；`--sampled-data --gamma
    --sequential-pass --urgent-drone --llm`）
  - `marllib/phase5_step_response.py`（P3 速度阶跃标定）
- 适配器：
  - `px4_adapter/node.py`（ROS2↔MQTT，velocity mode，`--control-rate-hz
    --telemetry-rate-hz --origin-offset-ned`）
  - `px4_adapter/docker/Dockerfile` + `run_adapter.sh`
  - `px4_adapter/p5/`（Phase 6 可复用的 fault-scan）

## 4. 决策文档

- `docs/decisions/hocbf_acceleration_aware.md`：HOCBF → sampled-data 的完整演变、
  诊断、SEQUENTIAL_PASS、PX4 结果。
- `docs/decisions/llm_async_mission_replanning.md`：LLM 职责，含 coordination
  deadlock resolution。
- `docs/decisions/phase5_px4_closed_loop.md` / `phase5_live_smoke.md`：Phase 5
  live 记录。

## 5. 可复现命令

```bash
cd /Users/lijiajun/Documents/drone/AegisAir

# 单元测试（98 个）
/opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s tests

# 轻量 4 机：sampled-data + SEQUENTIAL_PASS + LLM priority
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase5_runner.py \
  --scenario multi_uav --mode CBF_ONLY --llm qwen --seeds 1 --max-steps 800 \
  --dt 0.05 --sampled-data --gamma 0.1 --sequential-pass --urgent-drone 2 \
  --qwen-model /Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench \
  --qwen-max-tokens 64
```

### Live 4 机 runbook（按顺序）

```bash
# 1) broker
cd /Users/lijiajun/Documents/drone/AegisAir
/opt/anaconda3/envs/eai-swarm/bin/amqtt -c config/amqtt.yml

# 2) Gazebo + 4 PX4（另一个终端，阻塞）
cd '/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0'
POSES='2=-0.8,-3,0.5;3=0.8,3,0.5;4=0.8,-3,0.5;5=-0.8,3,0.5' \
  bash scripts/launch_multi_sitl_pose.sh S1 2,3,4,5

# 3) GCS heartbeat
for p in 18572 18573 18574 18575; do
  /opt/anaconda3/envs/eai-swarm/bin/python scripts/gcs_heartbeat.py --port $p --duration-s 1800 &
done

# 4) adapter（velocity mode 20Hz，共享原点偏移）
docker rm -f aegisair-adapters >/dev/null 2>&1 || true
docker run -d --name aegisair-adapters \
  -p 127.0.0.1:8889:8889/udp \
  -e XRCE_UDP_PORT=8889 \
  -e ADAPTER_INSTANCES='2 3 4 5' \
  -e ADAPTER_ARGS='--no-read-only --mqtt-host host.docker.internal --mqtt-port 1883 --control-rate-hz 20 --telemetry-rate-hz 20' \
  -e ORIGIN_OFFSET_2='-3,-0.8,0' \
  -e ORIGIN_OFFSET_3='3,0.8,0' \
  -e ORIGIN_OFFSET_4='-3,0.8,0' \
  -e ORIGIN_OFFSET_5='3,-0.8,0' \
  aegisair-px4-bridge:humble

# 5) 可选 GUI
GZ_PARTITION=safety_margin_gate0_multi gz sim -g

# 6) 4 机 sampled-data + SEQUENTIAL_PASS + LLM priority
cd /Users/lijiajun/Documents/drone/AegisAir
/opt/anaconda3/envs/eai-swarm/bin/python marllib/phase5_runner.py \
  --scenario multi_uav --mode CBF_ONLY --llm qwen --mqtt --drone-ids 2,3,4,5 \
  --max-steps 400 --rate-hz 20 --sampled-data --gamma 0.1 --tau-ctrl 0.2 \
  --sequential-pass --urgent-drone 4 \
  --qwen-model /Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench \
  --qwen-max-tokens 64 \
  --starts '2=-3,0.8,2.5;3=3,-0.8,2.5;4=-3,-0.8,2.5;5=3,0.8,2.5' \
  --goals '2=3,-0.8,2.5;3=-3,0.8,2.5;4=3,0.8,2.5;5=-3,-0.8,2.5'
```

注意：4 机 PX4 在这台 16GB 机器上有内存压力，偶发 `Killed: 9` 或单实例启动挂死；
多应用占用大时先释放内存。

## 6. Phase 6 目标

故障注入（plan.md）：

```text
- packet loss
- stale telemetry
- latency
- perception noise
- GCS / offboard loss
- LLM timeout
- invalid LLM command
```

已有可复用基础：`px4_adapter/p5/P5_PLAN.md`（本地 fault-scan 协议）、
`fault_scan.py`、`live_command_fault_scan.py`、`safety_override_fault_scan.py`。

建议顺序（最低成本先行）：

1. 跑 `px4_adapter/p5` 本地 fault-scan，冻结 adapter fail-closed 行为。
2. 单机 live fault injection（packet loss / stale telemetry / latency）。
3. 4 机 sampled-data + SEQUENTIAL_PASS 下的 fault injection，观察
   `min_rho`、collision、完成率、LLM timeout / invalid command 是否被 validator
   和 fallback 兜住。

### 6.1 架构决策：保留 shared-state，但升级成“共享状态估计”

当前 Phase 5 把 PX4 `vehicle_local_position` 直接当共享真值使用
（`perception_sigma=0.1` 固定标量，delay 只进 `M_comm`，无 dropout 维持）。

Phase 6 第一项实现改为：

```text
Retain centralized shared-state assumption;
replace perfect shared ground truth with delayed, dropout-prone,
covariance-bearing shared state estimates.
```

在 MQTT telemetry 与 Runtime Assurance 之间加 `SharedStateEstimator`：

```text
raw telemetry
-> SharedStateEstimator（per-drone 状态估计：position/velocity/P/t_last/dropped）
-> EstimatedState
-> RA（用 covariance + AoI 计算 d_safe）
```

映射到现有 margin：

- covariance -> `perception_margin(sigma_i, sigma_j)`，`sigma_i` 由协方差在
  “两机连线方向”的投影替代；
- delay -> 估计器实际输出延迟后的状态，同时 `M_comm` 继续用 age 放大；
- dropout -> hold 上次 estimate + 协方差随时间增长（或 stale fail-closed）。

这样 Phase 6 的 `perception noise / latency / stale telemetry` 就变成配置
`SharedStateEstimator` 的 delay/dropout/noise 参数，而不是到处改代码。

建议第一 gate：先在轻量 sim 里用 synthetic delay/dropout/covariance 验证 RA
的 `min_rho` 与 fail-closed 行为，再上 live PX4。

## 7. 重要约定

- 文档中文；代码/文档在仓库，checkpoint/结果/日志在外部盘。
- 不放松阈值、不加 seed 救失败假设；
- `Assume AI can fail`：LLM / MARL / Predictor 都不可信，Runtime Assurance
  保留最终硬安全。
- 任何新消息格式先冻结到 `swarm/interfaces.py` 或 `px4_adapter/mqtt_codec.py`
  并加测试。
- 有重大方向更新，需要在 `docs/decisions/` 中说明。

## 8. 新对话第一步建议

先跑：

```bash
cd /Users/lijiajun/Documents/drone/AegisAir
/opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s tests
/opt/anaconda3/envs/eai-swarm/bin/python -m unittest discover -s px4_adapter/tests
```

确认 98 个主测试 + 41 个 adapter 测试全绿后，再进入 Phase 6 的 fault-scan。
