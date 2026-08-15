# AegisAir

**Adaptive Runtime Assurance for Intelligent Multi-UAV Systems**

AegisAir 是从 SafeDrones 原型分离并升级后的研究系统。核心假设是 **LLM 与 MARL
都不可信**，由独立的 **Risk-Adaptive Runtime Assurance** 保留最终硬安全权，并
通过预测安全裕度与语义恢复在保证硬安全的前提下恢复任务性能。

## 研究方向

```text
Mission / User
   ↓
Local LLM Commander
   ↓ 高层目标 / 恢复计划
MARL Nominal Pilot
   ↓ u_nom
Risk-Adaptive Runtime Assurance
   ├── Dynamic Safety Boundary（dynamics + perception + communication margin）
   ├── Normalized Safety Margin ρ
   ├── Safety-Margin Degradation g
   ├── CV / CPA Predictive Monitor
   └── Time-Varying CBF / QP
   ↓
PX4 Offboard → PX4 SITL / Gazebo
   ↑
   └── Structured RiskEvent → Local LLM Semantic Recovery
```

闭环逻辑：`Risk → Margin → Prediction → Runtime Assurance → Semantic Recovery`。

## 与 SafeDrones 的边界

本仓库只保留升级后的主线：

- 冻结接口（Phase 0）
- 轻量 MAPPO 训练（Phase 1）
- Runtime Assurance / 感知 / ego 状态基础模块
- PX4/Gazebo 闭环（Phase 5 基础）

早期的 MockDrone / Unity 可视化、旧 Safety Gate 双向协议、stage1–5、group1–4、
DeepSeek LLM、DINOv3 训练等历史原型不再保留在 AegisAir 中。

## 目录

```text
plan.md             权威研究计划（v1.0）
marllib/            轻量多机 MARL 环境 + MAPPO（Phase 1）
swarm/              接口、几何、安全、感知、ego 状态
px4_adapter/        PX4 SITL/Gazebo/ROS 2 适配器（Phase 5）
schemas/            冻结接口的 JSON Schema（Phase 0）
examples/           冻结接口示例载荷
docs/interfaces/    冻结接口规范
scripts/            辅助脚本
tests/              单元测试
```

## 当前进度

- Phase 0：7 个跨层接口已冻结（Telemetry / Observation / MarlAction / RiskEvent /
  RecoveryPlan / SafetyDecision / LogEvent）。
- Phase 1：轻量 MAPPO 环境与训练器可用；head-on 对穿收敛（碰撞 0，双机到齐）。
- PX4 C3 head-on：10 seed × 4 模式（GT / 假 ego / 真双目 ego / 无门控）已完成，
  真双目 ego 碰撞率 0/10、近失 1/10。

## 训练（Phase 1）

训练数据默认写入移动硬盘：

```bash
python marllib/train_vec.py --scenario head_on --seed 1 \
  --total-steps 2000000 --num-envs 64 \
  --output /Volumes/Expansion/safedrones_marllib
```

有 CUDA 时自动使用 GPU（或显式 `--device cuda`）。

## 测试

```bash
python -m unittest discover -s tests
```
