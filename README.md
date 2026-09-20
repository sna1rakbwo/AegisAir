# AegisAir

Adaptive Runtime Assurance for intelligent multi-UAV systems. The nominal MARL
pilot and optional language-model recovery component are treated as untrusted;
the independent runtime-assurance layer retains the final safety decision.

This public repository contains source code, interface schemas, frozen
configuration manifests, and tests. It intentionally excludes manuscripts,
submission files, trained checkpoints, PX4/Gazebo logs, trajectories, and other
experiment artefacts.

## Quick start

```bash
conda env create -f environment.yml
conda activate eai-swarm
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/validate_pcbf_baseline.py
python scripts/validate_dynamic_admission.py
```

测试套件是不依赖硬件的最小复现入口，可检验 barrier/QP 安全层、fail-closed 任务恢复、接口校验和冻结 manifest 一致性，无需原始实验数据。PCBF 验证脚本另外检查安全状态的零值、闭环恢复时的值函数下降、随机状态约束残差与求解时延；动态接纳验证脚本检查连续失效轨迹、速度相关决策和一次性准入时延。

## Repository layout

```text
swarm/          Runtime assurance, safety geometry, recovery, and interfaces
marllib/        Lightweight multi-UAV MARL environments and runners
px4_adapter/    PX4 SITL, Gazebo, ROS 2, and MQTT integration code
schemas/        Versioned JSON schemas for cross-layer messages
configs/        Frozen experiment manifests and controller settings
examples/       Valid interface payload examples
tests/          Offline unit and protocol tests
docs/           Interface specifications and experiment decisions
scripts/        Analysis and controlled experiment helpers
```

## PX4/Gazebo reproduction

The closed-loop runners require a separately installed PX4 SITL/Gazebo/ROS 2
stack plus an MQTT broker and the AegisAir adapter bridge. Once those services
are running, use a manifest and a new output directory, for example:

```bash
python marllib/run_c3_gazebo.py \
  --manifest configs/c3_closed_loop_smoke_v1.json \
  --out-dir /path/to/new-output
```

其他正式实验使用 `configs/` 内对应的 sealed manifest 和同目录的 runner。运行器会拒绝覆盖已有输出目录。冻结 manifest 记录实验设置，但不包含历史轨迹、日志或训练后的模型 checkpoint。

任务接纳的 v1/v2/v3 manifest 只保留为历史协议记录；当前 runner 仅接受 `dynamic_admission_v4_nonblocking_bridge` calibration/qualification manifest，避免用新实现生成带旧协议标识的结果。v4 保持 v3 的几何、seed、条件顺序和算法参数不变，并将 adapter 的 MQTT 发布改为非阻塞入队，避免单线程 ROS executor 因同步等待网络线程而饿死遥测回调。新的 sealed-v4 manifest 只能在 v4 calibration 和 qualification 通过后生成。

完整 PX4/Gazebo 环境与单个冻结条件的启动方式见 [docs/PX4_GAZEBO_REPRODUCTION.md](docs/PX4_GAZEBO_REPRODUCTION.md)。

## Reproducibility boundary

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for supported checks, external
dependencies, and the scope of what this code release can and cannot reproduce.
