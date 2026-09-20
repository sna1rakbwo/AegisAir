# PX4/Gazebo 闭环复现

本目录使论文的 PX4/Gazebo 闭环路径不再依赖作者电脑上的私有目录或私有镜像。

## 前提

在 Linux 或能运行 PX4 SITL/Gazebo 的主机上，准备 Docker Compose v2、PX4-Autopilot **v1.16.0** 源码及其 Gazebo SITL 依赖，以及 Python 3.11 和本仓库的 `requirements.txt`。

```bash
git clone --recursive --branch v1.16.0 https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
bash Tools/setup/ubuntu.sh
make px4_sitl
export PX4_DIR="$PWD"
```

macOS 可用于离线测试；闭环建议使用 Linux，因为 PX4/Gazebo 的图形和 UDP/DDS 网络链路依赖主机环境。

## 启动 bridge

```bash
docker compose up --build -d
docker compose logs -f bridge
```

Compose 会启动 MQTT broker 与 ROS 2 bridge。bridge 镜像从公开固定提交构建 Micro-XRCE-DDS-Agent 和 `px4_msgs`，并把 UDP 8889 暴露给主机 PX4 SITL。

## 运行一个冻结条件

```bash
export PX4_DIR=/absolute/path/to/PX4-Autopilot
bash scripts/run_c1_paper_trial.sh \
  configs/c1_hocbf_v4_px4_validation_v1.json c1v4_01 AEGIS_HOCBF_V4 \
  /absolute/path/to/new-output
```

该脚本启动两机 S1 世界、发送 GCS heartbeat、等待 MQTT 遥测，再运行对应 runner；结束时只停止它启动的 PID。完整试验需要对冻结 manifest 内每个 `trial_id` 与 `condition_order` 重复调用，并保留每次输出。

修复后的任务接纳 calibration-v4 使用独立入口：

```bash
export PX4_DIR=/absolute/path/to/PX4-Autopilot
bash scripts/run_recoverability_admission_trial.sh \
  configs/c_recoverability_admission_calibration_v4.json \
  radm4_lateral_dev RECOVERABILITY_ADMISSION_RA \
  /absolute/path/to/new-output
```

先完成 calibration-v4 的所有冻结条件并运行分析器；只有结果为 `GO` 才能启动 qualification-v4。v4 将 adapter MQTT 发布冻结为非阻塞入队；旧 manifest 只保留为历史记录，当前任务接纳 runner 不接受旧协议标识。

## 边界

公开仓库不含历史 PX4/Gazebo 轨迹、日志或 checkpoint。重新运行得到的是新的复现实验，不能替代论文中封存的数值。
