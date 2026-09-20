# AegisAir

本文复现实验代码对应《Control-Authority Reserve and Selective Mission Recovery for Multi-UAV Runtime Assurance》。名义控制为确定性的目标比例速度指令；HOCBF/PB-CBF 运行时安全层保留最终命令权，任务恢复使用确定性规则。

本公开仓库包含源代码、接口 Schema、冻结配置与测试，不包含论文、投稿材料、训练 checkpoint、PX4/Gazebo 日志、轨迹及其他实验产物。

## 快速开始

```bash
conda env create -f environment.yml
conda activate eai-swarm
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/validate_pcbf_baseline.py
python scripts/validate_dynamic_admission.py
```

测试套件是不依赖硬件的最小复现入口，可检验 barrier/QP 安全层、fail-closed 任务恢复、接口校验和冻结 manifest 一致性，无需原始实验数据。PCBF 验证脚本另外检查安全状态的零值、闭环恢复时的值函数下降、随机状态约束残差与求解时延；动态接纳验证脚本检查连续失效轨迹、速度相关决策和一次性准入时延。

## 目录说明

```text
swarm/          运行时安全保障、安全几何、任务恢复与跨层接口
marllib/        论文的轻量仿真与 PX4/Gazebo 实验运行器（保留原目录名）
px4_adapter/    PX4 SITL、Gazebo、ROS 2 与 MQTT 适配代码
schemas/        版本化 JSON Schema
configs/        冻结实验 manifest 与控制器配置
examples/       合法接口载荷示例
tests/          离线单元测试与协议测试
docs/           接口规范
scripts/        分析与受控实验辅助脚本
```

## 论文实验范围

公开版本只保留下列论文中的实验代码、冻结配置及其依赖：

- 双机早期可行性恢复主对比、reserve/prediction 消融及 PCBF 对比；
- 执行模型敏感性和冻结参数下的 OOD 检查；
- 双机失效后的确定性任务恢复与选择性准入；
- 四机分组走廊闭环扩展。


## PX4/Gazebo 闭环复现

闭环运行另需安装 PX4 SITL、Gazebo、ROS 2、MQTT broker 及 AegisAir adapter bridge。服务启动后，使用冻结 manifest 与一个不存在的新输出目录运行，例如：

```bash
python marllib/run_c1_sota_cbf_gazebo.py \
  --manifest configs/c1_hocbf_v4_px4_validation_v1.json \
  --out-dir /path/to/new-output
```

其他正式实验使用 `configs/` 内对应的 sealed manifest 和同目录的 runner。运行器会拒绝覆盖已有输出目录。冻结 manifest 记录实验设置，但不包含历史轨迹、日志或训练后的模型 checkpoint。

任务接纳的 v1/v2/v3 manifest 只保留为历史协议记录；当前 runner 仅接受 `dynamic_admission_v4_nonblocking_bridge` calibration/qualification manifest，避免用新实现生成带旧协议标识的结果。v4 保持 v3 的几何、seed、条件顺序和算法参数不变，并将 adapter 的 MQTT 发布改为非阻塞入队，避免单线程 ROS executor 因同步等待网络线程而饿死遥测回调。新的 sealed-v4 manifest 只能在 v4 calibration 和 qualification 通过后生成。

完整 PX4/Gazebo 环境与单个冻结条件的启动方式见 [docs/PX4_GAZEBO_REPRODUCTION.md](docs/PX4_GAZEBO_REPRODUCTION.md)。

## 可复现性边界

请阅读 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，其中说明本公开版本支持的检查、外部依赖以及不能从该仓库重建的历史实验部分。
