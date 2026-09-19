# AegisAir 可复现性说明

## 本公开版本包含什么

本仓库提供论文最终采用的运行时安全保障、确定性任务恢复、控制器配置和离线测试。它支持复核算法逻辑、接口契约与冻结配置的一致性。

可以用相同配置重新运行新的独立实验。

## 最小可复现检查

```bash
conda env create -f environment.yml
conda activate eai-swarm
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/validate_pcbf_baseline.py
```

该检查不需要 GPU、PX4、Gazebo 或原始数据。测试覆盖加速度约束的 HOCBF/QP、不可行时制动回退、确定性任务恢复、消息 schema 以及关键 manifest 的结构。PCBF 验证使用 CasADi/IPOPT 运行确定性的两阶段非线性规划检查。

## 闭环仿真复现

完整 PX4/Gazebo 复现另需安装 PX4 SITL、Gazebo、ROS 2、MQTT broker 和 AegisAir adapter bridge。服务启动后，以冻结 manifest 运行对应 runner，并使用一个不存在的新输出目录：

```bash
python marllib/run_c1_sota_cbf_gazebo.py \
  --manifest configs/c1_hocbf_v4_px4_validation_v1.json \
  --out-dir /path/to/new-output
```

原始实验的 checkpoint、日志和轨迹未公开；运行结果应被视为新的复现实验，不应冒充为历史封存结果。

## 结果与失败的记录

有效碰撞、`min_rho <= 0`、runtime-assurance bypass、任务未完成和权限撤销失败均应作为有效结果保留。启动或 arm 前失败应单独标记，不应混入算法试验统计。
