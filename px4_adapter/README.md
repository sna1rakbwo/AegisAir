# AegisAir PX4 Adapter

PX4 SITL/Gazebo/ROS 2 接入层，负责 AegisAir MQTT 命令与 PX4 飞控之间的本地
安全边界。

## 当前状态

- `config.yaml`：运行配置，正式运行前必须冻结。
- `mqtt_codec.py`：MQTT topic、遥测与命令编解码，保留 `source_frame`。
- `px4_codec.py`：PX4 NED ↔ 内部 FLU 坐标变换，以及控制消息规格。
- `safety_state.py`：本地 fail-closed 状态机。
- `node.py`：ROS 2 节点；`control.read_only=true` 时只发布遥测，不发控制。
- `tests/`：纯逻辑单元测试，不依赖 ROS 2 / paho-mqtt。

## 测试

```bash
python -m unittest discover -s px4_adapter/tests -v
```

## 运行只读遥测节点

节点需要 `rclpy` 和 `px4_msgs`，这两个依赖只在 ROS 2 bridge 容器内提供。
在容器内把 `px4_adapter` 挂载后运行：

```bash
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
python3 /work/px4_adapter/node.py --config /work/px4_adapter/config.yaml
```

当前 `config.yaml` 的 `control.read_only` 为 `true`，因此该节点不会向 PX4
发布 offboard / vehicle command，只验证遥测链路。
