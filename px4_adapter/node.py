"""Run the SafeDrones PX4 adapter as a ROS 2 node.

In read-only mode the node only subscribes to PX4 state and republishes it over
MQTT; it publishes no control messages.  When ``control.read_only`` is false it
also owns the local safety state machine and command path.

The ROS 2 imports are intentionally lazy so the pure codec/state-machine tests
do not require ``rclpy`` or ``px4_msgs`` on the host.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from px4_adapter.mqtt_codec import (
    Px4Command,
    TelemetryState,
    base_topics,
    decode_command,
    encode_command_ack,
    encode_safedrones_telemetry,
    encode_stale_state,
    encode_telemetry,
    normalize_command_to_ned,
)
from px4_adapter.px4_codec import (
    build_control_plan,
    flu_to_px4_ned,
)
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits


def _load_config(path: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter config must be a YAML mapping")
    return config


def _apply_instance(config: dict[str, Any], instance_id: int) -> dict[str, Any]:
    """Rewrite the instance-derived ROS 2 topics for the selected PX4 id."""
    prefix = f"/px4_{instance_id}/fmu"
    ros = config.setdefault("ros", {})
    ros["namespace_prefix"] = prefix
    ros.setdefault("input", {})["vehicle_local_position"] = (
        f"{prefix}/out/vehicle_local_position"
    )
    ros.setdefault("input", {})["vehicle_status"] = (
        f"{prefix}/out/vehicle_status_v1"
    )
    ros.setdefault("input", {})["vehicle_attitude"] = (
        f"{prefix}/out/vehicle_attitude"
    )
    ros.setdefault("output", {})["offboard_control_mode"] = (
        f"{prefix}/in/offboard_control_mode"
    )
    ros.setdefault("output", {})["trajectory_setpoint"] = (
        f"{prefix}/in/trajectory_setpoint"
    )
    ros.setdefault("output", {})["vehicle_command"] = (
        f"{prefix}/in/vehicle_command"
    )
    control = config.setdefault("control", {})
    control["target_system"] = instance_id + 1
    control["target_component"] = 1
    control["source_system"] = 1
    control["source_component"] = 1
    return config


class MqttBridge:
    """Tiny paho-mqtt wrapper kept separate from the ROS 2 node."""

    def __init__(self, config: dict[str, Any], on_command: Any) -> None:
        self.config = config
        self.on_command = on_command
        self._client = self._build_client()
        self._connected = threading.Event()
        self.topics = base_topics(
            config["mqtt"]["base_topic"],
            int(config["instance_id"]),
        )
        self.safedrones_telemetry_topic = (
            f"swarm/drone/{int(config['instance_id'])}/telemetry"
        )
        self.safedrones_command_topic = (
            f"swarm/drone/{int(config['instance_id'])}/command"
        )

    def _build_client(self) -> Any:
        try:
            import paho.mqtt.client as mqtt
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: paho-mqtt. Install it with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        try:
            return mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"safedrones-px4-adapter-{self.config['instance_id']}",
            )
        except AttributeError:
            return mqtt.Client(
                client_id=f"safedrones-px4-adapter-{self.config['instance_id']}"
            )

    def connect(self) -> None:
        mqtt = self.config["mqtt"]

        def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
            self._connected.set()
            client.subscribe(self.topics["command"], qos=int(mqtt.get("qos", 0)))
            client.subscribe(self.safedrones_command_topic, qos=int(mqtt.get("qos", 0)))

        def on_message(client: Any, userdata: Any, message: Any) -> None:
            try:
                payload = json.loads(message.payload.decode("utf-8"))
                if message.topic == self.safedrones_command_topic:
                    command = decode_command(payload, default_source_frame="FLU")
                else:
                    command = decode_command(payload)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return
            self.on_command(normalize_command_to_ned(command))

        self._client.on_connect = on_connect
        self._client.on_message = on_message
        self._client.connect(str(mqtt["host"]), int(mqtt["port"]), keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(timeout=5):
            raise RuntimeError("adapter failed to connect to MQTT broker")

    def publish(self, topic: str, payload: str) -> None:
        result = self._client.publish(
            topic,
            payload=payload,
            qos=int(self.config["mqtt"].get("qos", 0)),
            retain=False,
        )
        result.wait_for_publish(timeout=5)

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeDrones PX4 adapter ROS 2 node")
    parser.add_argument("--config", default="px4_adapter/config.yaml")
    parser.add_argument("--instance", type=int, default=None)
    parser.add_argument(
        "--read-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--mqtt-host", default=None)
    parser.add_argument("--mqtt-port", type=int, default=None)
    parser.add_argument("--origin-offset-ned", default=None)
    parser.add_argument("--control-rate-hz", type=float, default=None)
    parser.add_argument("--telemetry-rate-hz", type=float, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)

    if args.instance is not None:
        config["instance_id"] = args.instance
    if args.read_only is not None:
        config.setdefault("control", {})["read_only"] = args.read_only
    if args.mqtt_host is not None:
        config.setdefault("mqtt", {})["host"] = args.mqtt_host
    if args.mqtt_port is not None:
        config.setdefault("mqtt", {})["port"] = args.mqtt_port
    if args.origin_offset_ned is not None:
        config.setdefault("telemetry", {})["origin_offset_ned"] = [
            float(v) for v in args.origin_offset_ned.split(",")
        ]
    if args.control_rate_hz is not None:
        config.setdefault("control", {})["rate_hz"] = args.control_rate_hz
    if args.telemetry_rate_hz is not None:
        config.setdefault("telemetry", {})["publish_rate_hz"] = args.telemetry_rate_hz
    _apply_instance(config, int(config["instance_id"]))

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleAttitude,
        VehicleLocalPosition,
        VehicleStatus,
    )

    class Px4AdapterNode(Node):
        def __init__(self) -> None:
            self.config = config
            self.instance_id = int(config["instance_id"])
            super().__init__(f"safedrones_px4_adapter_{self.instance_id}")
            self.source_frame = str(config["telemetry"]["source_frame"])
            self.read_only = bool(config.get("control", {}).get("read_only", True))
            self.last_position: Any = None
            self.last_status: Any = None
            self.last_attitude: Any = None
            self.last_message_arrival_ms: int = 0
            self.last_state: TelemetryState | None = None
            self.active_command: Px4Command | None = None
            self.last_plan_position: tuple[float, float, float] | None = None
            self.last_vehicle_command_id: str | None = None
            self.telemetry_stale_sec = float(
                config["safety"]["telemetry_stale_sec"]
            )

            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            ros = config["ros"]
            self.mode_pub = self.create_publisher(
                OffboardControlMode, ros["output"]["offboard_control_mode"], qos
            )
            self.point_pub = self.create_publisher(
                TrajectorySetpoint, ros["output"]["trajectory_setpoint"], qos
            )
            self.command_pub = self.create_publisher(
                VehicleCommand, ros["output"]["vehicle_command"], qos
            )
            self.create_subscription(
                VehicleLocalPosition,
                ros["input"]["vehicle_local_position"],
                self.on_local_position,
                qos,
            )
            self.create_subscription(
                VehicleStatus,
                ros["input"]["vehicle_status"],
                self.on_status,
                qos,
            )
            self.create_subscription(
                VehicleAttitude,
                ros["input"]["vehicle_attitude"],
                self.on_attitude,
                qos,
            )

            safety = config["safety"]
            self.safety = LocalSafetyState(
                SafetyLimits(
                    command_ttl_sec=float(safety["command_ttl_sec"]),
                    telemetry_stale_sec=float(safety["telemetry_stale_sec"]),
                    max_horizontal_speed_mps=float(safety["max_horizontal_speed_mps"]),
                    max_vertical_speed_mps=float(safety["max_vertical_speed_mps"]),
                    min_altitude_m=float(safety["min_altitude_m"]),
                    max_altitude_m=float(safety["max_altitude_m"]),
                    max_position_distance_m=float(safety["max_position_distance_m"]),
                    fail_closed_action=str(safety.get("fail_closed_action", "LAND")),
                )
            )
            self.mqtt = MqttBridge(config, self.on_command)
            self.mqtt.connect()
            self.create_timer(
                1.0 / float(config["telemetry"]["publish_rate_hz"]),
                self.telemetry_tick,
            )
            if not self.read_only:
                self.create_timer(
                    1.0 / float(config["control"].get("rate_hz", 10.0)),
                    self.control_tick,
                )

        def on_local_position(self, msg: Any) -> None:
            if self.last_position is None:
                self.get_logger().info("received first vehicle_local_position")
            self.last_position = msg
            self.last_message_arrival_ms = time.time_ns() // 1_000_000

        def on_status(self, msg: Any) -> None:
            if self.last_status is None:
                self.get_logger().info("received first vehicle_status")
            self.last_status = msg
            self.last_message_arrival_ms = time.time_ns() // 1_000_000

        def on_attitude(self, msg: Any) -> None:
            self.last_attitude = msg

        def yaw_rad(self) -> float:
            if self.last_attitude is None:
                return 0.0
            q = self.last_attitude
            # NED yaw from quaternion (heading about z-down).
            w, x, y, z = float(q.q[0]), float(q.q[1]), float(q.q[2]), float(q.q[3])
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            ned_yaw = math.atan2(siny_cosp, cosy_cosp)
            # Publish FLU heading (negate NED yaw for the left-handed y-axis).
            return -ned_yaw

        def on_command(self, command: Px4Command) -> None:
            self.active_command = command

        def now_us(self) -> int:
            return self.get_clock().now().nanoseconds // 1000

        def build_state(self) -> TelemetryState | None:
            if self.last_status is None or self.last_position is None:
                return None
            now_ms = time.time_ns() // 1_000_000
            if now_ms - self.last_message_arrival_ms > self.telemetry_stale_sec * 1000.0:
                return None
            position = self.last_position
            status = self.last_status
            origin = tuple(
                float(v) for v in self.config["telemetry"].get("origin_offset_ned", [0.0, 0.0, 0.0])
            )
            return TelemetryState(
                instance_id=self.instance_id,
                source_frame=self.source_frame,
                timestamp_ms=now_ms,
                source_timestamp_us=int(position.timestamp),
                armed=int(status.arming_state) == VehicleStatus.ARMING_STATE_ARMED,
                nav_state=int(status.nav_state),
                failsafe=bool(status.failsafe),
                connection_lost=bool(status.gcs_connection_lost),
                position=(
                    float(position.x) + origin[0],
                    float(position.y) + origin[1],
                    float(position.z) + origin[2],
                ),
                velocity=(float(position.vx), float(position.vy), float(position.vz)),
                yaw=self.yaw_rad(),
                xy_valid=bool(position.xy_valid),
                z_valid=bool(position.z_valid),
                v_xy_valid=bool(position.v_xy_valid),
                v_z_valid=bool(position.v_z_valid),
            )

        def telemetry_tick(self) -> None:
            state = self.build_state()
            if state is None:
                now_ms = time.time_ns() // 1_000_000
                if (
                    self.last_status is not None
                    and now_ms - self.last_message_arrival_ms
                    > self.telemetry_stale_sec * 1000.0
                ):
                    source_timestamp_us = (
                        int(self.last_position.timestamp)
                        if self.last_position is not None
                        else None
                    )
                    self.mqtt.publish(
                        self.mqtt.topics["state"],
                        encode_stale_state(
                            self.instance_id,
                            self.source_frame,
                            source_timestamp_us,
                        ),
                    )
                return
            self.last_state = state
            payload = encode_telemetry(state)
            self.mqtt.publish(self.mqtt.topics["telemetry"], payload)
            self.mqtt.publish(self.mqtt.topics["state"], payload)
            self.mqtt.publish(
                self.mqtt.safedrones_telemetry_topic,
                encode_safedrones_telemetry(state),
            )

        def _convert_target(self, command: Px4Command) -> tuple[float, float, float] | None:
            if command.target is None:
                return None
            if command.source_frame == "PX4_NED":
                return command.target
            return flu_to_px4_ned(*command.target)

        def _convert_velocity(self, command: Px4Command) -> tuple[float, float, float] | None:
            if command.velocity is None:
                return None
            if command.source_frame == "PX4_NED":
                return command.velocity
            return flu_to_px4_ned(*command.velocity)

        def control_tick(self) -> None:
            now_ms = time.time_ns() // 1_000_000
            state = self.last_state or self.build_state()
            decision = self.safety.evaluate(now_ms, state, self.active_command)
            if self.active_command is not None:
                self.mqtt.publish(
                    self.mqtt.topics["ack"],
                    encode_command_ack(
                        self.instance_id,
                        self.active_command.command_id,
                        decision.allowed,
                        decision.reason,
                        decision.state,
                    ),
                )

            if not decision.allowed or self.active_command is None:
                if state is not None and self.last_plan_position is not None:
                    hold_plan = build_control_plan(
                        "hold",
                        current_position=self.last_plan_position,
                        target_system=int(self.config["control"].get("target_system", self.instance_id + 1)),
                        target_component=int(self.config["control"].get("target_component", 1)),
                        source_system=int(self.config["control"].get("source_system", 1)),
                        source_component=int(self.config["control"].get("source_component", 1)),
                    )
                    self._publish_plan(hold_plan, send_commands=False)
                return

            command = self.active_command
            current_position = None
            if state is not None:
                current_position = state.position
            takeoff_altitude_m = (
                command.altitude_m
                if command.altitude_m is not None
                else float(self.config["control"].get("takeoff_altitude_m", 2.5))
            )
            plan = build_control_plan(
                command.action,
                target=self._convert_target(command),
                velocity=self._convert_velocity(command),
                yaw=command.yaw,
                takeoff_altitude_m=takeoff_altitude_m,
                current_position=current_position,
                target_system=int(self.config["control"].get("target_system", self.instance_id + 1)),
                target_component=int(self.config["control"].get("target_component", 1)),
                source_system=int(self.config["control"].get("source_system", 1)),
                source_component=int(self.config["control"].get("source_component", 1)),
            )
            send_commands = (
                self.active_command.command_id != self.last_vehicle_command_id
            )
            self._publish_plan(plan, send_commands=send_commands)
            if send_commands:
                self.last_vehicle_command_id = self.active_command.command_id

        def _publish_plan(self, plan: Any, send_commands: bool = True) -> None:
            now = self.now_us()
            mode = OffboardControlMode()
            mode.timestamp = now
            mode.position = bool(plan.offboard.position)
            mode.velocity = bool(plan.offboard.velocity)
            mode.acceleration = bool(plan.offboard.acceleration)
            mode.attitude = bool(plan.offboard.attitude)
            mode.body_rate = bool(plan.offboard.body_rate)
            self.mode_pub.publish(mode)

            point = TrajectorySetpoint()
            point.timestamp = now
            nan = float("nan")
            position = plan.trajectory.position or (nan, nan, nan)
            velocity = plan.trajectory.velocity or (nan, nan, nan)
            acceleration = plan.trajectory.acceleration or (nan, nan, nan)
            point.position = [float(v) for v in position]
            point.velocity = [float(v) for v in velocity]
            point.acceleration = [float(v) for v in acceleration]
            point.yaw = float(plan.trajectory.yaw)
            point.yawspeed = float(plan.trajectory.yawspeed)
            self.point_pub.publish(point)
            if plan.trajectory.position is not None:
                self.last_plan_position = tuple(float(v) for v in plan.trajectory.position)

            if send_commands:
                for spec in plan.commands:
                    msg = VehicleCommand()
                    msg.timestamp = now
                    msg.command = int(spec.command)
                    msg.param1 = float(spec.param1)
                    msg.param2 = float(spec.param2)
                    msg.param3 = float(spec.param3)
                    msg.param4 = float(spec.param4)
                    msg.param5 = float(spec.param5)
                    msg.param6 = float(spec.param6)
                    msg.param7 = float(spec.param7)
                    msg.target_system = int(spec.target_system)
                    msg.target_component = int(spec.target_component)
                    msg.source_system = int(spec.source_system)
                    msg.source_component = int(spec.source_component)
                    msg.from_external = bool(spec.from_external)
                    self.command_pub.publish(msg)

    rclpy.init()
    node = Px4AdapterNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.mqtt.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
