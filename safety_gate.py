from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from swarm.ego import EgoStateStore
from swarm.safety import (
    DroneSnapshot,
    SafetyConfig,
    SafetyGate,
    build_safety_command,
    build_override_event,
    build_status_event,
)


LOGGER = logging.getLogger("safety_gate")


def build_mqtt_client(client_id: str):
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: paho-mqtt. Install it in your active conda env with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based Safety Gate.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--qos", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--safe-distance", type=float, default=1.6)
    parser.add_argument("--escape-distance", type=float, default=1.2)
    parser.add_argument("--high-threshold", type=float, default=0.70)
    parser.add_argument("--low-threshold", type=float, default=0.32)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--override-command-interval", type=float, default=0.5)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--input-mode", choices=["gt", "ego"], default="gt")
    parser.add_argument("--ego-state-ttl-sec", type=float, default=0.6)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.ego_state_ttl_sec <= 0:
        raise SystemExit("--ego-state-ttl-sec must be positive")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SafetyConfig(
        safe_distance_m=args.safe_distance,
        escape_distance_m=args.escape_distance,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        hold_sec=args.hold_sec,
    )
    gate = SafetyGate(config)
    snapshots: dict[int, DroneSnapshot] = {}
    ego_state = EgoStateStore()
    active_overrides: set[int] = set()
    last_override_command_publish: dict[int, float] = {}
    last_status_publish = 0.0
    client = build_mqtt_client(client_id="safety-gate")

    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        LOGGER.info("connected to MQTT broker %s:%s with result=%s", args.host, args.port, reason_code)
        client.subscribe("swarm/drone/+/telemetry", qos=args.qos)
        if args.input_mode == "ego":
            client.subscribe("swarm/drone_seen/+/position", qos=args.qos)

    def on_ego_message(payload: dict[str, Any], topic: str) -> None:
        try:
            ego_state.update(payload)
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("ignored invalid ego position on %s: %s", topic, exc)

    def on_message(client: Any, userdata: Any, message: Any) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            LOGGER.warning("ignored invalid JSON on %s: %s", message.topic, exc)
            return
        if message.topic.endswith("/position") and "/drone_seen/" in f"/{message.topic}":
            on_ego_message(payload, message.topic)
            return
        try:
            snapshot = DroneSnapshot.from_telemetry(payload)
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("ignored invalid telemetry on %s: %s", message.topic, exc)
            return
        snapshots[snapshot.drone_id] = snapshot

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    try:
        while True:
            now = time.monotonic()
            snapshot_list = list(snapshots.values())
            if args.input_mode == "ego":
                snapshot_list = ego_state.snapshots_for_gate(snapshot_list, args.ego_state_ttl_sec, now)
            decisions = gate.evaluate(snapshot_list, now=now)
            if decisions and now - last_status_publish >= args.status_interval:
                for decision in decisions:
                    client.publish(
                        "swarm/commander/status",
                        payload=json.dumps(build_status_event(decision), separators=(",", ":")),
                        qos=args.qos,
                    )
                last_status_publish = now

            for decision in decisions:
                if decision.mode == "override":
                    first_override = decision.drone not in active_overrides
                    last_publish = last_override_command_publish.get(decision.drone, 0.0)
                    should_publish_command = first_override or (
                        now - last_publish >= args.override_command_interval
                    )
                    if should_publish_command:
                        command = build_safety_command(decision)
                        client.publish(
                            f"swarm/drone/{decision.drone}/command",
                            payload=json.dumps(command, separators=(",", ":")),
                            qos=args.qos,
                        )
                        last_override_command_publish[decision.drone] = now

                    if not first_override:
                        continue

                    event = build_override_event(decision)
                    client.publish(
                        "swarm/commander/override",
                        payload=json.dumps(event, separators=(",", ":")),
                        qos=args.qos,
                    )
                    active_overrides.add(decision.drone)
                    LOGGER.warning(
                        "override drone=%s risk=%.3f target=%s",
                        decision.drone,
                        decision.risk_level,
                        decision.target_drone,
                    )

                if decision.mode != "override":
                    active_overrides.discard(decision.drone)
                    last_override_command_publish.pop(decision.drone, None)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        LOGGER.info("stopping safety gate")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
