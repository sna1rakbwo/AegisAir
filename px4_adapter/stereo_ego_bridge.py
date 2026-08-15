#!/usr/bin/env python3
"""Stereo measurement → ego-position MQTT bridge.

This is the PX4 replacement for the older fake-ego/ego-bridge component. It
consumes a rectified stereo measurement and the observing drone's telemetry,
then publishes an absolute FLU ego position for the observed drone.

For a first deployment the observer yaw may be supplied as a fixed argument;
full integration should read yaw from PX4 ``vehicle_attitude`` and transform it
into the same FLU convention used by SafeDrones.
"""
from __future__ import annotations

import argparse
import json
import math
import time

from px4_adapter.stereo_ego import ObserverPose, StereoMeasurement, world_position_from_measurement
from swarm.stereo import calibration_from_horizontal_fov


def main() -> None:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: paho-mqtt") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--observer", type=int, default=2)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--horizontal-fov-deg", type=float, default=70.0)
    parser.add_argument("--baseline-m", type=float, default=0.12)
    args = parser.parse_args()

    calibration = calibration_from_horizontal_fov(
        args.width,
        args.height,
        args.horizontal_fov_deg,
        args.baseline_m,
    )
    observer_position = (0.0, 0.0, 0.0)
    observer_yaw_rad = math.radians(args.yaw_deg)

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe(f"px4/stereo/{args.observer}/measurement")
        client.subscribe(f"swarm/drone/{args.observer}/telemetry")

    def on_message(client, userdata, msg):
        nonlocal observer_position, observer_yaw_rad
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        if msg.topic.endswith("/telemetry"):
            if payload.get("position"):
                observer_position = tuple(float(v) for v in payload["position"])
            if payload.get("yaw_deg") is not None:
                observer_yaw_rad = math.radians(float(payload["yaw_deg"]))
            return

        try:
            measurement = StereoMeasurement(
                target_drone=int(payload["target_drone"]),
                u_px=float(payload["u_px"]),
                v_px=float(payload["v_px"]),
                disparity_px=float(payload["disparity_px"]),
                timestamp_ms=int(payload.get("timestamp_ms") or int(time.time() * 1000)),
                uncertainty_m=float(payload["uncertainty_m"]) if payload.get("uncertainty_m") is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return

        estimate = world_position_from_measurement(
            measurement,
            calibration,
            ObserverPose(observer_position, observer_yaw_rad),
        )
        output = {
            "drone": estimate.target_drone,
            "position": list(estimate.position_flu),
            "estimated_depth": estimate.depth_m,
            "depth_uncertainty_m": estimate.uncertainty_m,
            "source": "px4_stereo_ego",
            "timestamp_ms": estimate.timestamp_ms,
        }
        client.publish(
            f"swarm/drone_seen/{estimate.target_drone}/position",
            json.dumps(output, separators=(",", ":")),
        )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"px4-stereo-ego-{args.observer}",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
