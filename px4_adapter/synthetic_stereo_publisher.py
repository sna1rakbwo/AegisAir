#!/usr/bin/env python3
"""Plan-B synthetic stereo publisher.

Subscribes to adapter-published ``swarm/drone/+/telemetry`` and renders
left/right synthetic camera images with OpenCV, then publishes the Group 2
stereo MQTT topics. Camera pose is derived from telemetry position and a fixed
or telemetry-provided yaw.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2

from px4_adapter.synthetic_stereo import SyntheticObject, render_stereo_pair
from swarm.stereo import calibration_from_horizontal_fov


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def main() -> None:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: paho-mqtt") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--camera-drones", default="2,3")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--horizontal-fov-deg", type=float, default=70.0)
    parser.add_argument("--baseline-m", type=float, default=0.12)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--target-position", default="0,0,0")
    args = parser.parse_args()

    camera_drones = [int(value) for value in args.camera_drones.split(",") if value]
    calibration = calibration_from_horizontal_fov(
        args.width, args.height, args.horizontal_fov_deg, args.baseline_m
    )
    telemetry: dict[int, dict] = {}

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe("swarm/drone/+/telemetry")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            drone_id = int(payload.get("drone") or 0)
            if drone_id > 0:
                telemetry[drone_id] = payload
        except Exception:
            return

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="synthetic-stereo-publisher")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    target_position = tuple(float(v) for v in args.target_position.split(","))
    period_s = 1.0 / max(0.5, args.rate_hz)
    next_publish = 0.0

    try:
        while True:
            now = time.monotonic()
            if now < next_publish:
                time.sleep(0.01)
                continue
            next_publish = now + period_s

            for camera_drone in camera_drones:
                cam = telemetry.get(camera_drone)
                if cam is None or not cam.get("position"):
                    continue
                cam_pos = tuple(float(v) for v in cam["position"])
                yaw_deg = float(cam.get("yaw_deg", args.yaw_deg))
                yaw_rad = math.radians(yaw_deg)
                cam_forward = (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
                cam_up = (0.0, 0.0, 1.0)

                objects: list[SyntheticObject] = [
                    SyntheticObject(target_position, (0, 0, 255), radius_m=0.35)
                ]
                for drone_id, other in telemetry.items():
                    if drone_id == camera_drone or not other.get("position"):
                        continue
                    color = (
                        (0, 0, 255) if drone_id % 4 == 1 else
                        (255, 0, 0) if drone_id % 4 == 2 else
                        (0, 255, 0) if drone_id % 4 == 3 else
                        (0, 255, 255)
                    )
                    objects.append(
                        SyntheticObject(
                            tuple(float(v) for v in other["position"]),
                            color,
                            radius_m=0.35,
                        )
                    )

                left, right = render_stereo_pair(objects, cam_pos, cam_forward, cam_up, calibration)
                ok_left, left_jpeg = cv2.imencode(".jpg", left, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                ok_right, right_jpeg = cv2.imencode(".jpg", right, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok_left or not ok_right:
                    continue

                vertical_fov_deg = 2.0 * math.degrees(
                    math.atan(
                        math.tan(math.radians(args.horizontal_fov_deg) / 2.0)
                        * args.height
                        / args.width
                    )
                )
                meta = {
                    "drone_id": camera_drone,
                    "frame_id": int(time.time() * 10),
                    "timestamp_ms": now_ms(),
                    "width": args.width,
                    "height": args.height,
                    "fov_deg": vertical_fov_deg,
                    "is_stereo": True,
                    "baseline_m": args.baseline_m,
                    "cam_pos_xyz": list(cam_pos),
                    "cam_forward_xyz": list(cam_forward),
                    "cam_up_xyz": list(cam_up),
                }
                client.publish(f"swarm/cam/drone/{camera_drone}/meta", json.dumps(meta, separators=(",", ":")))
                client.publish(f"swarm/cam/drone/{camera_drone}/left/frame", left_jpeg.tobytes())
                client.publish(f"swarm/cam/drone/{camera_drone}/right/frame", right_jpeg.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
