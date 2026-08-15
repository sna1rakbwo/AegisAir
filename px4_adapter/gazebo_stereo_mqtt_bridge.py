#!/usr/bin/env python3
"""One-shot Gazebo stereo image topic -> MQTT bridge.

This is the first Plan-A bridge. It reads one left and one right camera image
message with ``gz topic -e -n 1``, converts R8G8B8 pixels to JPEG, and publishes
them to the Group 2 MQTT topics together with a fixed camera metadata message.

For a continuous bridge, replace the one-shot capture with a long-running
``gz topic -e -d`` parser or the native Gazebo Transport Python API.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import cv2
import numpy as np


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def capture_image(topic: str, partition: str) -> tuple[int, int, np.ndarray]:
    env = dict(os.environ, GZ_PARTITION=partition)
    text = subprocess.run(
        ["gz", "topic", "-e", "-t", topic, "-n", "1"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    return parse_gz_image(text)


def parse_gz_image(text: str) -> tuple[int, int, np.ndarray]:
    width = int(re.search(r"width:\s*(\d+)", text).group(1))
    height = int(re.search(r"height:\s*(\d+)", text).group(1))
    data_match = re.search(r'data:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if data_match is None:
        raise RuntimeError("could not find image data field")
    raw = data_match.group(1).encode("utf-8").decode("unicode_escape").encode("latin1")
    expected = width * height * 3
    if len(raw) < expected:
        raise RuntimeError(f"short image payload: got {len(raw)}, expected {expected}")
    array = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, width, 3))
    return width, height, array


def rgb_to_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()


def main() -> None:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: paho-mqtt") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--left-topic", required=True)
    parser.add_argument("--right-topic", required=True)
    parser.add_argument("--partition", default="stereo_smoke")
    parser.add_argument("--camera-drone", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--baseline-m", type=float, default=0.12)
    parser.add_argument("--cam-pos", default="0,0,1.5")
    parser.add_argument("--cam-forward", default="0,0,-1")
    parser.add_argument("--cam-up", default="0,1,0")
    args = parser.parse_args()

    left_width, left_height, left_rgb = capture_image(args.left_topic, args.partition)
    right_width, right_height, right_rgb = capture_image(args.right_topic, args.partition)
    if (left_width, left_height) != (right_width, right_height):
        raise SystemExit("left and right image dimensions differ")

    left_jpeg = rgb_to_jpeg(left_rgb)
    right_jpeg = rgb_to_jpeg(right_rgb)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"gazebo-stereo-{args.camera_drone}")
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    meta = {
        "drone_id": args.camera_drone,
        "frame_id": 0,
        "timestamp_ms": now_ms(),
        "width": left_width,
        "height": left_height,
        "fov_deg": args.fov_deg,
        "is_stereo": True,
        "baseline_m": args.baseline_m,
        "cam_pos_xyz": [float(v) for v in args.cam_pos.split(",")],
        "cam_forward_xyz": [float(v) for v in args.cam_forward.split(",")],
        "cam_up_xyz": [float(v) for v in args.cam_up.split(",")],
    }
    client.publish(f"swarm/cam/drone/{args.camera_drone}/meta", json.dumps(meta, separators=(",", ":")))
    client.publish(f"swarm/cam/drone/{args.camera_drone}/left/frame", left_jpeg)
    client.publish(f"swarm/cam/drone/{args.camera_drone}/right/frame", right_jpeg)
    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()
    print(json.dumps({"camera_drone": args.camera_drone, "left_bytes": len(left_jpeg), "right_bytes": len(right_jpeg)}))


if __name__ == "__main__":
    main()
