#!/usr/bin/env python3
"""等待指定 PX4 adapter 的四机遥测就绪；超时返回非零。"""

from __future__ import annotations

import argparse
import json
import time

import paho.mqtt.client as mqtt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--drone-ids", nargs="+", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args = parser.parse_args()
    required = set(args.drone_ids)
    received: set[int] = set()

    def on_connect(client: mqtt.Client, _userdata: object, _flags: object, _reason: object, _properties: object = None) -> None:
        client.subscribe("swarm/drone/+/telemetry", qos=0)

    def on_message(_client: mqtt.Client, _userdata: object, message: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            drone_id = int(payload["drone"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if drone_id in required:
            received.add(drone_id)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="aegisair-telemetry-readiness")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=15)
    client.loop_start()
    try:
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline and received != required:
            time.sleep(0.1)
    finally:
        client.loop_stop()
        client.disconnect()
    if received != required:
        missing = sorted(required - received)
        print(f"telemetry readiness timeout; missing={missing}")
        return 1
    print(f"telemetry ready for drone_ids={sorted(received)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
