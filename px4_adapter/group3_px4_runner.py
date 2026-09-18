#!/usr/bin/env python3
"""Minimal Group 3 crossing benchmark against the migrated PX4 system.

This runner assumes the external PX4/adapter/broker/synthetic-stereo/Safety
Gate stack is already running. It only issues task commands and records
telemetry, so it does not start mock drones.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on_crossing")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--phase", choices=("prepare", "cross"), default="cross")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lateral = rng.uniform(-0.15, 0.15)
    altitude = 1.0
    if args.scenario == "head_on_crossing":
        start = {2: (-3.0, lateral, altitude), 3: (3.0, -lateral, altitude)}
        goal = {2: (3.0, -lateral, altitude), 3: (-3.0, lateral, altitude)}
        separated = lambda t: t.get(2, {}).get("position", [0])[0] < -2.0 and t.get(3, {}).get("position", [0])[0] > 2.0
        complete = lambda t: t.get(2, {}).get("position", [0])[0] > 2.0 and t.get(3, {}).get("position", [0])[0] < -2.0
    else:
        raise SystemExit("only head_on_crossing is supported in this PX4 smoke runner")

    telemetry = {}
    overrides = 0
    acks = []

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe("swarm/drone/+/telemetry")
        client.subscribe("swarm/commander/override")
        client.subscribe("px4/+/command_ack")

    def on_message(client, userdata, msg):
        nonlocal overrides
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
        if msg.topic.endswith("/telemetry"):
            drone_id = int(payload.get("drone") or 0)
            telemetry[drone_id] = payload
        elif msg.topic == "swarm/commander/override":
            overrides += 1
        elif msg.topic.endswith("/command_ack"):
            acks.append(json.loads(msg.payload.decode()))

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="group3-px4-runner")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    time.sleep(0.8)

    telemetry_deadline = time.time() + 30.0
    while time.time() < telemetry_deadline and not all(
        telemetry.get(drone_id, {}).get("position") for drone_id in (2, 3)
    ):
        time.sleep(0.2)
    if not all(telemetry.get(drone_id, {}).get("position") for drone_id in (2, 3)):
        print("telemetry not ready", flush=True)
        client.loop_stop()
        client.disconnect()
        raise SystemExit(2)

    def publish(drone_id: int, action: str, target=None):
        native_topic = action in {"arm", "takeoff", "land", "rtl", "hold"}
        payload = {
            "drone": drone_id,
            "action": action,
            "ttl_sec": 5.0,
            "command_id": f"group3-{action}-{drone_id}-{now_ms()}",
            "timestamp_ms": now_ms(),
            "source_frame": "PX4_NED" if native_topic else "FLU",
            "priority": "normal",
        }
        if action == "takeoff":
            payload["altitude_m"] = 1.0
        if action == "move_to" and target is not None:
            payload["target"] = list(target)
            payload["waypoint"] = list(target)
        topic = f"px4/{drone_id}/command" if native_topic else f"swarm/drone/{drone_id}/command"
        client.publish(topic, json.dumps(payload, separators=(",", ":")))

    separated_ok = False
    if args.phase == "prepare":
        for drone_id in (2, 3):
            publish(drone_id, "arm")
        deadline = time.time() + 15
        while time.time() < deadline and not all(telemetry.get(d, {}).get("armed") for d in (2, 3)):
            time.sleep(0.2)
        for drone_id in (2, 3):
            publish(drone_id, "takeoff")
        deadline = time.time() + 20
        while time.time() < deadline and not all(
            telemetry.get(d, {}).get("position", [0, 0, 0])[2] > 0.5 for d in (2, 3)
        ):
            time.sleep(0.2)
        print("prepare_after_takeoff", json.dumps({
            d: {
                "armed": telemetry.get(d, {}).get("armed"),
                "position": telemetry.get(d, {}).get("position"),
            }
            for d in (2, 3)
        }), flush=True)
        for drone_id, target in start.items():
            publish(drone_id, "move_to", target)
            deadline = time.time() + 20
            next_publish = 0.0
            while time.time() < deadline:
                now = time.time()
                if now >= next_publish:
                    publish(drone_id, "move_to", target)
                    next_publish = now + 1.0
                pos = telemetry.get(drone_id, {}).get("position")
                if pos and abs(pos[0] - target[0]) < 0.5:
                    break
                time.sleep(0.2)
        deadline = time.time() + 20
        next_publish = 0.0
        last_debug = 0.0
        while time.time() < deadline and not separated(telemetry):
            now = time.time()
            if now >= next_publish:
                for drone_id, target in start.items():
                    publish(drone_id, "move_to", target)
                next_publish = now + 1.0
            if now - last_debug >= 2.0:
                last_debug = now
                print("prepare_separating", json.dumps({
                    d: telemetry.get(d, {}).get("position") for d in (2, 3)
                }), flush=True)
            time.sleep(0.2)
        separated_ok = separated(telemetry)
        print(json.dumps({"phase": "prepare", "separated": separated_ok}, indent=2), flush=True)
        client.loop_stop()
        client.disconnect()
        return

    # cross phase: assume drones are already armed/hovering at the start line.
    if not all(telemetry.get(d, {}).get("armed") for d in (2, 3)):
        for drone_id in (2, 3):
            publish(drone_id, "arm")
            publish(drone_id, "takeoff")
        time.sleep(5)

    for drone_id, target in goal.items():
        publish(drone_id, "move_to", target)

    deadline = time.time() + args.timeout_s
    min_distance = None
    completed = False
    next_goal_publish = 0.0
    while time.time() < deadline:
        now = time.time()
        if now >= next_goal_publish:
            for drone_id, target in goal.items():
                publish(drone_id, "move_to", target)
            next_goal_publish = now + 2.0
        positions = {d: telemetry.get(d, {}).get("position_flu") for d in (2, 3)}
        if all(positions.get(d) for d in (2, 3)):
            distance = math.dist(positions[2], positions[3])
            min_distance = distance if min_distance is None else min(min_distance, distance)
        if complete(telemetry):
            completed = True
            break
        time.sleep(0.1)

    for drone_id in (2, 3):
        publish(drone_id, "land")

    print("last_acks", json.dumps([
        a for a in acks if str(a.get("command_id", "")).startswith("group3-")
    ][-8:], indent=2), flush=True)

    result = {
        "scenario": args.scenario,
        "seed": args.seed,
        "input_mode": "ego",
        "ego_source": "group2_synthetic_stereo",
        "condition": "C3",
        "separated": bool(separated_ok),
        "completed": completed,
        "min_distance_m": None if min_distance is None else round(min_distance, 4),
        "override_count": overrides,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
