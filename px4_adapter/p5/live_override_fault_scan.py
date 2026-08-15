#!/usr/bin/env python3
"""Live two-drone safety-override fault scan.

Runs inside the ROS 2 bridge container. It subscribes to adapter-published
``swarm/drone/{id}/telemetry``, runs the same SafeDrones Safety Gate logic, and
injects the resulting override commands into the adapter command topics with
fault parameters (loss/latency). This measures the adapter's local command gate
against a real two-PX4 safety scenario.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import paho.mqtt.client as mqtt

from px4_adapter.px4_codec import flu_to_px4_ned
from swarm.safety import DroneSnapshot, SafetyConfig, SafetyGate, build_safety_command


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    telemetry: dict[int, dict] = {}
    acks: dict[str, list[dict]] = defaultdict(list)
    gate = SafetyGate(SafetyConfig())

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe("swarm/drone/+/telemetry")
        client.subscribe("px4/+/command_ack")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return
        if msg.topic.endswith("/telemetry"):
            drone_id = int(payload.get("drone") or 0)
            if drone_id in config["instances"]:
                telemetry[drone_id] = payload
        elif msg.topic.endswith("/command_ack"):
            command_id = str(payload.get("command_id") or "")
            if command_id:
                acks[command_id].append(payload)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, 60)
    client.loop_start()
    time.sleep(0.8)

    rows = []
    for fault in config["faults"]:
        for seed in config["seeds"]:
            rng = random.Random(seed)
            loss_rate = float(fault.get("command_loss_rate", 0.0))
            latency_ms = int(fault.get("command_latency_ms", 0))
            acks.clear()
            generated = 0
            dropped = 0
            published = 0

            for tick in range(int(config["ticks"])):
                t_ms = now_ms()
                snapshots = []
                for drone_id in config["instances"]:
                    payload = telemetry.get(drone_id)
                    if payload is None:
                        continue
                    try:
                        snapshots.append(DroneSnapshot.from_telemetry(payload))
                    except (KeyError, TypeError, ValueError):
                        continue
                if len(snapshots) < 2:
                    time.sleep(0.25)
                    continue

                decisions = gate.evaluate(snapshots, now=t_ms / 1000.0)
                for decision in decisions:
                    if decision.mode != "override":
                        continue
                    command = build_safety_command(decision)
                    generated += 1
                    if rng.random() < loss_rate:
                        dropped += 1
                        continue

                    command["timestamp_ms"] = t_ms - latency_ms
                    if command.get("target") is not None:
                        command["target"] = list(flu_to_px4_ned(*command["target"]))
                    command["source_frame"] = "PX4_NED"
                    command_id = command.get("command_id") or f"{fault['id']}-{seed}-{tick}"
                    command["command_id"] = command_id
                    client.publish(
                        f"px4/{decision.drone}/command",
                        json.dumps(command, separators=(",", ":")),
                    )
                    published += 1
                time.sleep(0.25)

            time.sleep(2.0)
            accepted = 0
            rejected = 0
            reasons: dict[str, int] = {}
            for command_id, results in acks.items():
                if any(r.get("accepted") for r in results):
                    accepted += 1
                else:
                    rejected += 1
                    reason = next(
                        (r.get("reason") for r in results if not r.get("accepted")),
                        "no_ack",
                    )
                    reasons[reason] = reasons.get(reason, 0) + 1

            rows.append(
                {
                    "fault_id": fault["id"],
                    "seed": seed,
                    "generated": generated,
                    "dropped": dropped,
                    "published": published,
                    "accepted": accepted,
                    "rejected": rejected,
                    "reasons": reasons,
                }
            )

    client.loop_stop()
    client.disconnect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
