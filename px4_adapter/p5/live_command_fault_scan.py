#!/usr/bin/env python3
"""Live command-loss/latency fault scan for one real PX4 adapter.

Run inside the ROS 2 bridge container with paho-mqtt installed. The drone is
kept disarmed and only ``hold`` commands are used, so this measures the
adapter's local command gate without moving the vehicle.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def run_fault(
    client: mqtt.Client,
    config: dict,
    fault: dict,
    seed: int,
    ack_results: dict[str, set[tuple[bool, str]]],
    instance: int,
    scenario: str,
) -> dict:
    rng = random.Random(seed)
    loss_rate = float(fault.get("command_loss_rate", 0.0))
    latency_ms = int(fault.get("command_latency_ms", 0))
    batch_size = int(config["batch_size"])
    interval_ms = int(config["command_interval_ms"])
    ack_results.clear()

    generated = 0
    dropped = 0
    published = 0
    for index in range(batch_size):
        generated += 1
        if rng.random() < loss_rate:
            dropped += 1
            continue
        command_id = f"{fault['id']}-{seed}-{index}"
        payload = {
            "drone": instance,
            "action": "hold",
            "ttl_sec": 1.0,
            "command_id": command_id,
            "timestamp_ms": now_ms() - latency_ms,
            "source_frame": "PX4_NED",
            "priority": "fault_scan",
        }
        client.publish(
            f"px4/{instance}/command",
            json.dumps(payload, separators=(",", ":")),
        )
        published += 1
        time.sleep(interval_ms / 1000.0)

    time.sleep(2.0)
    accepted = 0
    rejected = 0
    reasons: dict[str, int] = {}
    for result_set in ack_results.values():
        was_accepted = any(accepted_flag for accepted_flag, _ in result_set)
        if was_accepted:
            accepted += 1
        else:
            rejected += 1
            reason = next(
                (reason for accepted_flag, reason in result_set if not accepted_flag),
                "no_ack",
            )
            reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "scenario": scenario,
        "instance": instance,
        "fault_id": fault["id"],
        "seed": seed,
        "generated": generated,
        "dropped": dropped,
        "published": published,
        "accepted": accepted,
        "rejected": rejected,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance", type=int, default=2)
    parser.add_argument("--scenario", default="S1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    ack_results: dict[str, set[tuple[bool, str]]] = {}

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe(f"px4/{args.instance}/command_ack")

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        command_id = str(payload.get("command_id") or "")
        if not command_id:
            return
        ack_results.setdefault(command_id, set()).add(
            (bool(payload.get("accepted")), str(payload.get("reason") or ""))
        )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, 60)
    client.loop_start()
    time.sleep(0.5)

    command_faults = [
        fault
        for fault in config["faults"]
        if fault.get("telemetry_stale_at_ms") is None
        and fault.get("gcs_lost_at_ms") is None
    ]
    rows = []
    for fault in command_faults:
        for seed in config["seeds"]:
            rows.append(
                run_fault(
                    client,
                    config,
                    fault,
                    seed,
                    ack_results,
                    args.instance,
                    args.scenario,
                )
            )

    client.loop_stop()
    client.disconnect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(rows), "faults": [r["fault_id"] for r in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
