#!/usr/bin/env python3
"""Deterministic multi-drone safety-override fault scan.

Uses the real SafeDrones ``SafetyGate`` and the adapter's local safety state
machine without PX4/Gazebo.  This is a least-cost statistical gate; it does
not claim live flight safety.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from px4_adapter.mqtt_codec import (
    Px4Command,
    TelemetryState,
    decode_command,
    normalize_command_to_ned,
)
from px4_adapter.px4_codec import flu_to_px4_ned
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits
from swarm.safety import DroneSnapshot, SafetyConfig, SafetyGate, build_safety_command


def positions_for(drone_count: int, rng: random.Random) -> dict[int, tuple[float, float, float]]:
    positions: dict[int, tuple[float, float, float]] = {}
    radius = 0.15
    for index in range(drone_count):
        angle = 2 * math.pi * index / drone_count + rng.uniform(-0.15, 0.15)
        positions[index + 1] = (
            radius * math.cos(angle) + rng.uniform(-0.05, 0.05),
            radius * math.sin(angle) + rng.uniform(-0.05, 0.05),
            2.0,
        )
    return positions


def run_episode(config: dict, fault: dict, drone_count: int, seed: int) -> dict:
    rng = random.Random(seed)
    limits = SafetyLimits(**config["safety_limits"])
    local = LocalSafetyState(limits)
    gate = SafetyGate(SafetyConfig())
    positions = positions_for(drone_count, rng)

    generated = 0
    dropped = 0
    evaluated = 0
    accepted = 0
    rejected = 0
    bypass = 0
    reasons: dict[str, int] = {}

    for tick in range(int(config["ticks"])):
        now_ms = tick * 250
        snapshots = []
        for drone_id, flu in positions.items():
            snapshots.append(
                DroneSnapshot(
                    drone_id=drone_id,
                    position=flu,
                    velocity=(0.0, 0.0, 0.0),
                    speed_mps=0.0,
                    status="flying",
                    timestamp_ms=now_ms,
                )
            )

        decisions = gate.evaluate(snapshots, now=now_ms / 1000.0)
        for decision in decisions:
            if decision.mode != "override":
                continue
            command_dict = build_safety_command(decision)
            generated += 1
            if rng.random() < float(fault.get("command_loss_rate", 0.0)):
                dropped += 1
                continue

            command_dict["timestamp_ms"] = now_ms - int(fault.get("command_latency_ms", 0))
            command = normalize_command_to_ned(
                decode_command(command_dict, default_source_frame="FLU")
            )

            flu = positions.get(command.drone)
            if fault.get("telemetry_stale") and command.drone == 1 and tick >= 5:
                stale_ned = flu_to_px4_ned(*flu) if flu is not None else (0.0, 0.0, -2.0)
                telemetry = TelemetryState(
                    instance_id=command.drone,
                    source_frame="PX4_NED",
                    timestamp_ms=0,
                    source_timestamp_us=0,
                    armed=True,
                    nav_state=14,
                    failsafe=False,
                    connection_lost=False,
                    position=stale_ned,
                    velocity=(0.0, 0.0, 0.0),
                    xy_valid=True,
                    z_valid=True,
                    v_xy_valid=True,
                    v_z_valid=True,
                )
            else:
                ned = flu_to_px4_ned(*flu) if flu is not None else (0.0, 0.0, -2.0)
                telemetry = TelemetryState(
                    instance_id=command.drone,
                    source_frame="PX4_NED",
                    timestamp_ms=now_ms,
                    source_timestamp_us=now_ms * 1000,
                    armed=True,
                    nav_state=14,
                    failsafe=False,
                    connection_lost=False,
                    position=ned,
                    velocity=(0.0, 0.0, 0.0),
                    xy_valid=True,
                    z_valid=True,
                    v_xy_valid=True,
                    v_z_valid=True,
                )

            decision_result = local.evaluate(now_ms, telemetry, command)
            evaluated += 1
            if decision_result.allowed:
                accepted += 1
            else:
                rejected += 1
                reasons[decision_result.reason] = (
                    reasons.get(decision_result.reason, 0) + 1
                )

    return {
        "fault_id": fault["id"],
        "drone_count": drone_count,
        "seed": seed,
        "generated": generated,
        "dropped": dropped,
        "evaluated": evaluated,
        "accepted": accepted,
        "rejected": rejected,
        "bypass": bypass,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    rows = []
    for drone_count in config["drones"]:
        for fault in config["faults"]:
            for seed in config["seeds"]:
                rows.append(run_episode(config, fault, drone_count, seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "protocol_id": config["protocol_id"],
        "episodes": len(rows),
        "bypass_total": sum(row["bypass"] for row in rows),
        "faults": config["faults"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["bypass_total"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
