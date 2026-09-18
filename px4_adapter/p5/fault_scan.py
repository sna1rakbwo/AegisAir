#!/usr/bin/env python3
"""Deterministic local fault scan for the PX4 adapter safety boundary."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from px4_adapter.mqtt_codec import Px4Command
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(
    action: str,
    timestamp_ms: int,
    command_id: str,
    rng: random.Random,
    invalid_target_rate: float,
) -> Px4Command:
    target = None
    velocity = None
    altitude_m = None
    if action == "move_to":
        if rng.random() < invalid_target_rate:
            target = (0.0, 0.0, -20.0)
        else:
            target = (1.0, 0.0, -2.0)
            velocity = (1.0, 0.0, 0.0)
    elif action == "takeoff":
        altitude_m = 2.5

    return Px4Command(
        drone=2,
        action=action,
        target=target,
        velocity=velocity,
        altitude_m=altitude_m,
        yaw=0.0,
        ttl_sec=1.0,
        command_id=command_id,
        timestamp_ms=timestamp_ms,
        source_frame="PX4_NED",
        priority="normal",
    )


def command_bypasses_limits(command: Px4Command, limits: SafetyLimits) -> bool:
    if command.action != "move_to" or command.target is None:
        return False
    altitude = -command.target[2]
    horizontal = math.hypot(command.target[0], command.target[1])
    if not (limits.min_altitude_m <= altitude <= limits.max_altitude_m):
        return True
    if horizontal > limits.max_position_distance_m:
        return True
    if command.velocity is not None:
        if math.hypot(command.velocity[0], command.velocity[1]) > limits.max_horizontal_speed_mps:
            return True
        if abs(command.velocity[2]) > limits.max_vertical_speed_mps:
            return True
    return False


def run_episode(config: dict, fault: dict, seed: int) -> dict:
    rng = random.Random(seed)
    limits = SafetyLimits(**config["safety_limits"])
    safety = LocalSafetyState(limits)
    duration_ms = int(config["duration_ms"])
    command_interval_ms = int(config["command_interval_ms"])

    generated = 0
    dropped = 0
    evaluated = 0
    accepted = 0
    rejected = 0
    bypass = 0
    reasons: dict[str, int] = {}
    first_fail_closed_ms: int | None = None

    next_command_ms = command_interval_ms
    command_index = 0
    telemetry_timestamp_ms = 0
    connection_lost = False

    for now_ms in range(0, duration_ms + 1, 50):
        stale_at_ms = fault.get("telemetry_stale_at_ms")
        gcs_lost_at_ms = fault.get("gcs_lost_at_ms")
        stale_active = bool(stale_at_ms is not None and now_ms >= int(stale_at_ms))
        if not stale_active:
            telemetry_timestamp_ms = now_ms
        connection_lost = bool(
            gcs_lost_at_ms is not None and now_ms >= int(gcs_lost_at_ms)
        )

        if now_ms >= next_command_ms:
            action = config["command_mix"][command_index % len(config["command_mix"])]
            command_index += 1
            next_command_ms += command_interval_ms
            generated += 1

            if rng.random() < float(fault.get("command_loss_rate", 0.0)):
                dropped += 1
            else:
                command_timestamp_ms = now_ms - int(fault.get("command_latency_ms", 0))
                command = build_command(
                    action,
                    command_timestamp_ms,
                    f"{fault['id']}-{seed}-{generated}",
                    rng,
                    float(config["invalid_target_rate"]),
                )

                from px4_adapter.mqtt_codec import TelemetryState

                telemetry = TelemetryState(
                    instance_id=2,
                    source_frame="PX4_NED",
                    timestamp_ms=telemetry_timestamp_ms,
                    source_timestamp_us=telemetry_timestamp_ms * 1000,
                    armed=False,
                    nav_state=4,
                    failsafe=False,
                    connection_lost=connection_lost,
                    position=(0.0, 0.0, 0.0),
                    velocity=(0.0, 0.0, 0.0),
                    xy_valid=True,
                    z_valid=True,
                    v_xy_valid=True,
                    v_z_valid=True,
                )

                decision = safety.evaluate(now_ms, telemetry, command)
                evaluated += 1
                if decision.allowed:
                    accepted += 1
                    if command_bypasses_limits(command, limits):
                        bypass += 1
                else:
                    rejected += 1
                    reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
                    if first_fail_closed_ms is None and decision.reason in {
                        "telemetry_missing",
                        "telemetry_stale",
                        "gcs_connection_lost",
                    }:
                        first_fail_closed_ms = now_ms

    return {
        "protocol_id": config["protocol_id"],
        "fault_id": fault["id"],
        "seed": seed,
        "generated": generated,
        "dropped": dropped,
        "evaluated": evaluated,
        "accepted": accepted,
        "rejected": rejected,
        "bypass": bypass,
        "reasons": reasons,
        "first_fail_closed_ms": first_fail_closed_ms,
        "fault_onset_ms": (
            int(fault["telemetry_stale_at_ms"])
            if fault.get("telemetry_stale_at_ms") is not None
            else int(fault["gcs_lost_at_ms"])
            if fault.get("gcs_lost_at_ms") is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    rows = []
    for fault in config["faults"]:
        for seed in config["seeds"]:
            rows.append(run_episode(config, fault, seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    aggregate = {
        "protocol_id": config["protocol_id"],
        "episodes": len(rows),
        "bypass_total": sum(r["bypass"] for r in rows),
        "by_fault": {
            fault["id"]: {
                "episodes": sum(1 for r in rows if r["fault_id"] == fault["id"]),
                "accepted": sum(r["accepted"] for r in rows if r["fault_id"] == fault["id"]),
                "rejected": sum(r["rejected"] for r in rows if r["fault_id"] == fault["id"]),
                "bypass": sum(r["bypass"] for r in rows if r["fault_id"] == fault["id"]),
            }
            for fault in config["faults"]
        },
    }
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["bypass_total"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
