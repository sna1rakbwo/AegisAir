#!/usr/bin/env python3
"""Run deterministic offline checks for dynamic recoverability admission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.recovery.recoverability_admission import (
    RecoverabilityAdmissionConfig,
    RecoverabilityAdmissionCoordinator,
)
from swarm.safety import DroneSnapshot


def _decision(
    config: RecoverabilityAdmissionConfig,
    *,
    failed_position: tuple[float, float, float],
    failed_velocity: tuple[float, float, float],
    healthy_position: tuple[float, float, float],
    healthy_velocity: tuple[float, float, float],
    orphan_goal: tuple[float, float, float],
) -> tuple[dict[str, object], float]:
    coordinator = RecoverabilityAdmissionCoordinator(config)
    started = time.perf_counter()
    coordinator.step(
        step=30,
        snapshots={
            2: DroneSnapshot(2, failed_position, velocity=failed_velocity),
            3: DroneSnapshot(3, healthy_position, velocity=healthy_velocity),
        },
        base_goals={2: orphan_goal, 3: (4.0, -2.0, 2.5)},
        mission_change={"kind": "fail_drone", "drone": 2},
    )
    elapsed_ms = 1_000.0 * (time.perf_counter() - started)
    return coordinator.summary(), elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("timing repeats must be positive")

    config = RecoverabilityAdmissionConfig()
    admitted, _ = _decision(
        config,
        failed_position=(-1.5, 2.0, 2.5),
        failed_velocity=(1.2, 0.0, 0.0),
        healthy_position=(-4.0, -2.0, 2.5),
        healthy_velocity=(1.0, 0.0, 0.0),
        orphan_goal=(4.0, 2.0, 2.5),
    )
    if admitted["state"] != "admitted":
        raise RuntimeError("representative dynamically feasible route was rejected")

    occupied, _ = _decision(
        config,
        failed_position=(2.0, 0.0, 2.5),
        failed_velocity=(0.0, 0.0, 0.0),
        healthy_position=(-4.0, -2.0, 2.5),
        healthy_velocity=(1.0, 0.0, 0.0),
        orphan_goal=(2.0, 0.0, 2.5),
    )
    if occupied["state"] != "rejected_hold":
        raise RuntimeError("occupied orphan goal was admitted")

    velocity_config = RecoverabilityAdmissionConfig(
        clearance_m=0.5,
        tracking_error_buffer_m=0.0,
        ring_extra_m=(0.5,),
        ring_samples=8,
        max_route_length_m=4.0,
        rollout_horizon_s=2.0,
    )
    toward, _ = _decision(
        velocity_config,
        failed_position=(4.0, 4.0, 2.5),
        failed_velocity=(0.0, 0.0, 0.0),
        healthy_position=(-1.5, 0.0, 2.5),
        healthy_velocity=(1.0, 0.0, 0.0),
        orphan_goal=(0.0, 0.0, 2.5),
    )
    away, _ = _decision(
        velocity_config,
        failed_position=(4.0, 4.0, 2.5),
        failed_velocity=(0.0, 0.0, 0.0),
        healthy_position=(-1.5, 0.0, 2.5),
        healthy_velocity=(-1.0, 0.0, 0.0),
        orphan_goal=(0.0, 0.0, 2.5),
    )
    if toward["state"] != "admitted" or away["state"] != "rejected_hold":
        raise RuntimeError("healthy velocity did not affect dynamic admission")

    latencies = []
    for _ in range(args.timing_repeats):
        _, elapsed_ms = _decision(
            config,
            failed_position=(-1.5, 2.0, 2.5),
            failed_velocity=(1.2, 0.0, 0.0),
            healthy_position=(-4.0, -2.0, 2.5),
            healthy_velocity=(1.0, 0.0, 0.0),
            orphan_goal=(4.0, 2.0, 2.5),
        )
        latencies.append(elapsed_ms)
    latency_p95 = float(np.percentile(latencies, 95))
    if latency_p95 >= 50.0:
        raise RuntimeError(f"dynamic admission P95 exceeded 50 ms: {latency_p95}")

    report = {
        "model": "time_aligned_dynamic_rollout_v2",
        "representative_admit": admitted,
        "occupied_goal_reject": occupied,
        "velocity_sensitivity": {
            "toward_state": toward["state"],
            "away_state": away["state"],
        },
        "timing": {
            "repeats": args.timing_repeats,
            "median_ms": float(np.median(latencies)),
            "p95_ms": latency_p95,
            "max_ms": float(np.max(latencies)),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
