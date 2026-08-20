#!/usr/bin/env python3
"""Probe the local Qwen's compact decision on the ``reassign_3`` failure context.

This does not run the full sim; it only asks the model what it would choose so
we can see, before any full matrix, whether it picks the non-nearest drone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.recovery import MlxLmClient, RecoveryContext
from swarm.safety import DroneSnapshot


def make_context() -> RecoveryContext:
    positions = {
        0: (-5.0, 1.0, 0.0),
        1: (0.0, 0.0, 0.0),
        2: (0.0, 4.0, 0.0),
    }
    goals = {
        0: (1.0, 0.0, 0.0),
        1: (2.0, 0.0, 0.0),
        2: (5.0, -4.0, 0.0),
    }
    return RecoveryContext(
        event="MISSION_PLAN_INVALIDATED",
        agent_i=0,
        agent_j=0,
        current_margin=1.0,
        predicted_min_margin=None,
        margin_degradation=None,
        intervention_count=0,
        cause="MISSION_CHANGE",
        severity="MEDIUM",
        snapshots={
            i: DroneSnapshot(drone_id=i, position=positions[i], velocity=(0.0, 0.0, 0.0))
            for i in positions
        },
        current_goals=goals,
        base_goals=goals,
        priorities={0: "normal", 1: "normal", 2: "normal"},
        timestamp_ms=0,
        mission_change={"kind": "fail_drone", "drone": 0},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen reassign decision")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    client = MlxLmClient(model_id=args.qwen_model, max_tokens=48, load=True)
    context = make_context()
    for i in range(args.n):
        result = client.generate(context)
        print(
            f"probe={i + 1} latency={result.latency_s:.2f}s "
            f"raw={result.raw!r}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
