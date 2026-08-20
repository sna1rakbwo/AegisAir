#!/usr/bin/env python3
"""D0--D3 diagnosis for stale-observation consistency.

This is a new protocol.  It never overwrites the frozen C1 v4 result.  Its
only question is whether 300 ms failure comes from a mismatch between the
nominal pilot's observation and RA's observation, before adding any new
safety mechanism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import (
    OBSERVATION_LEGACY_ASYMMETRIC,
    OBSERVATION_LOCAL_FRESH_SELF,
    OBSERVATION_SHARED_CURRENT,
    OBSERVATION_SHARED_STALE,
    _scenario,
    run_sim_episode,
)
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance


PROTOCOL_ID = "aegisair-stale-observation-diagnosis-v1"
SEEDS = tuple(range(601, 651))
MAX_STEPS = 200
SCENARIOS = {
    "straight_line": "head_on",
    "crossing": "crossing",
    "dense_multi_uav": "multi_uav",
}
DIAGNOSTICS: tuple[dict[str, Any], ...] = (
    {
        "id": "D0_current_shared",
        "observation_mode": OBSERVATION_SHARED_CURRENT,
        "fault": {},
    },
    {
        "id": "D1_legacy_asymmetric",
        "observation_mode": OBSERVATION_LEGACY_ASYMMETRIC,
        "fault": {"estimator_delay_ms": 300},
    },
    {
        "id": "D2_shared_stale",
        "observation_mode": OBSERVATION_SHARED_STALE,
        "fault": {"estimator_delay_ms": 300},
    },
    {
        "id": "D3_local_fresh_self",
        "observation_mode": OBSERVATION_LOCAL_FRESH_SELF,
        "fault": {"estimator_delay_ms": 300},
    },
)


def make_ra(speed_limit: float) -> RuntimeAssurance:
    """Use the C1 exact-ZOH controller; only the observation architecture varies."""
    return RuntimeAssurance(
        params=RuntimeAssuranceParams(),
        perception_sigma=0.0,
        v_max=speed_limit,
        a_max=2.0,
        kv=2.0,
        sampled_data=True,
        gamma=0.1,
        tau_px4=0.7,
        tau_px4_min=0.53,
        tau_px4_max=1.76,
        qp_max_iters=300,
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for scenario in SCENARIOS:
        for diagnostic in DIAGNOSTICS:
            group = [
                row for row in rows
                if row["scenario"] == scenario and row["diagnostic"] == diagnostic["id"]
            ]
            summary.append(
                {
                    "scenario": scenario,
                    "diagnostic": diagnostic["id"],
                    "episodes": len(group),
                    "collision_rate": float(np.mean([row["collision"] for row in group])),
                    "completion_rate": float(np.mean([row["completed"] for row in group])),
                    "mean_min_pairwise_distance_m": float(np.mean([
                        row["min_pairwise_distance_m"] for row in group
                        if row["min_pairwise_distance_m"] is not None
                    ])),
                    "mean_cbf_events": float(np.mean([row["cbf_events"] for row in group])),
                    "mean_control_effort": float(np.mean([
                        row["control_effort"] for row in group
                        if row["control_effort"] is not None
                    ])),
                }
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="D0-D3 stale observation diagnosis")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_name in SCENARIOS.items():
        spec = _scenario(scenario_name)
        for seed in SEEDS:
            for diagnostic in DIAGNOSTICS:
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_ra(spec["scenario"].speed_limit),
                    mode="CBF_ONLY",
                    llm_client=None,
                    llm_fallback=None,
                    max_steps=MAX_STEPS,
                    real_time=False,
                    fault=diagnostic["fault"],
                    observation_mode=diagnostic["observation_mode"],
                )
                rows.append(
                    {
                        "scenario": scenario_id,
                        "seed": seed,
                        "diagnostic": diagnostic["id"],
                        **run,
                    }
                )

    payload = {
        "protocol": PROTOCOL_ID,
        "config": {
            "seeds": list(SEEDS),
            "max_steps": MAX_STEPS,
            "scenarios": SCENARIOS,
            "diagnostics": list(DIAGNOSTICS),
        },
        "summary": summarize(rows),
        "episodes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
