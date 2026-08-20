#!/usr/bin/env python3
"""Frozen 4-UAV minimum C1 envelope ablation.

The four ablations share the exact same scenario/seed/fault stream.  They
change only which terms contribute to ``d_safe``; every condition keeps the
same sampled-data exact-ZOH RA and actuator limits.
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

from marllib.config import randomized
from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance


ABLATIONS = ("fixed_distance", "full_envelope", "no_perception_margin", "no_aoi_margin")

SCENARIOS: dict[str, dict[str, Any]] = {
    "randomized_start_goal": {"fault": {}},
    "dense_intersection": {"fault": {}},
    "perception_dropout": {
        "fault": {
            "perception_noise_pos_m": 0.2,
            "perception_noise_vel_mps": 0.1,
            "estimator_dropout_rate": 0.3,
        }
    },
    # Feed RA a truly stale estimate rather than current truth plus only an
    # AoI scalar.  ``run_sim_episode`` then dead-reckons it to the control
    # instant and derives AoI from the same timestamp.
    "telemetry_delay": {"fault": {"estimator_delay_ms": 300}},
}


def scenario_spec(name: str) -> dict[str, Any]:
    if name == "randomized_start_goal":
        return {
            "name": name,
            "scenario": randomized(4),
            "mission_change": None,
            "change_step": None,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }
    if name in ("dense_intersection", "perception_dropout", "telemetry_delay"):
        spec = dict(_scenario("multi_uav"))
        spec["name"] = name
        return spec
    raise ValueError(f"unknown C1 scenario: {name}")


def params_for(ablation: str) -> RuntimeAssuranceParams:
    if ablation == "fixed_distance":
        return RuntimeAssuranceParams(
            tau_r=0.0,
            a_eff=1e9,
            beta=0.0,
            v_max=0.0,
            a_max=0.0,
        )
    if ablation == "full_envelope":
        return RuntimeAssuranceParams()
    if ablation == "no_perception_margin":
        return RuntimeAssuranceParams(beta=0.0)
    if ablation == "no_aoi_margin":
        return RuntimeAssuranceParams(v_max=0.0, a_max=0.0)
    raise ValueError(f"unknown ablation: {ablation}")


def make_controller(ablation: str, *, qp_max_iters: int, speed_limit: float) -> RuntimeAssurance:
    return RuntimeAssurance(
        params=params_for(ablation),
        perception_sigma=0.0,
        v_max=speed_limit,
        a_max=2.0,
        kv=2.0,
        sampled_data=True,
        gamma=0.1,
        tau_px4=0.7,
        tau_px4_min=0.53,
        tau_px4_max=1.76,
        qp_max_iters=qp_max_iters,
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scenario in SCENARIOS:
        for ablation in ABLATIONS:
            group = [row for row in rows if row["scenario"] == scenario and row["ablation"] == ablation]
            if not group:
                continue
            output.append({
                "scenario": scenario,
                "ablation": ablation,
                "episodes": len(group),
                "collision_rate": float(np.mean([row["collision"] for row in group])),
                "boundary_violation_rate": float(np.mean([row["min_rho"] < 0.0 for row in group])),
                "completion_rate": float(np.mean([row["completed"] for row in group])),
                "min_pairwise_distance_m": float(np.min([
                    row["min_pairwise_distance_m"] for row in group
                    if row["min_pairwise_distance_m"] is not None
                ])),
                "mean_min_pairwise_distance_m": float(np.mean([
                    row["min_pairwise_distance_m"] for row in group
                    if row["min_pairwise_distance_m"] is not None
                ])),
                "min_rho": float(np.min([row["min_rho"] for row in group])),
                "mean_min_rho": float(np.mean([row["min_rho"] for row in group])),
                "mean_cbf_events": float(np.mean([row["cbf_events"] for row in group])),
                "mean_control_effort": float(np.mean([row["control_effort"] for row in group])),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimum C1 4-UAV paired ablation")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(201, 251)))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        parser.error("C1 minimum protocol requires exactly 50 unique paired seeds")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_config in SCENARIOS.items():
        spec = scenario_spec(scenario_name)
        for seed in seeds:
            for ablation in ABLATIONS:
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_controller(
                        ablation,
                        qp_max_iters=args.qp_max_iters,
                        speed_limit=spec["scenario"].speed_limit,
                    ),
                    mode="CBF_ONLY",
                    llm_client=None,
                    llm_fallback=None,
                    max_steps=args.max_steps,
                    real_time=False,
                    fault=scenario_config["fault"],
                )
                rows.append({"scenario": scenario_name, "seed": seed, "ablation": ablation, **run})

    payload = {
        "config": {
            "protocol": "aegisair-minimum-c1-v1",
            "seeds": seeds,
            "ablations": list(ABLATIONS),
            "scenarios": SCENARIOS,
            "max_steps": args.max_steps,
            "qp_max_iters": args.qp_max_iters,
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
