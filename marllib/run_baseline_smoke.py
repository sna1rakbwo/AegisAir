#!/usr/bin/env python3
"""Run the frozen S0/S1/S2/S3/S6 lightweight-simulation smoke matrix.

This runner is deliberately limited to smoke validation.  It uses identical
scenario seeds for every safety condition and writes one JSON artifact with
the unaggregated per-episode records.  It is not the final-statistics runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance


CONDITIONS = ("S0", "S1", "S2", "S3", "S6")


class ObserveOnlyRA:
    """Measure margins with RA but return the unfiltered nominal command."""

    def __init__(self, observer: RuntimeAssurance) -> None:
        self._observer = observer

    def filter(self, snapshots, nominal_actions, t, aoi=None):
        observed = self._observer.filter(snapshots, nominal_actions, t, aoi)
        return {
            drone: replace(
                result,
                mode="nominal",
                safe_action=(
                    float(nominal_actions[drone][0]),
                    float(nominal_actions[drone][1]),
                ),
                intervened=False,
            )
            for drone, result in observed.items()
        }


def make_controller(condition: str, *, qp_max_iters: int):
    """Create one independent controller for an episode and condition."""
    common: dict[str, Any] = {
        "params": RuntimeAssuranceParams(tau_ctrl=0.0),
        "v_max": 1.5,
        "a_max": 2.0,
        "kv": 2.0,
        "qp_max_iters": qp_max_iters,
    }
    if condition == "S0":
        return ObserveOnlyRA(RuntimeAssurance(**common))
    if condition == "S1":
        return RuntimeAssurance(**common)
    if condition == "S2":
        return RuntimeAssurance(**common, use_hocbf=True, hocbf_k1=1.0, hocbf_k2=1.0)
    if condition == "S3":
        return RuntimeAssurance(**common, sampled_data=True, gamma=0.1, tau_px4=0.0)
    if condition == "S6":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=0.7,
            tau_px4_min=0.53,
            tau_px4_max=1.76,
        )
    raise ValueError(f"unknown condition: {condition}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for condition in CONDITIONS:
            group = [r for r in rows if r["scenario"] == scenario and r["condition"] == condition]
            if not group:
                continue
            summary.append({
                "scenario": scenario,
                "condition": condition,
                "episodes": len(group),
                "collision_rate": float(np.mean([r["collision"] for r in group])),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
                "min_rho": float(np.min([r["min_rho"] for r in group])),
                "mean_min_rho": float(np.mean([r["min_rho"] for r in group])),
                "mean_cbf_events": float(np.mean([r["cbf_events"] for r in group])),
                "adapter_rejections": int(sum(r["rejected_commands"] for r in group)),
            })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired S0/S1/S2/S3/S6 smoke matrix")
    parser.add_argument("--scenarios", default="head_on,multi_uav")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--telemetry-stale-ms", type=int, default=0)
    parser.add_argument("--estimator-delay-ms", type=int, default=0)
    parser.add_argument("--estimator-dropout-rate", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.qp_max_iters < 1:
        parser.error("--qp-max-iters must be positive")
    if not 0.0 <= args.estimator_dropout_rate <= 1.0:
        parser.error("--estimator-dropout-rate must be in [0, 1]")
    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    if not scenario_names or not seeds:
        parser.error("at least one scenario and one seed are required")

    rows: list[dict[str, Any]] = []
    fault = {
        "telemetry_stale_ms": args.telemetry_stale_ms,
        "estimator_delay_ms": args.estimator_delay_ms,
        "estimator_dropout_rate": args.estimator_dropout_rate,
    }
    for scenario_name in scenario_names:
        spec = _scenario(scenario_name)
        for seed in seeds:
            for condition in CONDITIONS:
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_controller(condition, qp_max_iters=args.qp_max_iters),
                    mode="CBF_ONLY",
                    llm_client=None,
                    llm_fallback=None,
                    max_steps=args.max_steps,
                    real_time=False,
                    fault=fault,
                )
                rows.append({
                    "scenario": scenario_name,
                    "condition": condition,
                    "seed": seed,
                    **run,
                })

    config = {
        "protocol": "aegisair-baseline-smoke-v1",
        "scenarios": scenario_names,
        "seeds": seeds,
        "conditions": list(CONDITIONS),
        "max_steps": args.max_steps,
        "qp_max_iters": args.qp_max_iters,
        "fault": fault,
    }
    output = {"config": config, "summary": summarize(rows), "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
