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


def make_controller(
    condition: str,
    *,
    qp_max_iters: int,
    v_max: float = 1.5,
    tau_px4: float = 0.7,
    tau_ctrl: float = 0.0,
    robust_tau_interval: bool = True,
    a_max: float = 2.0,
    kv: float = 2.0,
):
    """Create one independent controller for an episode and condition."""
    common: dict[str, Any] = {
        "params": RuntimeAssuranceParams(tau_ctrl=tau_ctrl),
        "v_max": v_max,
        "a_max": a_max,
        "kv": kv,
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
    if condition == "S4":
        return RuntimeAssurance(**common, sampled_data=True, gamma=0.1, tau_px4=0.0)
    if condition == "S5":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=tau_px4,
            tau_px4_min=tau_px4,
            tau_px4_max=tau_px4,
            execution_model="legacy_trapezoidal",
        )
    if condition == "E0":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=0.0,
        )
    if condition == "E1":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=tau_px4,
            tau_px4_min=tau_px4,
            tau_px4_max=tau_px4,
            execution_model="legacy_trapezoidal",
        )
    if condition == "E2":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=tau_px4,
            tau_px4_min=tau_px4,
            tau_px4_max=tau_px4,
        )
    if condition == "S6":
        return RuntimeAssurance(
            **common,
            sampled_data=True,
            gamma=0.1,
            tau_px4=tau_px4,
            tau_px4_min=0.53 if robust_tau_interval else tau_px4,
            tau_px4_max=1.76 if robust_tau_interval else tau_px4,
        )
    raise ValueError(f"unknown condition: {condition}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows}):
            group = [r for r in rows if r["scenario"] == scenario and r["condition"] == condition]
            if not group:
                continue
            audits = [
                record
                for row in group
                for record in (row.get("intersample_records") or [])
            ]
            latencies = [
                latency
                for row in group
                for latency in row.get("qp_latency_ms", [])
            ]
            summary.append({
                "scenario": scenario,
                "condition": condition,
                "episodes": len(group),
                "collision_rate": float(np.mean([r["collision"] for r in group])),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
                "min_rho": float(np.min([r["min_rho"] for r in group])),
                "mean_min_rho": float(np.mean([r["min_rho"] for r in group])),
                "mean_cbf_events": float(np.mean([r["cbf_events"] for r in group])),
                "mean_qp_infeasible_steps": float(
                    np.mean([r.get("qp_infeasible_steps", 0) for r in group])
                ),
                "intersample_min_rho": (
                    float(min(record["intersample_min_rho"] for record in audits))
                    if audits else None
                ),
                "intersample_min_distance_m": (
                    float(min(record["intersample_min_distance_m"] for record in audits))
                    if audits else None
                ),
                "qp_latency_ms": (
                    {
                        "p50": float(np.percentile(latencies, 50)),
                        "p95": float(np.percentile(latencies, 95)),
                        "p99": float(np.percentile(latencies, 99)),
                        "max": float(np.max(latencies)),
                    }
                    if latencies else None
                ),
                "adapter_rejections": int(sum(r["rejected_commands"] for r in group)),
            })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired S0/S1/S2/S3/S6 smoke matrix")
    parser.add_argument("--scenarios", default="head_on,multi_uav")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--telemetry-stale-ms", type=int, default=0)
    parser.add_argument("--estimator-delay-ms", type=int, default=0)
    parser.add_argument("--estimator-dropout-rate", type=float, default=0.0)
    parser.add_argument(
        "--execution-tau-s",
        type=float,
        default=0.0,
        help="opt-in exact-ZOH plant lag for C2 execution ablations",
    )
    parser.add_argument("--controller-tau-s", type=float, default=0.7)
    parser.add_argument(
        "--tau-ctrl-s",
        type=float,
        default=0.0,
        help="frozen controller delay used by the selected experiment",
    )
    parser.add_argument(
        "--execution-audit-samples",
        type=int,
        default=0,
        help="exact-ZOH intersample points per control interval (C2: at least 100)",
    )
    parser.add_argument("--a-max", type=float, default=2.0)
    parser.add_argument("--kv", type=float, default=2.0)
    parser.add_argument("--sequential-pass", action="store_true")
    parser.add_argument(
        "--nominal-zoh",
        action="store_true",
        help="set S6 tau interval to the identified nominal tau for C2 E2",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.qp_max_iters < 1:
        parser.error("--qp-max-iters must be positive")
    if not 0.0 <= args.estimator_dropout_rate <= 1.0:
        parser.error("--estimator-dropout-rate must be in [0, 1]")
    if args.execution_tau_s < 0.0:
        parser.error("--execution-tau-s must be non-negative")
    if args.controller_tau_s < 0.0:
        parser.error("--controller-tau-s must be non-negative")
    if args.tau_ctrl_s < 0.0:
        parser.error("--tau-ctrl-s must be non-negative")
    if args.execution_audit_samples < 0:
        parser.error("--execution-audit-samples must be non-negative")
    if args.execution_audit_samples and args.execution_tau_s <= 0.0:
        parser.error("--execution-audit-samples requires --execution-tau-s > 0")
    if args.a_max <= 0.0 or args.kv <= 0.0:
        parser.error("--a-max and --kv must be positive")
    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    if not scenario_names or not seeds or not conditions:
        parser.error("at least one scenario, seed, and condition are required")
    if any(condition not in {*CONDITIONS, "S4", "S5", "E0", "E1", "E2"} for condition in conditions):
        parser.error("unknown condition")

    rows: list[dict[str, Any]] = []
    fault = {
        "telemetry_stale_ms": args.telemetry_stale_ms,
        "estimator_delay_ms": args.estimator_delay_ms,
        "estimator_dropout_rate": args.estimator_dropout_rate,
    }
    for scenario_name in scenario_names:
        spec = _scenario(scenario_name)
        for seed in seeds:
            for condition in conditions:
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_controller(
                        condition,
                        qp_max_iters=args.qp_max_iters,
                        v_max=spec["scenario"].speed_limit,
                        tau_px4=args.controller_tau_s,
                        tau_ctrl=args.tau_ctrl_s,
                        robust_tau_interval=not args.nominal_zoh,
                        a_max=args.a_max,
                        kv=args.kv,
                    ),
                    mode="CBF_ONLY",
                    llm_client=None,
                    llm_fallback=None,
                    max_steps=args.max_steps,
                    real_time=False,
                    fault=fault,
                    execution_tau_s=args.execution_tau_s,
                    execution_audit_samples=args.execution_audit_samples,
                    sequential_pass=args.sequential_pass,
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
        "conditions": conditions,
        "max_steps": args.max_steps,
        "qp_max_iters": args.qp_max_iters,
        "fault": fault,
        "execution_tau_s": args.execution_tau_s,
        "controller_tau_s": args.controller_tau_s,
        "tau_ctrl_s": args.tau_ctrl_s,
        "execution_audit_samples": args.execution_audit_samples,
        "nominal_zoh": args.nominal_zoh,
        "sequential_pass": args.sequential_pass,
        "a_max": args.a_max,
        "kv": args.kv,
    }
    output = {"config": config, "summary": summarize(rows), "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
