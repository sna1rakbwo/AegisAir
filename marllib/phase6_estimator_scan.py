#!/usr/bin/env python3
"""Deterministic Phase 6 SharedStateEstimator gate (no PX4/Gazebo).

Validates the first architecture decision in
``docs/decisions/phase6_shared_state_estimator.md``: raw shared telemetry ->
``SharedStateEstimator`` -> covariance-bearing ``EstimatedState`` -> RA.  Delay,
dropout, and covariance are estimator configuration, not scattered code changes.

Claim boundary: this validates the estimator -> RA decision path in the
lightweight point-mass environment.  Adapter fail-closed behavior is frozen
separately by ``px4_adapter/p5``; live PX4 remains a later gate.
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
from marllib.phase5_runner import (
    CRUISE_ALTITUDE_M,
    _go_to_goal,
    _scenario,
    _snapshots,
)
from swarm.estimation import SharedStateEstimator, SharedStateEstimatorConfig
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import RecoveryOverrides


PROTOCOL_ID = "safedrones-aegisair-phase6-estimator-local-v1"
SCENARIO = "head_on"
SEEDS = list(range(1, 11))
MAX_STEPS = 300
DT = 0.05


CONFIGS: list[tuple[str, SharedStateEstimatorConfig]] = [
    ("NONE", SharedStateEstimatorConfig()),
    ("DELAY_300MS", SharedStateEstimatorConfig(delay_ms=300)),
    ("DELAY_2000MS", SharedStateEstimatorConfig(delay_ms=2000)),
    ("DROPOUT_30", SharedStateEstimatorConfig(dropout_rate=0.3)),
    ("COV_HIGH", SharedStateEstimatorConfig(measurement_cov_m2=0.25)),
]


def _make_ra() -> RuntimeAssurance:
    return RuntimeAssurance(
        params=RuntimeAssuranceParams(tau_ctrl=0.0),
        v_max=1.5,
        sampled_data=True,
        gamma=0.1,
    )


def _run_episode(
    *,
    spec: dict[str, Any],
    config_id: str,
    seed: int,
    config: SharedStateEstimatorConfig,
) -> dict[str, Any]:
    env = MultiUAVEnv(spec["scenario"])
    env.reset(seed=seed)
    ra = _make_ra()
    estimator = SharedStateEstimator(config)
    rng = np.random.default_rng(seed)

    base_goals = {
        i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
        for i in env.agent_ids
    }
    overrides = RecoveryOverrides()
    failed: set[int] = set()
    aborted: set[int] = set()

    collision = False
    completed = False
    completion_step = MAX_STEPS
    cbf_events = 0
    min_rho = float("inf")
    age_ms_sum = 0.0
    age_ms_n = 0
    max_age_ms = 0.0
    cov_sum = 0.0
    cov_n = 0

    for step in range(MAX_STEPS):
        t = step * env.scenario.dt
        now_ms = int(t * 1000.0)
        nominal = _go_to_goal(
            env,
            base_goals=base_goals,
            overrides=overrides,
            failed=failed,
            aborted=aborted,
        )
        raw = _snapshots(env)
        estimated = estimator.step(raw, now_ms, rng=rng)

        aoi: dict[tuple[int, int], float] = {}
        for i in env.agent_ids:
            age_i = max(0.0, (now_ms - estimated[i].timestamp_ms) / 1000.0)
            age_ms_sum += age_i * 1000.0
            age_ms_n += 1
            max_age_ms = max(max_age_ms, age_i * 1000.0)
            cov_sum += float(estimated[i].covariance[0])
            cov_n += 1
            for j in env.agent_ids:
                if i != j:
                    age_j = max(0.0, (now_ms - estimated[j].timestamp_ms) / 1000.0)
                    aoi[(i, j)] = max(age_i, age_j)

        results = ra.filter(estimated, nominal, t=t, aoi=aoi)
        cbf_events += sum(1 for r in results.values() if r.intervened)
        min_rho = min(min_rho, min(r.safety_margin for r in results.values()))

        final = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
        _, _, _, _, infos = env.step(final)

        if any(i["collided"] for i in infos.values()):
            collision = True
            completion_step = step + 1
            break

        reached_base = {
            i
            for i in env.agent_ids
            if np.linalg.norm(env.positions[i] - np.asarray(base_goals[i][:2]))
            < env.scenario.goal_epsilon
        }
        if all(i in aborted or i in reached_base for i in env.agent_ids):
            completed = True
            completion_step = step + 1
            break

    return {
        "protocol_id": PROTOCOL_ID,
        "config_id": config_id,
        "seed": seed,
        "collision": collision,
        "completed": completed,
        "completion_steps": completion_step,
        "min_rho": round(min_rho, 6) if min_rho != float("inf") else None,
        "cbf_events": cbf_events,
        "mean_age_ms": round(age_ms_sum / age_ms_n, 3) if age_ms_n else 0.0,
        "max_age_ms": round(max_age_ms, 3),
        "mean_cov_m2": round(cov_sum / cov_n, 6) if cov_n else 0.0,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by(name: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["config_id"] == name]

    configs: dict[str, dict[str, Any]] = {}
    for name, _ in CONFIGS:
        sub = by(name)
        rho = [r["min_rho"] for r in sub if r["min_rho"] is not None]
        configs[name] = {
            "episodes": len(sub),
            "collisions": sum(1 for r in sub if r["collision"]),
            "completed": sum(1 for r in sub if r["completed"]),
            "min_rho": (
                {"min": round(min(rho), 6), "max": round(max(rho), 6)}
                if rho
                else None
            ),
            "mean_age_ms": round(
                sum(r["mean_age_ms"] for r in sub) / len(sub), 3
            ),
            "max_age_ms": max(r["max_age_ms"] for r in sub),
            "mean_cov_m2": round(
                sum(r["mean_cov_m2"] for r in sub) / len(sub), 6
            ),
        }

    none = configs["NONE"]
    delay = configs["DELAY_300MS"]
    drop = configs["DROPOUT_30"]
    cov_high = configs["COV_HIGH"]

    checks = {
        "no_collision": all(r["collision"] is False for r in rows),
        "baseline_min_rho_ge_0": (
            none["min_rho"] is not None and none["min_rho"]["min"] >= 0.0
        ),
        "delay_increases_age": delay["mean_age_ms"] > none["mean_age_ms"],
        "dropout_increases_covariance": drop["mean_cov_m2"]
        > none["mean_cov_m2"],
        "high_covariance_reduces_min_rho": (
            cov_high["min_rho"] is not None
            and none["min_rho"] is not None
            and cov_high["min_rho"]["min"] < none["min_rho"]["min"]
        ),
    }

    return {
        "protocol_id": PROTOCOL_ID,
        "episodes": len(rows),
        "checks": checks,
        "pass": all(checks.values()),
        "configs": configs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    spec = _scenario(SCENARIO)
    spec["scenario"] = replace(spec["scenario"], dt=DT)

    rows: list[dict[str, Any]] = []
    for name, config in CONFIGS:
        for seed in SEEDS:
            rows.append(
                _run_episode(spec=spec, config_id=name, seed=seed, config=config)
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )
    summary = _aggregate(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
