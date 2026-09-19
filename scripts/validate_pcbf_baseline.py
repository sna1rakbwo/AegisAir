#!/usr/bin/env python3
"""Run deterministic offline checks for the nonlinear PCBF baseline."""

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

from swarm.ra.pcbf import PCBFConfig, _predict, solve_pcbf


def _solve_timed(**kwargs):
    started = time.perf_counter()
    result = solve_pcbf(**kwargs)
    return result, 1_000.0 * (time.perf_counter() - started)


def _safe_hover_case(config: PCBFConfig) -> dict:
    nominal = {0: np.array([0.25, 0.0]), 1: np.array([-0.25, 0.0])}
    result, latency_ms = _solve_timed(
        nominal_accelerations=nominal,
        positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
        velocities={0: np.zeros(2), 1: np.zeros(2)},
        safe_distances={(0, 1): 2.0},
        config=config,
    )
    if not result.feasible or result.value > config.acceptable_tolerance:
        raise RuntimeError("safe-hover PCBF check failed")
    if not result.tie_break_applied:
        raise RuntimeError("safe-hover nominal tie-break was not applied")
    for drone in nominal:
        if not np.allclose(result.accelerations[drone], nominal[drone], atol=1e-5):
            raise RuntimeError("safe-hover nominal command was not preserved")
    return {
        "value": result.value,
        "slack_sum": result.slack_sum,
        "tracking_cost": result.tracking_cost,
        "cold_start_latency_ms": latency_ms,
        "stage1_status": result.stage1_status,
        "stage2_status": result.stage2_status,
    }


def _closed_loop_recovery(
    config: PCBFConfig, steps: int, *, use_warm_start: bool
) -> dict:
    positions = {0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])}
    velocities = {0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])}
    nominal = {0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])}
    values: list[float] = []
    latencies: list[float] = []
    residuals: list[float] = []
    warm_latencies: list[float] = []
    previous_plan: np.ndarray | None = None
    warm_start_steps = 0

    for _ in range(steps):
        result, latency_ms = _solve_timed(
            nominal_accelerations=nominal,
            positions=positions,
            velocities=velocities,
            safe_distances={(0, 1): 2.0},
            config=config,
            warm_start_plan=previous_plan if use_warm_start else None,
        )
        if not result.feasible or result.plan is None:
            raise RuntimeError(
                f"closed-loop recovery failed: {result.fail_closed_reason}"
            )
        values.append(result.value)
        latencies.append(latency_ms)
        residuals.append(result.max_constraint_violation)
        warm_start_steps += int(result.warm_start_used)
        if result.warm_start_used:
            warm_latencies.append(latency_ms)
        previous_plan = result.plan.copy()
        predicted_p, predicted_v = _predict(
            result.plan,
            positions=positions,
            velocities=velocities,
            drone_ids=[0, 1],
            config=config,
        )
        positions = {drone: predicted_p[drone][1].copy() for drone in positions}
        velocities = {drone: predicted_v[drone][1].copy() for drone in velocities}

    increases = np.diff(np.asarray(values, dtype=np.float64))
    max_increase = float(max(0.0, np.max(increases, initial=0.0)))
    monotonic_tolerance = 20.0 * config.acceptable_tolerance
    if max_increase > monotonic_tolerance:
        raise RuntimeError(f"PCBF value increased by {max_increase}")
    if values[-1] > config.acceptable_tolerance:
        raise RuntimeError("closed-loop recovery did not reach the zero-level set")
    report = {
        "steps": steps,
        "mode": "warm_start" if use_warm_start else "cold_multistart",
        "warm_start_steps": warm_start_steps,
        "values": values,
        "nonincreasing_within_tolerance": True,
        "max_value_increase": max_increase,
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_max_ms": float(np.max(latencies)),
        "max_constraint_violation": float(np.max(residuals)),
    }
    if warm_latencies:
        report["warm_only_latency_median_ms"] = float(np.median(warm_latencies))
        report["warm_only_latency_p95_ms"] = float(
            np.percentile(warm_latencies, 95)
        )
        report["warm_only_latency_max_ms"] = float(np.max(warm_latencies))
    return report


def _random_stress(config: PCBFConfig, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    latencies: list[float] = []
    residuals: list[float] = []
    maximum_input = 0.0
    successful = 0

    for _ in range(samples):
        angle = float(rng.uniform(-np.pi, np.pi))
        direction = np.array([np.cos(angle), np.sin(angle)])
        center = rng.uniform(-2.0, 2.0, size=2)
        distance = float(rng.uniform(1.4, 5.0))
        positions = {
            0: center - 0.5 * distance * direction,
            1: center + 0.5 * distance * direction,
        }
        velocities = {
            0: rng.uniform(-0.5, 0.5, size=2),
            1: rng.uniform(-0.5, 0.5, size=2),
        }
        nominal = {
            0: rng.uniform(-config.a_max, config.a_max, size=2),
            1: rng.uniform(-config.a_max, config.a_max, size=2),
        }
        result, latency_ms = _solve_timed(
            nominal_accelerations=nominal,
            positions=positions,
            velocities=velocities,
            safe_distances={(0, 1): 2.0},
            config=config,
        )
        latencies.append(latency_ms)
        residuals.append(result.max_constraint_violation)
        if not result.feasible or result.plan is None:
            continue
        successful += 1
        maximum_input = max(maximum_input, float(np.max(np.abs(result.plan))))
        predicted_p, predicted_v = _predict(
            result.plan,
            positions=positions,
            velocities=velocities,
            drone_ids=[0, 1],
            config=config,
        )
        for drone in positions:
            if np.max(np.abs(predicted_v[drone][-1])) > 1e-4:
                raise RuntimeError("stress plan violated the terminal stop set")
        terminal_distance = float(
            np.linalg.norm(predicted_p[0][-1] - predicted_p[1][-1])
        )
        if terminal_distance < 2.0 + config.terminal_buffer_m - 1e-4:
            raise RuntimeError("stress plan violated terminal separation")

    if successful != samples:
        raise RuntimeError(f"random stress solve failures: {samples - successful}")
    return {
        "samples": samples,
        "seed": seed,
        "successful": successful,
        "failed_closed": samples - successful,
        "max_constraint_violation": float(np.max(residuals)),
        "max_abs_input": maximum_input,
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_max_ms": float(np.max(latencies)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--recovery-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.recovery_steps < 1:
        parser.error("sample and recovery counts must be positive")

    config = PCBFConfig(horizon=24, multistart_count=3)
    cold_recovery = _closed_loop_recovery(
        config, args.recovery_steps, use_warm_start=False
    )
    warm_recovery = _closed_loop_recovery(
        config, args.recovery_steps, use_warm_start=True
    )
    value_delta = np.asarray(warm_recovery["values"]) - np.asarray(
        cold_recovery["values"]
    )
    max_warm_value_increase = float(np.max(value_delta, initial=0.0))
    if max_warm_value_increase > 20.0 * config.acceptable_tolerance:
        raise RuntimeError(
            "warm start degraded the recovery PCBF value by "
            f"{max_warm_value_increase}"
        )

    report = {
        "solver": "CasADi/IPOPT multistart nonlinear program",
        "config": {
            "horizon": config.horizon,
            "dt": config.dt,
            "a_max": config.a_max,
            "terminal_buffer_m": config.terminal_buffer_m,
            "multistart_count": config.multistart_count,
        },
        "safe_hover": _safe_hover_case(config),
        "closed_loop_recovery_cold": cold_recovery,
        "closed_loop_recovery_warm": warm_recovery,
        "warm_start_quality": {
            "max_warm_minus_cold_value": max_warm_value_increase,
            "value_guard_passed": True,
        },
        "random_stress": _random_stress(config, args.samples, args.seed),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
