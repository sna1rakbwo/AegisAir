#!/usr/bin/env python3
"""Minimal HOCBF-vs-ZOCBF comparison on a two-UAV double integrator.

This is a controlled numerical comparison, not a full reproduction of the
ZOCBF paper.  The one-dimensional head-on geometry lets the exact one-step
squared-distance ZOCBF condition be represented as a linear half-space while
the vehicle ordering is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.ra.hocbf import solve_acceleration_qp


def _zocbf_acceleration(
    *, positions: dict[int, np.ndarray], velocities: dict[int, np.ndarray],
    a_nom: dict[int, np.ndarray], d_safe: float, dt: float, gamma_c: float,
    delta: float, a_max: float,
) -> tuple[dict[int, np.ndarray], bool]:
    """Project nominal acceleration onto the exact 1-D ZOCBF constraint."""
    r = float(positions[0][0] - positions[1][0])
    v = float(velocities[0][0] - velocities[1][0])
    if r >= 0.0:
        return {
            i: np.clip(-velocities[i], -a_max, a_max) for i in (0, 1)
        }, False
    h_now = r * r - d_safe * d_safe
    required_h = max(0.0, (1.0 - gamma_c) * h_now + delta)
    target_distance = math.sqrt(d_safe * d_safe + required_h)

    # r_next = r + v*dt + 0.5*(a0-a1)*dt^2 <= -target_distance.
    # Write c^T A >= b for A=[a0x,a0y,a1x,a1y].
    c = np.array([-0.5 * dt * dt, 0.0, 0.5 * dt * dt, 0.0])
    b = target_distance + r + v * dt
    x = np.array([a_nom[0][0], a_nom[0][1], a_nom[1][0], a_nom[1][1]], dtype=float)

    # Dykstra projection for one box and one half-space.
    corrections = [np.zeros_like(x), np.zeros_like(x)]
    for _ in range(200):
        y = x + corrections[0]
        projected = np.clip(y, -a_max, a_max)
        corrections[0] = y - projected
        x = projected
        y = x + corrections[1]
        residual = b - float(np.dot(c, y))
        projected = y if residual <= 0.0 else y + residual * c / float(np.dot(c, c))
        corrections[1] = y - projected
        x = projected

    feasible = bool(np.all(np.abs(x) <= a_max + 1e-6) and np.dot(c, x) >= b - 1e-6)
    if not feasible:
        return {
            0: np.array([-min(abs(velocities[0][0]), a_max), 0.0]),
            1: np.array([min(abs(velocities[1][0]), a_max), 0.0]),
        }, False
    return {0: x[0:2].copy(), 1: x[2:4].copy()}, True


def run_case(*, controller: str, distance: float, speed: float, d_safe: float,
             dt: float, steps: int, k1: float, k2: float, gamma_c: float,
             delta: float, a_max: float) -> dict:
    positions = {0: np.array([-distance / 2.0, 0.0]), 1: np.array([distance / 2.0, 0.0])}
    velocities = {0: np.array([speed, 0.0]), 1: np.array([-speed, 0.0])}
    a_nom = {0: np.zeros(2), 1: np.zeros(2)}
    min_h_sample = float("inf")
    min_h_intersample = float("inf")
    interventions = 0
    infeasible = 0
    effort = 0.0

    for _ in range(steps):
        if controller == "hocbf":
            accelerations, feasible, _ = solve_acceleration_qp(
                a_nom=a_nom, positions=positions, velocities=velocities,
                d_safe={(0, 1): d_safe}, k1=k1, k2=k2, a_max=a_max,
            )
        elif controller == "zocbf":
            accelerations, feasible = _zocbf_acceleration(
                positions=positions, velocities=velocities, a_nom=a_nom,
                d_safe=d_safe, dt=dt, gamma_c=gamma_c, delta=delta,
                a_max=a_max,
            )
        else:
            raise ValueError(controller)
        infeasible += int(not feasible)
        interventions += int(any(np.linalg.norm(accelerations[i] - a_nom[i]) > 1e-6 for i in (0, 1)))
        effort += sum(float(np.linalg.norm(accelerations[i])) for i in (0, 1)) * dt

        p_old = {i: positions[i].copy() for i in (0, 1)}
        v_old = {i: velocities[i].copy() for i in (0, 1)}
        for fraction in np.linspace(0.0, 1.0, 21):
            tau = fraction * dt
            r_tau = (
                p_old[0] + v_old[0] * tau + 0.5 * accelerations[0] * tau * tau
                - p_old[1] - v_old[1] * tau - 0.5 * accelerations[1] * tau * tau
            )
            min_h_intersample = min(
                min_h_intersample, float(np.dot(r_tau, r_tau) - d_safe * d_safe)
            )
        for i in (0, 1):
            positions[i] = p_old[i] + v_old[i] * dt + 0.5 * accelerations[i] * dt * dt
            velocities[i] = v_old[i] + accelerations[i] * dt
        r = positions[0] - positions[1]
        min_h_sample = min(min_h_sample, float(np.dot(r, r) - d_safe * d_safe))

    return {
        "controller": controller, "distance_start_m": distance,
        "speed_each_mps": speed, "min_h_sample": min_h_sample,
        "min_h_intersample": min_h_intersample,
        "sample_violation": min_h_sample < -1e-8,
        "intersample_violation": min_h_intersample < -1e-8,
        "intervention_rate": interventions / steps,
        "qp_infeasible_rate": infeasible / steps, "control_effort": effort,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=float, nargs="+", default=[4.0, 5.0, 6.0])
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--d-safe", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--k1", type=float, default=3.0)
    parser.add_argument("--k2", type=float, default=3.0)
    parser.add_argument("--gamma-c", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--a-max", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        run_case(
            controller=controller, distance=distance, speed=speed,
            d_safe=args.d_safe, dt=args.dt, steps=args.steps, k1=args.k1,
            k2=args.k2, gamma_c=args.gamma_c, delta=args.delta,
            a_max=args.a_max,
        )
        for distance in args.distances for speed in args.speeds
        for controller in ("hocbf", "zocbf")
    ]
    summary = {}
    for controller in ("hocbf", "zocbf"):
        group = [row for row in rows if row["controller"] == controller]
        summary[controller] = {
            "cases": len(group),
            "sample_violation_cases": sum(row["sample_violation"] for row in group),
            "intersample_violation_cases": sum(row["intersample_violation"] for row in group),
            "infeasible_cases": sum(row["qp_infeasible_rate"] > 0.0 for row in group),
            "mean_min_h_sample": float(np.mean([row["min_h_sample"] for row in group])),
            "worst_min_h_sample": float(np.min([row["min_h_sample"] for row in group])),
            "mean_min_h_intersample": float(np.mean([row["min_h_intersample"] for row in group])),
            "worst_min_h_intersample": float(np.min([row["min_h_intersample"] for row in group])),
            "mean_intervention_rate": float(np.mean([row["intervention_rate"] for row in group])),
            "mean_control_effort": float(np.mean([row["control_effort"] for row in group])),
        }
    output = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "scope": "controlled numerical comparison; not a full ZOCBF theorem reproduction",
              "summary": summary, "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
