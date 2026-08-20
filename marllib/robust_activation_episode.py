#!/usr/bin/env python3
"""Replay a fixed two-UAV activation snapshot under hidden PX4 tau values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.robust_activation_search import (
    _alpha,
    _projected_g,
    _safe,
    _state,
)
from swarm.ra.hocbf import beta_of_tau, solve_robust_sampled_data_qp
from swarm.ra.margins import RuntimeAssuranceParams


def run_episode(
    *, case: dict, tau_true: float, robust: bool, steps: int,
    dt: float, gamma: float, tau_hat: float, tau_min: float, tau_max: float,
) -> dict:
    params = RuntimeAssuranceParams()
    closing0 = float(case["closing_speed_mps"])
    yielding0 = float(case["yielding_speed_mps"])
    s0 = _safe(closing0, 0.0, params)
    positions, velocities, nominal_velocity = _state(
        s0 + float(case["delta_m"]), closing0, yielding0
    )
    command = {i: nominal_velocity[i].copy() for i in (0, 1)}
    min_rho = float("inf")
    violation_steps = 0
    activation_steps = 0
    intervention_steps = 0
    delta_u_max = 0.0
    rows = []
    for step in range(steps):
        r = positions[0] - positions[1]
        distance = float(np.linalg.norm(r))
        vrel = velocities[0] - velocities[1]
        closing = max(0.0, -float(np.dot(r, vrel)) / distance)
        s_now = _safe(closing, 0.0, params)
        rho = (distance - s_now) / s_now
        min_rho = min(min_rho, rho)
        violation_steps += int(rho < 0.0)

        a_nom = {
            i: np.clip(2.0 * (command[i] - velocities[i]), -2.0, 2.0)
            for i in (0, 1)
        }
        alpha_hat = _alpha(dt, tau_hat)
        v_hat = {
            i: velocities[i] + alpha_hat * a_nom[i] * dt for i in (0, 1)
        }
        vcl_hat = max(
            0.0,
            -float(np.dot(r, v_hat[0] - v_hat[1])) / distance,
        )
        s_next = _safe(vcl_hat, 0.0, params)
        s_next_robust = max(
            _safe(
                max(
                    0.0,
                    -float(np.dot(
                        r,
                        (
                            velocities[0] + _alpha(dt, ti) * a_nom[0] * dt
                            - velocities[1] - _alpha(dt, tj) * a_nom[1] * dt
                        ),
                    )) / distance,
                ),
                0.0,
                params,
            )
            for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
        )
        beta_hat = beta_of_tau(dt, tau_hat)
        beta_cfg = (
            {0: (beta_hat, beta_hat), 1: (beta_hat, beta_hat)}
            if not robust
            else {
                0: (beta_of_tau(dt, tau_min), beta_of_tau(dt, tau_max)),
                1: (beta_of_tau(dt, tau_min), beta_of_tau(dt, tau_max)),
            }
        )
        a_safe, feasible, _ = solve_robust_sampled_data_qp(
            a_nom=a_nom, positions=positions, velocities=velocities,
            s_now={(0, 1): s_now},
            s_next={(0, 1): s_next_robust if robust else s_next}, dt=dt,
            gamma=gamma, a_max=2.0, beta=beta_cfg,
        )
        delta_u = max(float(np.linalg.norm(a_safe[i] - a_nom[i])) for i in (0, 1))
        intervention_steps += int(delta_u > 1e-8)
        delta_u_max = max(delta_u_max, delta_u)
        _, _, _, _, _, g_nom, g_worst = _metrics_for_current(
            positions, velocities, a_nom, dt, gamma, tau_hat, tau_min, tau_max,
            s_now, s_next, s_next_robust,
        )
        active = g_nom >= 0.0 and g_worst < 0.0
        activation_steps += int(active)
        if step < 5:
            rows.append({
                "step": step, "rho": rho, "g_nominal": g_nom,
                "g_worst": g_worst, "robust_active": active,
                "feasible": bool(feasible), "delta_u": delta_u,
            })

        alpha_true = _alpha(dt, tau_true)
        beta_true = beta_of_tau(dt, tau_true)
        old_positions = {i: positions[i].copy() for i in (0, 1)}
        old_velocities = {i: velocities[i].copy() for i in (0, 1)}
        for i in (0, 1):
            velocities[i] = old_velocities[i] + alpha_true * a_safe[i] * dt
            positions[i] = old_positions[i] + old_velocities[i] * dt + beta_true * a_safe[i] * dt
    return {
        "tau_true": tau_true, "robust": robust, "steps": steps,
        "min_rho": min_rho, "violation_steps": violation_steps,
        "violation_rate": violation_steps / max(steps, 1),
        "robust_activation_rate": activation_steps / max(steps, 1),
        "intervention_rate": intervention_steps / max(steps, 1),
        "delta_u_max": delta_u_max, "first_rows": rows,
    }


def _metrics_for_current(positions, velocities, a_nom, dt, gamma, tau_hat, tau_min, tau_max, s_now, s_next, s_next_robust):
    r = positions[0] - positions[1]
    v = velocities[0] - velocities[1]
    def g(ti, tj):
        return _projected_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, s_next)
    return positions, velocities, a_nom, s_now, s_next, g(tau_hat, tau_hat), min(
        _projected_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, s_next_robust)
        for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.search.read_text())
    case = data["cases"][args.case_index]
    common = dict(
        case=case, steps=args.steps, dt=float(data["dt"]), gamma=float(data["gamma"]),
        tau_hat=float(data["tau_hat"]), tau_min=float(data["tau_interval"][0]),
        tau_max=float(data["tau_interval"][1]),
    )
    results = []
    for tau_true in data["tau_interval"]:
        for robust in (False, True):
            results.append(run_episode(tau_true=tau_true, robust=robust, **common))
    output = {"case": case, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
