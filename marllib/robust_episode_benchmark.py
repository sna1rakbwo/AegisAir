#!/usr/bin/env python3
"""Two-UAV episode benchmark for nominal vs robust PX4 execution uncertainty."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.robust_activation_search import _next_safe, _projected_g, _safe
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.safety import DroneSnapshot


def run_episode(*, distance: float, robust: bool, tau_true: float, seed: int,
                max_steps: int, dt: float, tau_hat: float,
                tau_min: float, tau_max: float, qp_max_iters: int,
                scenario: str, guard_trigger_m: float, guard_release_m: float,
                guard_speed_mps: float, guard_closing_threshold_mps: float,
                guard_release_steps: int) -> dict:
    del seed  # reserved for future lateral perturbations
    params = RuntimeAssuranceParams()
    ra = RuntimeAssurance(
        params=params, v_max=1.5, sampled_data=True, gamma=0.1,
        tau_px4=tau_hat,
        tau_px4_min=tau_min if robust else tau_hat,
        tau_px4_max=tau_max if robust else tau_hat,
        qp_max_iters=qp_max_iters,
    )
    positions = {0: np.array([-distance / 2.0, 0.0]), 1: np.array([distance / 2.0, 0.0])}
    velocities = {0: np.zeros(2), 1: np.zeros(2)}
    if scenario == "swap_1d":
        goals = {0: np.array([distance / 2.0, 0.0]), 1: np.array([-distance / 2.0, 0.0])}
    elif scenario == "lateral_crossing":
        goals = {0: np.array([distance / 2.0, 1.5]), 1: np.array([-distance / 2.0, -1.5])}
    elif scenario == "near_crossing":
        goals = {0: np.array([distance / 2.0, 0.5]), 1: np.array([-distance / 2.0, 0.5])}
    elif scenario == "guarded_crossing":
        # A deliberately close crossing.  The finite-state guard should
        # separate the vehicles briefly, then release and let the mission
        # controller finish; a large lateral offset would never activate it.
        goals = {0: np.array([distance / 2.0, 0.5]), 1: np.array([-distance / 2.0, -0.5])}
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    min_rho = float("inf")
    min_distance = float("inf")
    violations = 0
    collisions = 0
    infeasible = 0
    interventions = 0
    robust_active = 0
    intervention_onset = None
    effort = 0.0
    completed = False
    guard_state = "NORMAL"
    guard_clear_count = 0
    guard_recover_count = 0
    guard_steps = 0
    guard_activations = 0

    for step in range(max_steps):
        t = step * dt
        snapshots = {
            i: DroneSnapshot(
                drone_id=i,
                position=(float(positions[i][0]), float(positions[i][1]), 0.0),
                velocity=(float(velocities[i][0]), float(velocities[i][1]), 0.0),
                timestamp_ms=int(round(t * 1000.0)),
            )
            for i in (0, 1)
        }
        nominal = {
            i: np.clip(2.0 * (goals[i] - positions[i]), -1.5, 1.5)
            for i in (0, 1)
        }
        if scenario == "guarded_crossing":
            delta_now = positions[0] - positions[1]
            distance_now = float(np.linalg.norm(delta_now))
            vrel_now = velocities[0] - velocities[1]
            closing_now = max(
                0.0,
                -float(np.dot(delta_now, vrel_now)) / distance_now,
            ) if distance_now > 1e-9 else 0.0
            # Finite-state guard: activate once near a closing encounter,
            # then release with hysteresis instead of overriding the mission
            # controller forever.
            if guard_state == "NORMAL":
                if (closing_now > guard_closing_threshold_mps
                        and distance_now <= guard_trigger_m):
                    guard_state = "SEPARATE"
                    guard_clear_count = 0
                    guard_activations += 1
            elif guard_state == "SEPARATE":
                guard_steps += 1
                clear_now = (distance_now >= guard_release_m
                             or closing_now <= guard_closing_threshold_mps)
                guard_clear_count = guard_clear_count + 1 if clear_now else 0
                if guard_clear_count >= guard_release_steps:
                    guard_state = "RECOVER"
                    guard_recover_count = 0
            elif guard_state == "RECOVER":
                guard_recover_count += 1
                if guard_recover_count >= guard_release_steps:
                    guard_state = "NORMAL"
            if guard_state == "SEPARATE":
                nominal[0][1] += guard_speed_mps
                nominal[1][1] -= guard_speed_mps
                nominal = {i: np.clip(nominal[i], -1.5, 1.5) for i in (0, 1)}
        results = ra.filter(snapshots, nominal, t=t)
        safe_commands = {i: np.asarray(results[i].safe_action) for i in (0, 1)}
        interventions += int(any(results[i].intervened for i in (0, 1)))
        if intervention_onset is None and any(results[i].intervened for i in (0, 1)):
            intervention_onset = t
        effort += sum(float(np.linalg.norm(safe_commands[i] - nominal[i])) for i in (0, 1)) * dt
        infeasible += int(not bool(ra.last_qp_feasible))

        delta = positions[0] - positions[1]
        distance_now = float(np.linalg.norm(delta))
        vrel = velocities[0] - velocities[1]
        closing = max(0.0, -float(np.dot(delta, vrel)) / distance_now) if distance_now > 1e-9 else 0.0
        d_safe = _safe(closing, 0.0, params)
        rho = (distance_now - d_safe) / d_safe
        min_rho = min(min_rho, rho)
        min_distance = min(min_distance, distance_now)
        violations += int(rho < 0.0)
        collisions += int(distance_now < params.d0)

        # Robust Constraint Activation Rate, evaluated against the shared
        # nominal acceleration before the controller changes it.
        a_nom = {
            i: np.asarray(results[i].a_nom, dtype=np.float64) for i in (0, 1)
        }
        s_now = d_safe
        alpha_hat = 1.0 - np.exp(-dt / tau_hat) if tau_hat > 0 else 1.0
        v_hat = {i: velocities[i] + alpha_hat * a_nom[i] * dt for i in (0, 1)}
        vcl_hat = max(0.0, -float(np.dot(delta, v_hat[0] - v_hat[1])) / distance_now)
        s_next_nom = _safe(vcl_hat, 0.0, params)
        s_next_rob = max(
            _next_safe(
                positions, velocities, a_nom, dt, ti, tj, params, 0.0
            )[1]
            for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
        )
        n = delta / distance_now if distance_now > 1e-9 else np.array([1.0, 0.0])
        g_nom = _projected_g(
            positions, velocities, a_nom, dt, 0.1, tau_hat, tau_hat, s_now, s_next_nom
        )
        g_rob = min(
            _projected_g(
                positions, velocities, a_nom, dt, 0.1, ti, tj, s_now, s_next_rob
            )
            for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
        )
        robust_active += int(g_nom >= 0.0 and g_rob < 0.0)

        # Exact PX4 ZOH execution under the hidden tau_true.
        old_positions = {i: positions[i].copy() for i in (0, 1)}
        old_velocities = {i: velocities[i].copy() for i in (0, 1)}
        alpha_true = 1.0 - np.exp(-dt / tau_true) if tau_true > 0 else 1.0
        beta_true = dt - tau_true * alpha_true if tau_true > 0 else dt
        for i in (0, 1):
            velocities[i] = old_velocities[i] + alpha_true * (safe_commands[i] - old_velocities[i])
            positions[i] = old_positions[i] + old_velocities[i] * dt + beta_true * (
                safe_commands[i] - old_velocities[i]
            )

        if all(float(np.linalg.norm(positions[i] - goals[i])) <= 0.2 for i in (0, 1)):
            completed = True
            break

    return {
        "distance_start_m": distance,
        "robust": robust,
        "tau_true": tau_true,
        "steps": step + 1,
        "completed": completed,
        "min_rho": min_rho,
        "min_distance_m": min_distance,
        "violation": violations > 0,
        "violation_rate": violations / max(step + 1, 1),
        "collision": collisions > 0,
        "qp_infeasible_rate": infeasible / max(step + 1, 1),
        "intervention_onset_s": intervention_onset,
        "intervention_rate": interventions / max(step + 1, 1),
        "robust_activation_rate": robust_active / max(step + 1, 1),
        "control_effort": effort,
        "guard_activations": guard_activations,
        "guard_activation_rate": guard_steps / max(step + 1, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--distances", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    parser.add_argument("--scenario", choices=["swap_1d", "lateral_crossing", "near_crossing", "guarded_crossing"], default="guarded_crossing")
    parser.add_argument("--guard-trigger-m", type=float, default=4.0)
    parser.add_argument("--guard-release-m", type=float, default=4.5)
    parser.add_argument("--guard-speed-mps", type=float, default=0.8)
    parser.add_argument("--guard-closing-threshold-mps", type=float, default=0.1)
    parser.add_argument("--guard-release-steps", type=int, default=3)
    parser.add_argument("--tau-min", type=float, default=0.53)
    parser.add_argument("--tau-max", type=float, default=1.76)
    parser.add_argument("--tau-hat", type=float, default=0.7)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    results = []
    for distance in args.distances:
        for episode in range(args.episodes):
            tau_true = float(rng.uniform(args.tau_min, args.tau_max))
            for robust in (False, True):
                results.append(run_episode(
                    distance=distance, robust=robust, tau_true=tau_true,
                    seed=args.seed + episode, max_steps=args.max_steps,
                    dt=args.dt, tau_hat=args.tau_hat,
                    tau_min=args.tau_min, tau_max=args.tau_max,
                    qp_max_iters=args.qp_max_iters,
                    scenario=args.scenario,
                    guard_trigger_m=args.guard_trigger_m,
                    guard_release_m=args.guard_release_m,
                    guard_speed_mps=args.guard_speed_mps,
                    guard_closing_threshold_mps=args.guard_closing_threshold_mps,
                    guard_release_steps=args.guard_release_steps,
                ))
    summary = []
    for distance in args.distances:
        for robust in (False, True):
            rows = [r for r in results if r["distance_start_m"] == distance and r["robust"] == robust]
            summary.append({
                "distance_start_m": distance,
                "robust": robust,
                "episodes": len(rows),
                "violation_rate": float(np.mean([r["violation"] for r in rows])),
                "collision_rate": float(np.mean([r["collision"] for r in rows])),
                "completion_rate": float(np.mean([r["completed"] for r in rows])),
                "mean_min_rho": float(np.mean([r["min_rho"] for r in rows])),
                "min_min_rho": float(np.min([r["min_rho"] for r in rows])),
                "qp_infeasible_rate": float(np.mean([r["qp_infeasible_rate"] for r in rows])),
                "mean_intervention_onset_s": float(np.mean([
                    r["intervention_onset_s"] if r["intervention_onset_s"] is not None else args.max_steps * args.dt
                    for r in rows
                ])),
                "mean_control_effort": float(np.mean([r["control_effort"] for r in rows])),
                "mean_robust_activation_rate": float(np.mean([r["robust_activation_rate"] for r in rows])),
                "mean_guard_activation_rate": float(np.mean([r["guard_activation_rate"] for r in rows])),
                "mean_guard_activations": float(np.mean([r["guard_activations"] for r in rows])),
            })
    output = {"config": vars(args) | {"out": str(args.out)}, "summary": summary, "episodes": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
