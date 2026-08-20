#!/usr/bin/env python3
"""Controlled rollback episodes seeded from one-step tau-activation cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.robust_activation_search import _safe
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.safety import DroneSnapshot


def run_episode(case: dict, *, rollback_s: float, robust: bool, tau_true: float,
                steps: int, dt: float, tau_hat: float, tau_min: float,
                tau_max: float, qp_max_iters: int, recovery_trigger_m: float,
                recovery_release_m: float, recovery_speed_mps: float,
                recovery_release_steps: int) -> dict:
    params = RuntimeAssuranceParams()
    ra = RuntimeAssurance(
        params=params, v_max=1.5, sampled_data=True, gamma=0.1,
        tau_px4=tau_hat,
        tau_px4_min=tau_min if robust else tau_hat,
        tau_px4_max=tau_max if robust else tau_hat,
        qp_max_iters=qp_max_iters,
    )
    closing = float(case["closing_speed_mps"])
    yielding = float(case["yielding_speed_mps"])
    activation_d = float(case["s_now_m"] + case["delta_m"])
    positions = {0: np.array([0.0, 0.0]), 1: np.array([activation_d + rollback_s * closing, 0.0])}
    velocities = {0: np.array([closing + yielding, 0.0]), 1: np.array([yielding, 0.0])}
    start_x = {i: float(positions[i][0]) for i in (0, 1)}
    final_goals = {0: np.array([start_x[1], 0.0]), 1: np.array([start_x[0], 0.0])}
    waypoints = {0: np.array([start_x[0], 1.5]), 1: np.array([start_x[1], -1.5])}
    cross_goals = {0: np.array([start_x[1], 1.5]), 1: np.array([start_x[0], -1.5])}
    min_rho = float("inf")
    interventions = 0
    infeasible = 0
    activation = 0
    onset = None
    effort = 0.0
    completed = False
    recovery_state = "NORMAL"
    clear_count = 0
    mission_phase = "SEPARATE"
    for step in range(steps):
        t = step * dt
        snapshots = {
            i: DroneSnapshot(
                drone_id=i,
                position=(float(positions[i][0]), float(positions[i][1]), 0.0),
                velocity=(float(velocities[i][0]), float(velocities[i][1]), 0.0),
                timestamp_ms=int(round(t * 1000.0)),
            ) for i in (0, 1)
        }
        delta = positions[0] - positions[1]
        distance = float(np.linalg.norm(delta))
        vrel = velocities[0] - velocities[1]
        closing_now = max(0.0, -float(np.dot(delta, vrel)) / distance) if distance > 1e-9 else 0.0
        # Finite-state, bounded lateral recovery. This makes the corridor
        # viable without permanently replacing the mission controller.
        if recovery_state == "NORMAL" and distance <= recovery_trigger_m and closing_now > 0.1:
            recovery_state = "SEPARATE"
            clear_count = 0
        elif recovery_state == "SEPARATE":
            clear_now = distance >= recovery_release_m or closing_now <= 0.1
            clear_count = clear_count + 1 if clear_now else 0
            if clear_count >= recovery_release_steps:
                recovery_state = "RECOVER"
                clear_count = 0
        elif recovery_state == "RECOVER":
            clear_count += 1
            if clear_count >= recovery_release_steps:
                recovery_state = "NORMAL"
        if mission_phase == "SEPARATE":
            targets = waypoints
            if all(abs(float(positions[i][1] - waypoints[i][1])) <= 0.2 for i in (0, 1)):
                mission_phase = "CROSS"
        elif mission_phase == "CROSS":
            targets = cross_goals
            if all(abs(float(positions[i][0] - cross_goals[i][0])) <= 0.2 for i in (0, 1)):
                mission_phase = "REJOIN"
        else:
            targets = final_goals
        nominal = {i: np.clip(2.0 * (targets[i] - positions[i]), -1.5, 1.5) for i in (0, 1)}
        if recovery_state == "SEPARATE":
            nominal[0][1] += recovery_speed_mps
            nominal[1][1] -= recovery_speed_mps
            nominal = {i: np.clip(nominal[i], -1.5, 1.5) for i in (0, 1)}
        results = ra.filter(snapshots, nominal, t=t)
        commands = {i: np.asarray(results[i].safe_action) for i in (0, 1)}
        interventions += int(any(results[i].intervened for i in (0, 1)))
        if onset is None and any(results[i].intervened for i in (0, 1)):
            onset = t
        infeasible += int(not bool(ra.last_qp_feasible))
        effort += sum(float(np.linalg.norm(commands[i] - nominal[i])) for i in (0, 1)) * dt

        # The same activation diagnostic used by the episode benchmark.
        a_nom = {i: np.asarray(results[i].a_nom, dtype=float) for i in (0, 1)}
        activation += int(any(results[i].intervened for i in (0, 1)))
        s_now = _safe(closing_now, 0.0, params)
        rho = (distance - s_now) / s_now
        min_rho = min(min_rho, rho)
        old_positions = {i: positions[i].copy() for i in (0, 1)}
        old_velocities = {i: velocities[i].copy() for i in (0, 1)}
        alpha = 1.0 - np.exp(-dt / tau_true)
        beta = dt - tau_true * alpha
        for i in (0, 1):
            velocities[i] = old_velocities[i] + alpha * (commands[i] - old_velocities[i])
            positions[i] = old_positions[i] + old_velocities[i] * dt + beta * (commands[i] - old_velocities[i])
        if mission_phase == "REJOIN" and all(
                float(np.linalg.norm(positions[i] - final_goals[i])) <= 0.2 for i in (0, 1)):
            completed = True
            break
    return {
        "rollback_s": rollback_s, "robust": robust, "tau_true": tau_true,
        "steps": step + 1, "completed": completed, "min_rho": min_rho,
        "violation": min_rho < 0.0, "qp_infeasible_rate": infeasible / max(step + 1, 1),
        "intervention_onset_s": onset, "intervention_rate": interventions / max(step + 1, 1),
        "robust_activation_rate": activation / max(step + 1, 1), "control_effort": effort,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--rollback-seconds", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tau-min", type=float, default=0.53)
    parser.add_argument("--tau-max", type=float, default=1.76)
    parser.add_argument("--tau-hat", type=float, default=0.7)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--recovery-trigger-m", type=float, default=3.5)
    parser.add_argument("--recovery-release-m", type=float, default=4.0)
    parser.add_argument("--recovery-speed-mps", type=float, default=0.8)
    parser.add_argument("--recovery-release-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.search.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if c.get("robust_feasible")]
    if not cases:
        raise RuntimeError("search file contains no robust-feasible activation cases")
    case = cases[args.case_index % len(cases)]
    rng = np.random.default_rng(args.seed)
    rows = []
    for rollback in args.rollback_seconds:
        for _ in range(args.episodes):
            tau_true = float(rng.uniform(args.tau_min, args.tau_max))
            for robust in (False, True):
                rows.append(run_episode(
                    case, rollback_s=rollback, robust=robust, tau_true=tau_true,
                    steps=args.steps, dt=args.dt, tau_hat=args.tau_hat,
                    tau_min=args.tau_min, tau_max=args.tau_max,
                    qp_max_iters=args.qp_max_iters,
                    recovery_trigger_m=args.recovery_trigger_m,
                    recovery_release_m=args.recovery_release_m,
                    recovery_speed_mps=args.recovery_speed_mps,
                    recovery_release_steps=args.recovery_release_steps,
                ))
    summary = []
    for rollback in args.rollback_seconds:
        for robust in (False, True):
            group = [r for r in rows if r["rollback_s"] == rollback and r["robust"] == robust]
            summary.append({
                "rollback_s": rollback, "robust": robust, "episodes": len(group),
                "violation_rate": float(np.mean([r["violation"] for r in group])),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
                "mean_min_rho": float(np.mean([r["min_rho"] for r in group])),
                "qp_infeasible_rate": float(np.mean([r["qp_infeasible_rate"] for r in group])),
                "mean_intervention_onset_s": float(np.mean([
                    r["intervention_onset_s"] if r["intervention_onset_s"] is not None else args.steps * args.dt
                    for r in group
                ])),
                "mean_robust_activation_rate": float(np.mean([r["robust_activation_rate"] for r in group])),
            })
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    output = {"config": config, "case": case,
              "summary": summary, "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
