#!/usr/bin/env python3
"""Exact sampled-data barrier residual reproduction in the lightweight sim.

Recomputes the QP's per-pair ``h_now`` / ``h_next_pred`` and compares them with
the actual next-step ``h`` from ``MultiUAVEnv.step``, so the source of a
``rho < 0`` violation can be decomposed into position vs d_safe prediction
error without PX4/Gazebo noise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, _snapshots
from marllib.policies.mappo import MappoPilot
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import (
    RuntimeAssurance,
    _closing_speed_2d,
    _d_safe_2d,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau-ctrl", type=float, default=0.0)
    args = parser.parse_args()

    spec = _scenario("multi_uav")
    spec["scenario"] = replace(spec["scenario"], dt=args.dt)
    env = MultiUAVEnv(spec["scenario"])
    env.reset(seed=0)

    pilot = MappoPilot(
        args.checkpoint,
        obs_dim=4 + 5 * spec["scenario"].max_neighbors,
        num_agents=spec["scenario"].num_agents,
        speed_limit=spec["scenario"].speed_limit,
        max_neighbors=spec["scenario"].max_neighbors,
    )
    params = RuntimeAssuranceParams(tau_ctrl=args.tau_ctrl)
    ra = RuntimeAssurance(params=params, v_max=1.5, sampled_data=True, gamma=args.gamma)
    sigma = ra.perception_sigma

    base_goals = {
        i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
        for i in env.agent_ids
    }

    first_neg = None
    min_rho = float("inf")
    min_rho_report = None
    for step in range(args.max_steps):
        t = step * env.scenario.dt
        positions = {i: env.positions[i] for i in env.agent_ids}
        velocities = {i: env.velocities[i] for i in env.agent_ids}
        goals = {i: np.asarray(base_goals[i][:2]) for i in env.agent_ids}
        nominal = pilot.actions(positions=positions, velocities=velocities, goals=goals)
        snapshots = _snapshots(env)
        results = ra.filter(snapshots, nominal, t=t)

        a_nom = {i: np.asarray(results[i].a_nom) for i in env.agent_ids}
        a_safe = {i: np.asarray(results[i].a_safe) for i in env.agent_ids}
        v_safe = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}

        # Identify the pair with the worst actual h_now.
        ids = env.agent_ids
        worst = None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                vcl = _closing_speed_2d(positions[i], positions[j], velocities[i], velocities[j])
                s_now = _d_safe_2d(vcl, 0.0, sigma, sigma, params)
                d = float(np.linalg.norm(positions[i] - positions[j]))
                h_now = d * d - s_now * s_now
                if worst is None or h_now < worst[0]:
                    worst = (h_now, i, j, s_now, d)

        h_now, i, j, s_now, d_now = worst
        vcl_next = _closing_speed_2d(
            positions[i], positions[j],
            velocities[i] + a_nom[i] * args.dt,
            velocities[j] + a_nom[j] * args.dt,
        )
        s_next = _d_safe_2d(vcl_next, 0.0, sigma, sigma, params)
        r = positions[i] - positions[j]
        v = velocities[i] - velocities[j]
        a_rel = a_nom[i] - a_nom[j]
        r_pred = r + v * args.dt + 0.5 * a_rel * args.dt * args.dt
        h_next_pred = float(np.dot(r_pred, r_pred)) - s_next * s_next

        # Step the actual environment with the safe actions.
        _, _, _, _, infos = env.step(v_safe)
        p_i = env.positions[i]
        p_j = env.positions[j]
        d_actual = float(np.linalg.norm(p_i - p_j))
        vcl_actual = _closing_speed_2d(p_i, p_j, env.velocities[i], env.velocities[j])
        s_actual = _d_safe_2d(vcl_actual, 0.0, sigma, sigma, params)
        h_next_actual = d_actual * d_actual - s_actual * s_actual

        rho = float(np.min([r.safety_margin for r in results.values()]))
        if rho < min_rho:
            min_rho = rho
            min_rho_report = (
                step, t, i, j, d_now, s_now, h_now,
                float(np.linalg.norm(r_pred)), s_next, h_next_pred,
                (1.0 - args.gamma) * h_now,
                d_actual, s_actual, h_next_actual,
                h_next_actual - (1.0 - args.gamma) * h_now,
                max(float(np.linalg.norm(a_safe[k])) for k in ids),
            )
        if first_neg is None and rho < 0:
            first_neg = step
            break

    label = "first rho<0" if first_neg is not None else "min-rho point"
    (
        step, t, i, j, d_now, s_now, h_now,
        r_pred_norm, s_next, h_next_pred, budget,
        d_actual, s_actual, h_next_actual, residual, a_safe_max,
    ) = min_rho_report
    print(
        "%s at step=%d t=%.2f pair=(%d,%d) rho=%.4f\n"
        "  d_now=%.4f s_now=%.4f h_now=%.5f\n"
        "  pred: |r_pred|=%.4f s_next=%.4f h_next_pred=%.5f  budget=%.5f\n"
        "  actual: d_next=%.4f s_actual=%.4f h_next_actual=%.5f\n"
        "  residual = h_next_actual - budget = %.5f\n"
        "  |a_safe|_max=%.3f (a_max=2.0)"
        % (
            label, step, t, i, j, min_rho,
            d_now, s_now, h_now,
            r_pred_norm, s_next, h_next_pred, budget,
            d_actual, s_actual, h_next_actual, residual, a_safe_max,
        )
    )
    return 0 if first_neg is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
