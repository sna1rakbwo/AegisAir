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
from marllib.phase5_runner import _priority_order, _scenario, _snapshots
from marllib.policies.mappo import MappoPilot
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import (
    RuntimeAssurance,
    _closing_speed_2d,
    _d_safe_2d,
)


def _step_px4(env: MultiUAVEnv, v_cmd, dt: float, tau_px4: float) -> None:
    """First-order PX4 velocity-tracking step (positions/velocities only)."""
    arena = env.scenario.arena
    for idx in range(env.num_agents):
        v = env.velocities[idx]
        cmd = np.asarray(v_cmd[env.agent_ids[idx]], dtype=np.float64)
        if tau_px4 > 0:
            alpha = dt / (dt + tau_px4)
            v_next = v + alpha * (cmd - v)
        else:
            v_next = cmd
        v_next = v + np.clip(
            v_next - v, -env.scenario.accel_limit * dt, env.scenario.accel_limit * dt
        )
        env.velocities[idx] = v_next
    env.positions = env.positions + env.velocities * dt
    env.positions = np.clip(
        env.positions, (arena[0], arena[2]), (arena[1], arena[3])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--tau-ctrl", type=float, default=0.0)
    parser.add_argument("--sequential-pass", action="store_true")
    parser.add_argument("--urgent-drone", type=int, default=2)
    parser.add_argument("--aoi-ms", type=float, default=0.0)
    parser.add_argument("--tau-px4", type=float, default=0.0)
    parser.add_argument("--tau-px4-min", type=float, default=None)
    parser.add_argument("--tau-px4-max", type=float, default=None)
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
    ra = RuntimeAssurance(
        params=params,
        v_max=1.5,
        sampled_data=True,
        gamma=args.gamma,
        tau_px4=args.tau_px4,
        tau_px4_min=args.tau_px4_min,
        tau_px4_max=args.tau_px4_max,
    )
    sigma = ra.perception_sigma

    base_goals = {
        i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
        for i in env.agent_ids
    }
    aoi = args.aoi_ms / 1000.0
    seq_state = {
        "active": False,
        "order": _priority_order(env.agent_ids, args.urgent_drone),
        "idx": 0,
        "best": {i: float("inf") for i in env.agent_ids},
        "stall": {i: 0 for i in env.agent_ids},
    }
    velocity_scale = {i: 1.0 for i in env.agent_ids}

    first_neg = None
    min_rho = float("inf")
    min_rho_report = None
    for step in range(args.max_steps):
        t = step * env.scenario.dt
        positions = {i: env.positions[i] for i in env.agent_ids}
        velocities = {i: env.velocities[i] for i in env.agent_ids}
        goals = {i: np.asarray(base_goals[i][:2]) for i in env.agent_ids}

        if args.sequential_pass:
            for i in env.agent_ids:
                dist = float(
                    np.linalg.norm(env.positions[i] - np.asarray(base_goals[i][:2]))
                )
                if dist < seq_state["best"][i] - 0.02:
                    seq_state["best"][i] = dist
                    seq_state["stall"][i] = 0
                else:
                    seq_state["stall"][i] += 1
            if not seq_state["active"] and any(
                v >= 8 for v in seq_state["stall"].values()
            ):
                seq_state["active"] = True
                seq_state["idx"] = 0
                seq_state["best"] = {i: float("inf") for i in env.agent_ids}
                seq_state["stall"] = {i: 0 for i in env.agent_ids}
                seq_state["order"] = _priority_order(env.agent_ids, args.urgent_drone)
            if seq_state["active"]:
                right_of_way = seq_state["order"][seq_state["idx"]]
                for i in env.agent_ids:
                    velocity_scale[i] = 1.0 if i == right_of_way else 0.0
                if (
                    np.linalg.norm(
                        env.positions[right_of_way]
                        - np.asarray(base_goals[right_of_way][:2])
                    )
                    < env.scenario.goal_epsilon
                ):
                    seq_state["idx"] += 1
                    if seq_state["idx"] >= len(seq_state["order"]):
                        seq_state["active"] = False

        nominal = pilot.actions(positions=positions, velocities=velocities, goals=goals)
        nominal = {i: nominal[i] * velocity_scale.get(i, 1.0) for i in env.agent_ids}
        snapshots = _snapshots(env)
        aoi_dict = {
            (i, j): aoi
            for i in env.agent_ids
            for j in env.agent_ids
            if i != j
        }
        results = ra.filter(snapshots, nominal, t=t, aoi=aoi_dict)

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
                s_now = _d_safe_2d(vcl, aoi, sigma, sigma, params)
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
        s_next = _d_safe_2d(vcl_next, aoi, sigma, sigma, params)
        r = positions[i] - positions[j]
        v = velocities[i] - velocities[j]
        a_rel = a_nom[i] - a_nom[j]
        r_pred = r + v * args.dt + 0.5 * a_rel * args.dt * args.dt
        h_next_pred = float(np.dot(r_pred, r_pred)) - s_next * s_next

        # Step the actual environment with a PX4-like velocity-tracking model.
        _step_px4(env, v_safe, args.dt, args.tau_px4)
        p_i = env.positions[i]
        p_j = env.positions[j]
        d_actual = float(np.linalg.norm(p_i - p_j))
        vcl_actual = _closing_speed_2d(p_i, p_j, env.velocities[i], env.velocities[j])
        s_actual = _d_safe_2d(vcl_actual, aoi, sigma, sigma, params)
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
