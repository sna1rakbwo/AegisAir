"""Offline evaluation of the predictive safety monitor.

Runs the baseline MARL trajectory (no CBF), then compares the predictor's
future-margin output against the actual future trajectory:
    - margin prediction error;
    - TTSB error;
    - conflict detection precision / recall / F1 / FP / FN.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import default_curriculum
from marllib.envs.multi_uav import MultiUAVEnv
from marllib.policies.mappo import MAPPO
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.predictor import PredictiveMonitor


def actual_future_margin(
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    t: int,
    horizon: float,
    dt: float,
    params: RuntimeAssuranceParams,
) -> tuple[float, float | None]:
    """Return (min actual rho over the future, time to first rho<=0)."""
    steps = int(round(horizon / dt))
    end = min(t + steps, len(positions) - 1)
    best_rho = float("inf")
    ttsb = None
    for k in range(t + 1, end + 1):
        p_i = positions[k][0]
        p_j = positions[k][1]
        r = p_i - p_j
        distance = float(np.linalg.norm(r))
        v_i = velocities[k][0]
        v_j = velocities[k][1]
        if distance == 0:
            rho = -1.0
        else:
            v_cl = max(0.0, -float(np.dot(r, v_i - v_j)) / distance)
            d_safe = (
                params.d0
                + v_cl * params.tau_r
                + v_cl**2 / (2.0 * params.a_eff)
                + params.beta * math.sqrt(2.0 * 0.0)
                + params.v_max * 0.0
                + 0.5 * params.a_max * 0.0**2
            )
            rho = (distance - d_safe) / d_safe
        if rho < best_rho:
            best_rho = rho
        if ttsb is None and rho <= 0.0:
            ttsb = (k - t) * dt
    return best_rho, ttsb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--rho-pred", type=float, default=0.3)
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
    params = RuntimeAssuranceParams()
    monitor = PredictiveMonitor()

    margin_errors: list[float] = []
    ttsb_errors: list[float] = []
    tp = fp = fn = tn = 0

    for seed in range(args.eval_seeds):
        obs, _ = env.reset(seed=seed)
        positions = [env.positions.copy()]
        velocities = [env.velocities.copy()]
        for step in range(200):
            nominal = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
            obs, _, terminated, _, infos = env.step(nominal)
            positions.append(env.positions.copy())
            velocities.append(env.velocities.copy())
            if any(i["collided"] for i in infos.values()) or all(i["reached"] for i in infos.values()):
                break

        if scenario.num_agents < 2:
            continue

        for t in range(len(positions) - 1):
            p_i = (float(positions[t][0, 0]), float(positions[t][0, 1]), 0.0)
            p_j = (float(positions[t][1, 0]), float(positions[t][1, 1]), 0.0)
            v_i = (float(velocities[t][0, 0]), float(velocities[t][0, 1]), 0.0)
            v_j = (float(velocities[t][1, 0]), float(velocities[t][1, 1]), 0.0)
            monitor.update_acceleration(0, v_i, scenario.dt)
            monitor.update_acceleration(1, v_j, scenario.dt)
            pred = monitor.predict(
                agent_i=0,
                agent_j=1,
                p_i=p_i,
                p_j=p_j,
                v_i=v_i,
                v_j=v_j,
                sigma_i=0.0,
                sigma_j=0.0,
                aoi=0.0,
                params=params,
                horizon=params.prediction_horizon,
            )
            actual_rho, actual_ttsb = actual_future_margin(
                positions, velocities, t, params.prediction_horizon, scenario.dt, params
            )
            margin_errors.append(abs(pred.rho_min_pred - actual_rho))
            if pred.ttsb is not None and actual_ttsb is not None:
                ttsb_errors.append(abs(pred.ttsb - actual_ttsb))

            pred_conflict = pred.rho_min_pred < args.rho_pred
            actual_conflict = actual_rho <= 0.0
            if pred_conflict and actual_conflict:
                tp += 1
            elif pred_conflict and not actual_conflict:
                fp += 1
            elif not pred_conflict and actual_conflict:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "scenario": scenario.name,
        "eval_seeds": args.eval_seeds,
        "mean_margin_error": float(np.mean(margin_errors)) if margin_errors else None,
        "mean_ttsb_error_s": float(np.mean(ttsb_errors)) if ttsb_errors else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
