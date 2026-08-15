"""Predictor ablation: P0 CPA / P1 CV / P2 CV+U / P3 CA+U / P4 ensemble."""

from __future__ import annotations

import argparse
import json
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
from swarm.ra.predictor import PredictiveMonitor, closest_point_of_approach


def collect_trajectories(scenario, checkpoint, eval_seeds):
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(checkpoint, weights_only=False))
    trajs = []
    for seed in range(eval_seeds):
        obs, _ = env.reset(seed=seed)
        positions = [env.positions.copy()]
        velocities = [env.velocities.copy()]
        for _ in range(200):
            nominal = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
            obs, _, terminated, _, infos = env.step(nominal)
            positions.append(env.positions.copy())
            velocities.append(env.velocities.copy())
            if any(i["collided"] for i in infos.values()) or all(i["reached"] for i in infos.values()):
                break
        trajs.append((positions, velocities))
    return trajs


def actual_future(positions, velocities, t, i, j, horizon, dt, params):
    steps = int(round(horizon / dt))
    end = min(t + steps, len(positions) - 1)
    best = float("inf")
    for k in range(t + 1, end + 1):
        r = positions[k][i] - positions[k][j]
        d = float(np.linalg.norm(r))
        v_cl = max(0.0, -float(np.dot(r, velocities[k][i] - velocities[k][j])) / d) if d > 0 else 0.0
        d_safe = params.d0 + v_cl * params.tau_r + v_cl**2 / (2.0 * params.a_eff)
        rho = (d - d_safe) / d_safe if d_safe > 0 else -1.0
        best = min(best, rho)
    return best


def evaluate_variant(trajs, scenario, params, variant, q_pred):
    dt = scenario.dt
    horizon = params.prediction_horizon
    tp = fp = fn = tn = 0
    margin_errors = []
    for positions, velocities in trajs:
        cv_mon = PredictiveMonitor(use_ca=False, q_pred=q_pred)
        ca_mon = PredictiveMonitor(use_ca=True, q_pred=q_pred)
        for t in range(len(positions) - 1):
            for i in range(scenario.num_agents):
                for j in range(i + 1, scenario.num_agents):
                    p_i = (float(positions[t][i, 0]), float(positions[t][i, 1]), 0.0)
                    p_j = (float(positions[t][j, 0]), float(positions[t][j, 1]), 0.0)
                    v_i = (float(velocities[t][i, 0]), float(velocities[t][i, 1]), 0.0)
                    v_j = (float(velocities[t][j, 0]), float(velocities[t][j, 1]), 0.0)
                    cv_mon.update_acceleration(i, v_i, dt)
                    cv_mon.update_acceleration(j, v_j, dt)
                    ca_mon.update_acceleration(i, v_i, dt)
                    ca_mon.update_acceleration(j, v_j, dt)

                    actual = actual_future(positions, velocities, t, i, j, horizon, dt, params)

                    if variant == "P0_CPA":
                        cpa = closest_point_of_approach(p_i, p_j, v_i, v_j, horizon)
                        r = np.array(p_i[:2]) - np.array(p_j[:2])
                        d = float(np.linalg.norm(r))
                        v_cl = max(0.0, -float(np.dot(r, np.array(v_i[:2]) - np.array(v_j[:2]))) / d) if d > 0 else 0.0
                        d_safe = params.d0 + v_cl * params.tau_r + v_cl**2 / (2.0 * params.a_eff)
                        pred_conflict = cpa.d_cpa < d_safe
                    elif variant == "P1_CV":
                        pred = cv_mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
                        pred_conflict = pred.rho_min_pred < params.rho_pred_threshold
                        margin_errors.append(abs(pred.rho_min_pred - actual))
                    elif variant == "P2_CV_U":
                        pred = cv_mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
                        pred_conflict = pred.rho_min_pred < params.rho_pred_threshold
                    elif variant == "P3_CA_U":
                        pred = ca_mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
                        pred_conflict = pred.rho_min_pred < params.rho_pred_threshold
                        margin_errors.append(abs(pred.rho_min_pred - actual))
                    elif variant == "P4_ENSEMBLE":
                        pred_cv = cv_mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
                        pred_ca = ca_mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
                        rho_min = min(pred_cv.rho_min_pred, pred_ca.rho_min_pred)
                        pred_conflict = rho_min < params.rho_pred_threshold
                    else:
                        raise ValueError(variant)

                    actual_conflict = actual <= 0.0
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
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_margin_error": float(np.mean(margin_errors)) if margin_errors else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-seeds", type=int, default=10)
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    trajs = collect_trajectories(scenario, args.checkpoint, args.eval_seeds)
    params = RuntimeAssuranceParams()

    results = {}
    q_map = {"P0_CPA": 0.0, "P1_CV": 0.0, "P2_CV_U": 0.01, "P3_CA_U": 0.01, "P4_ENSEMBLE": 0.01}
    for variant in ["P0_CPA", "P1_CV", "P2_CV_U", "P3_CA_U", "P4_ENSEMBLE"]:
        results[variant] = evaluate_variant(trajs, scenario, params, variant, q_map[variant])
        r = results[variant]
        print(
            f"{variant}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
            f"F1={r['f1']:.3f} margin_err={r['mean_margin_error']}"
        )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
