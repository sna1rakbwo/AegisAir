"""Phase 3 final evaluation: margin, threshold sweep, reliability, recovery."""

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
from swarm.ra.predictor import PredictiveMonitor


def collect(scenario, checkpoint, seeds):
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(checkpoint, weights_only=False))
    trajs = []
    for seed in seeds:
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


def evaluate(trajs, scenario, params):
    dt = scenario.dt
    horizon = params.prediction_horizon
    margin_errors = []
    ttsb_errors = []
    rows = []  # (rho_pred, actual_rho, ttsb, confidence)
    for positions, velocities in trajs:
        mon = PredictiveMonitor(use_ca=True, q_pred=params.q_pred)
        for t in range(len(positions) - 1):
            for i in range(scenario.num_agents):
                for j in range(i + 1, scenario.num_agents):
                    p_i = (float(positions[t][i, 0]), float(positions[t][i, 1]), 0.0)
                    p_j = (float(positions[t][j, 0]), float(positions[t][j, 1]), 0.0)
                    v_i = (float(velocities[t][i, 0]), float(velocities[t][i, 1]), 0.0)
                    v_j = (float(velocities[t][j, 0]), float(velocities[t][j, 1]), 0.0)
                    mon.update_acceleration(i, v_i, dt)
                    mon.update_acceleration(j, v_j, dt)
                    pred = mon.predict(
                        agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j,
                        sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon,
                    )
                    # actual future min margin + ttsb
                    steps = int(round(horizon / dt))
                    end = min(t + steps, len(positions) - 1)
                    best = float("inf")
                    actual_ttsb = None
                    for k in range(t + 1, end + 1):
                        r = positions[k][i] - positions[k][j]
                        d = float(np.linalg.norm(r))
                        v_cl = max(0.0, -float(np.dot(r, velocities[k][i] - velocities[k][j])) / d) if d > 0 else 0.0
                        d_safe = params.d0 + v_cl * params.tau_r + v_cl**2 / (2.0 * params.a_eff)
                        rho = (d - d_safe) / d_safe if d_safe > 0 else -1.0
                        if rho < best:
                            best = rho
                        if actual_ttsb is None and rho <= 0.0:
                            actual_ttsb = (k - t) * dt
                    margin_errors.append(abs(pred.rho_min_pred - best))
                    if pred.ttsb is not None and actual_ttsb is not None:
                        ttsb_errors.append(abs(pred.ttsb - actual_ttsb))
                    rows.append((pred.rho_min_pred, best, pred.ttsb, pred.confidence))
    return rows, margin_errors, ttsb_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-seeds", default="10,11,12,13,14")
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    trajs = collect(scenario, args.checkpoint, seeds)
    params = RuntimeAssuranceParams()
    rows, margin_errors, ttsb_errors = evaluate(trajs, scenario, params)

    margin_errors = np.array(margin_errors)
    ttsb_errors = np.array(ttsb_errors)
    margin = {
        "mae": float(np.mean(margin_errors)),
        "rmse": float(np.sqrt(np.mean(margin_errors**2))),
        "median": float(np.median(margin_errors)),
        "p90": float(np.quantile(margin_errors, 0.90)),
        "p95": float(np.quantile(margin_errors, 0.95)),
    }

    # Threshold sweep.
    sweep = {}
    for th in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        tp = fp = fn = tn = 0
        for rho_pred, actual, _, _ in rows:
            pc = rho_pred < th
            ac = actual <= 0.0
            if pc and ac:
                tp += 1
            elif pc and not ac:
                fp += 1
            elif not pc and ac:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        sweep[str(th)] = {
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "fnr": fnr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # Reliability bins.
    reliability = []
    for lo in [0.0, 0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2
        bin_rows = [(r, a, t, c) for r, a, t, c in rows if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if not bin_rows:
            continue
        errs = [abs(r - a) for r, a, t, c in bin_rows]
        reliability.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(bin_rows),
            "mean_confidence": float(np.mean([c for _, _, _, c in bin_rows])),
            "margin_mae": float(np.mean(errs)),
        })

    # Recovery feasibility vs latency.
    feasibility = []
    conflicts = [r for r in rows if r[0] < params.rho_pred_threshold]
    for latency in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        feasible = sum(1 for _, _, ttsb, _ in conflicts if ttsb is not None and ttsb > latency)
        feasibility.append({
            "latency": latency,
            "r_feasible": feasible / len(conflicts) if conflicts else 0.0,
        })

    result = {
        "scenario": scenario.name,
        "margin": margin,
        "ttsb_mae": float(np.mean(ttsb_errors)) if len(ttsb_errors) else None,
        "threshold_sweep": sweep,
        "reliability": reliability,
        "recovery_feasibility": feasibility,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
