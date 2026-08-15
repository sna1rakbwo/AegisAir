"""Phase 3 final: prediction-error calibration (Tasks A-D).

Measures CV and filtered-CA prediction error as a function of horizon on a
calibration seed set, then fits an empirical quantile bound for the prediction
margin ``M_pred(tau)``.
"""

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
from swarm.ra.predictor import PredictiveMonitor


HORIZONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


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


def radial_errors(trajs, scenario, use_ca: bool):
    dt = scenario.dt
    errors: dict[float, list[float]] = {h: [] for h in HORIZONS}
    for positions, velocities in trajs:
        mon = PredictiveMonitor(use_ca=use_ca)
        for t in range(len(positions) - 1):
            for i in range(scenario.num_agents):
                for j in range(i + 1, scenario.num_agents):
                    p_i = (float(positions[t][i, 0]), float(positions[t][i, 1]), 0.0)
                    p_j = (float(positions[t][j, 0]), float(positions[t][j, 1]), 0.0)
                    v_i = (float(velocities[t][i, 0]), float(velocities[t][i, 1]), 0.0)
                    v_j = (float(velocities[t][j, 0]), float(velocities[t][j, 1]), 0.0)
                    mon.update_acceleration(i, v_i, dt)
                    mon.update_acceleration(j, v_j, dt)
                    a_i = mon._acc_filter(i).ema if use_ca else 0.0
                    a_j = mon._acc_filter(j).ema if use_ca else 0.0
                    for h in HORIZONS:
                        steps = int(round(h / dt))
                        if t + steps >= len(positions):
                            continue
                        # predicted relative position
                        pred_i = np.array(p_i[:2]) + np.array(v_i[:2]) * h + 0.5 * a_i * h * h * np.array([1.0, 0.0])
                        pred_j = np.array(p_j[:2]) + np.array(v_j[:2]) * h + 0.5 * a_j * h * h * np.array([1.0, 0.0])
                        pred_rel = pred_i - pred_j
                        # actual relative position
                        act_i = positions[t + steps][i]
                        act_j = positions[t + steps][j]
                        act_rel = act_i - act_j
                        e_ij = act_rel - pred_rel
                        norm = float(np.linalg.norm(pred_rel))
                        if norm > 0:
                            n_hat = pred_rel / norm
                            e_r = float(np.dot(n_hat, e_ij))
                            errors[h].append(e_r)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--out", default="/Volumes/Expansion/safedrones_marllib_vec/prediction_calibration.json")
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    seeds = [int(s) for s in args.calib_seeds.split(",")]
    trajs = collect(scenario, args.checkpoint, seeds)

    result = {"scenario": scenario.name, "horizons": {}}
    for use_ca, name in [(False, "CV"), (True, "CA")]:
        errors = radial_errors(trajs, scenario, use_ca)
        for h in HORIZONS:
            vals = np.array(errors[h])
            abs_vals = np.abs(vals)
            result["horizons"].setdefault(h, {})
            result["horizons"][h][name] = {
                "rmse": float(np.sqrt(np.mean(vals**2))) if len(vals) else None,
                "q90": float(np.quantile(abs_vals, 0.90)) if len(vals) else None,
                "q95": float(np.quantile(abs_vals, 0.95)) if len(vals) else None,
                "q99": float(np.quantile(abs_vals, 0.99)) if len(vals) else None,
                "n": int(len(vals)),
            }

    # Empirical quantile curve for the CA model (the recommended M_pred).
    quantile_curve = {h: result["horizons"][h]["CA"]["q95"] for h in HORIZONS}
    result["quantile_curve_q95"] = quantile_curve

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
