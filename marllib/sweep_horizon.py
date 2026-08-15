"""Final Phase 3 horizon-only sweep (validation seeds 10-14).

Frozen config: Filtered CA, Q=0, rho_th=0.  Only the prediction horizon H is
varied.  Reports offline precision/recall/F1 + recovery feasibility, and online
lead time + false-positive episode rate.
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
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.predictor import PredictiveMonitor
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.safety import DroneSnapshot


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


def offline_metrics(trajs, scenario, horizon):
    params = RuntimeAssuranceParams(prediction_horizon=horizon)
    dt = scenario.dt
    tp = fp = fn = tn = 0
    conflicts = []
    for positions, velocities in trajs:
        mon = PredictiveMonitor(use_ca=True, q_pred=0.0)
        for t in range(len(positions) - 1):
            for i in range(scenario.num_agents):
                for j in range(i + 1, scenario.num_agents):
                    p_i = (float(positions[t][i, 0]), float(positions[t][i, 1]), 0.0)
                    p_j = (float(positions[t][j, 0]), float(positions[t][j, 1]), 0.0)
                    v_i = (float(velocities[t][i, 0]), float(velocities[t][i, 1]), 0.0)
                    v_j = (float(velocities[t][j, 0]), float(velocities[t][j, 1]), 0.0)
                    mon.update_acceleration(i, v_i, dt)
                    mon.update_acceleration(j, v_j, dt)
                    pred = mon.predict(agent_i=i, agent_j=j, p_i=p_i, p_j=p_j, v_i=v_i, v_j=v_j, sigma_i=0.0, sigma_j=0.0, aoi=0.0, params=params, horizon=horizon)
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
                    pc = pred.rho_min_pred < 0.0
                    ac = best <= 0.0
                    if pc and ac:
                        tp += 1
                    elif pc and not ac:
                        fp += 1
                    elif not pc and ac:
                        fn += 1
                    else:
                        tn += 1
                    if pc:
                        conflicts.append(pred.ttsb)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    r_feasible = {}
    for latency in [0.4, 0.6]:
        feasible = sum(1 for t in conflicts if t is not None and t > latency)
        r_feasible[str(latency)] = feasible / len(conflicts) if conflicts else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "r_feasible": r_feasible}


def online_metrics(scenario, checkpoint, seeds, horizon):
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(checkpoint, weights_only=False))
    params = RuntimeAssuranceParams(prediction_horizon=horizon)
    ra = RuntimeAssurance(params, v_max=1.5, perception_sigma=0.0)
    leads = []
    fp_episodes = 0
    pro_episodes = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        pro = []
        intv = []
        for step in range(200):
            nominal = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
            snapshots = {
                i: DroneSnapshot(drone_id=i, position=(float(env.positions[i, 0]), float(env.positions[i, 1]), 0.0), velocity=(float(env.velocities[i, 0]), float(env.velocities[i, 1]), 0.0))
                for i in env.agent_ids
            }
            results = ra.filter(snapshots, nominal, t=step * scenario.dt)
            if any(r.proactive for r in results.values()):
                pro.append(step)
            if any(r.intervened for r in results.values()):
                intv.append(step)
            actions = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
            obs, _, terminated, _, infos = env.step(actions)
            if any(i["collided"] for i in infos.values()) or all(i["reached"] for i in infos.values()):
                break
        if pro and intv:
            leads.append(intv[0] - pro[0])
        if pro and not intv:
            fp_episodes += 1
        if pro:
            pro_episodes += 1
    return {
        "lead_time_s": float(np.mean(leads) * scenario.dt) if leads else None,
        "fp_episode_rate": fp_episodes / pro_episodes if pro_episodes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-seeds", default="10,11,12,13,14")
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    seeds = [int(s) for s in args.val_seeds.split(",")]
    trajs = collect(scenario, args.checkpoint, seeds)

    result = {}
    for horizon in [1.0, 1.25, 1.5, 2.0]:
        off = offline_metrics(trajs, scenario, horizon)
        on = online_metrics(scenario, args.checkpoint, seeds, horizon)
        result[str(horizon)] = {**off, **on}
        print(f"H={horizon}: precision={off['precision']:.3f} recall={off['recall']:.3f} F1={off['f1']:.3f} "
              f"lead={on['lead_time_s']} fp_ep={on['fp_episode_rate']:.3f} "
              f"R@0.4={off['r_feasible']['0.4']:.3f} R@0.6={off['r_feasible']['0.6']:.3f}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
