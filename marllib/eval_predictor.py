"""Evaluate the predictive monitor lead time and error (Phase 3)."""

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
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.ra.predictor import predicted_distance
from swarm.safety import DroneSnapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-seeds", type=int, default=20)
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=False))
    params = RuntimeAssuranceParams(d0=0.5, tau_r=0.3, a_eff=scenario.accel_limit, alpha=1.0, v_max=1.5)
    ra = RuntimeAssurance(params, v_max=1.5, perception_sigma=0.0)

    lead_times: list[int] = []
    pred_errors: list[float] = []
    false_positives = 0
    total_proactive = 0

    for seed in range(args.eval_seeds):
        obs, _ = env.reset(seed=seed)
        history: list[np.ndarray] = [env.positions.copy()]
        proactive_steps: list[int] = []
        intervention_steps: list[int] = []

        for step in range(200):
            nominal = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
            snapshots = {
                i: DroneSnapshot(
                    drone_id=i,
                    position=(float(env.positions[i, 0]), float(env.positions[i, 1]), 0.0),
                    velocity=(float(env.velocities[i, 0]), float(env.velocities[i, 1]), 0.0),
                )
                for i in env.agent_ids
            }
            results = ra.filter(snapshots, nominal, t=step * scenario.dt)
            if any(r.proactive for r in results.values()):
                proactive_steps.append(step)
            if any(r.intervened for r in results.values()):
                intervention_steps.append(step)
            actions = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
            obs, _, terminated, _, infos = env.step(actions)
            history.append(env.positions.copy())
            if any(i["collided"] for i in infos.values()) or all(i["reached"] for i in infos.values()):
                break
            if terminated[0]:
                break

        if proactive_steps and intervention_steps:
            first_proactive = proactive_steps[0]
            first_intervention = intervention_steps[0]
            lead_times.append(first_intervention - first_proactive)

        # Prediction error at each step: predicted min distance over horizon vs
        # the actual min distance seen in the recorded future.
        horizon_steps = int(round(params.prediction_horizon / scenario.dt))
        for step in range(min(len(history) - 1, 200)):
            p_i = history[step][0]
            p_j = history[step][1] if scenario.num_agents > 1 else None
            if p_j is None:
                continue
            v_i = np.zeros(2) if step == 0 else (history[step][0] - history[step - 1][0]) / scenario.dt
            v_j = np.zeros(2) if step == 0 else (history[step][1] - history[step - 1][1]) / scenario.dt
            p_hat = predicted_distance(
                (float(p_i[0]), float(p_i[1]), 0.0),
                (float(p_j[0]), float(p_j[1]), 0.0),
                (float(v_i[0]), float(v_i[1]), 0.0),
                (float(v_j[0]), float(v_j[1]), 0.0),
                params.prediction_horizon,
            )
            future = history[step : step + horizon_steps + 1]
            actual_min = min(float(np.linalg.norm(h[0] - h[1])) for h in future)
            pred_errors.append(abs(p_hat - actual_min))

        # A proactive trigger with no intervention in that episode is a false positive.
        if proactive_steps and not intervention_steps:
            false_positives += 1
        if proactive_steps:
            total_proactive += 1

    result = {
        "scenario": scenario.name,
        "eval_seeds": args.eval_seeds,
        "mean_lead_time_steps": float(np.mean(lead_times)) if lead_times else None,
        "mean_prediction_error_m": float(np.mean(pred_errors)) if pred_errors else None,
        "proactive_episodes": total_proactive,
        "false_positive_episodes": false_positives,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
