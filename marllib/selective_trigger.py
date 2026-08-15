"""Selective proactive trigger experiments (event-level + persistence + CPA)."""

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


def build_signals(trajs, scenario, params):
    """Return per-pair sequences of (step, rho_pred, ttsb, d_cpa, actual_violation)."""
    dt = scenario.dt
    horizon = params.prediction_horizon
    horizon_steps = int(round(horizon / dt))
    sequences = {}
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
                    cpa = closest_point_of_approach(p_i, p_j, v_i, v_j, horizon)
                    # actual violation within horizon
                    end = min(t + horizon_steps, len(positions) - 1)
                    violation = False
                    for k in range(t + 1, end + 1):
                        r = positions[k][i] - positions[k][j]
                        d = float(np.linalg.norm(r))
                        v_cl = max(0.0, -float(np.dot(r, velocities[k][i] - velocities[k][j])) / d) if d > 0 else 0.0
                        d_safe = params.d0 + v_cl * params.tau_r + v_cl**2 / (2.0 * params.a_eff)
                        if d <= d_safe:
                            violation = True
                            break
                    sequences.setdefault((i, j), []).append((t, pred.rho_min_pred, pred.ttsb, cpa.d_cpa, violation))
    return sequences


def evaluate(sequences, persistence, cpa_th, b_min, dt):
    """Frame-level and event-level metrics for one trigger config."""
    tp = fp = fn = tn = 0
    event_tp = event_fp = event_fn = 0
    for pair, rows in sequences.items():
        # persistence: consecutive rho<0 count.
        streak = 0
        active = False
        gap = 0
        for step, rho, ttsb, d_cpa, violation in rows:
            base = rho < 0.0 and (ttsb is None or ttsb > b_min) and (cpa_th is None or d_cpa < cpa_th)
            streak = streak + 1 if base else 0
            trigger = streak >= persistence
            if trigger:
                if not active:
                    active = True
                    gap = 0
                    # new event
                    if violation:
                        event_tp += 1
                    else:
                        event_fp += 1
                gap = 0
            else:
                if active:
                    gap += 1
                    if gap > 2:
                        active = False
            if trigger and violation:
                tp += 1
            elif trigger and not violation:
                fp += 1
            elif not trigger and violation:
                fn += 1
            else:
                tn += 1
    frame = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
    frame["f1"] = 2 * frame["precision"] * frame["recall"] / (frame["precision"] + frame["recall"]) if frame["precision"] + frame["recall"] else 0.0
    event = {
        "precision": event_tp / (event_tp + event_fp) if event_tp + event_fp else 0.0,
        "events": event_tp + event_fp,
    }
    return frame, event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-seeds", default="15,16,17,18,19,20,21,22,23,24")
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    seeds = [int(s) for s in args.test_seeds.split(",")]
    trajs = collect(scenario, args.checkpoint, seeds)
    params = RuntimeAssuranceParams()
    sequences = build_signals(trajs, scenario, params)

    results = []
    configs = [
        (1, None, 0.0),
        (2, None, 0.0),
        (3, None, 0.0),
        (1, 2.0, 0.0),
        (1, 1.5, 0.0),
        (2, 1.5, 0.0),
        (2, 2.0, 0.0),
        (3, 2.0, 0.0),
    ]
    for persistence, cpa_th, b_min in configs:
        frame, event = evaluate(sequences, persistence, cpa_th, b_min, scenario.dt)
        results.append({"persistence": persistence, "cpa_th": cpa_th, "frame": frame, "event": event})
        print(
            f"K={persistence} cpa_th={cpa_th}: "
            f"frame_prec={frame['precision']:.3f} frame_rec={frame['recall']:.3f} | "
            f"event_prec={event['precision']:.3f} events={event['events']}"
        )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
