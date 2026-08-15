"""Evaluate a trained MARL policy with and without Runtime Assurance.

This is the Phase 2 acceptance probe: the same imperfect MARL checkpoint is
rolled out twice, once directly and once through the CBF safety filter, so the
collision-rate reduction is attributable to Runtime Assurance.
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
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.safety import DroneSnapshot


def rollout(scenario, model, use_ra: bool, ra: RuntimeAssurance | None, eval_seeds: int) -> dict:
    env = MultiUAVEnv(scenario, max_steps=200)
    reached = 0
    collisions = 0
    interventions = 0
    for seed in range(eval_seeds):
        obs, _ = env.reset(seed=seed)
        done_reached = False
        done_collision = False
        for step in range(200):
            nominal = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
            actions = nominal
            if use_ra:
                snapshots = {
                    i: DroneSnapshot(
                        drone_id=i,
                        position=(float(env.positions[i, 0]), float(env.positions[i, 1]), 0.0),
                        velocity=(float(env.velocities[i, 0]), float(env.velocities[i, 1]), 0.0),
                    )
                    for i in env.agent_ids
                }
                results = ra.filter(snapshots, nominal, t=step * scenario.dt)
                actions = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
                interventions += sum(1 for i in env.agent_ids if results[i].intervened)
            obs, _, terminated, _, infos = env.step(actions)
            if any(i["collided"] for i in infos.values()):
                done_collision = True
                break
            if all(i["reached"] for i in infos.values()):
                done_reached = True
                break
            if terminated[0]:
                break
        reached += int(done_reached)
        collisions += int(done_collision)
    total = eval_seeds
    return {
        "eval_seeds": total,
        "completion_rate": reached / total,
        "collision_rate": collisions / total,
        "interventions": interventions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--v-max", type=float, default=1.5)
    parser.add_argument("--perception-sigma", type=float, default=0.0)
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=False))

    params = RuntimeAssuranceParams(
        d0=0.5,
        tau_r=0.3,
        a_eff=scenario.accel_limit,
        alpha=1.0,
        v_max=args.v_max,
    )
    ra = RuntimeAssurance(params, v_max=args.v_max, perception_sigma=args.perception_sigma)

    baseline = rollout(scenario, model, use_ra=False, ra=None, eval_seeds=args.eval_seeds)
    filtered = rollout(scenario, model, use_ra=True, ra=ra, eval_seeds=args.eval_seeds)
    result = {"scenario": scenario.name, "baseline": baseline, "with_ra": filtered}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
