"""Evaluate all trained checkpoints in the Phase 1 sweep.

Loads every ``final.pt`` under an output root and runs a deterministic rollout
over a fixed evaluation seed set, reporting collision rate and completion rate
per (scenario, training seed).
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


def evaluate_checkpoint(scenario, checkpoint: Path, eval_seeds: int = 20) -> dict:
    env = MultiUAVEnv(scenario, max_steps=200)
    model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
    model.load_state_dict(torch.load(checkpoint, weights_only=False))

    reached = 0
    collisions = 0
    for seed in range(eval_seeds):
        obs, _ = env.reset(seed=seed)
        done_reached = False
        done_collision = False
        for _ in range(200):
            actions = {
                i: model.actor.deterministic(torch.from_numpy(o).float().unsqueeze(0)).squeeze(0).numpy()
                for i, o in obs.items()
            }
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
    return {
        "reached": reached,
        "collisions": collisions,
        "eval_seeds": eval_seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 1 sweep checkpoints")
    parser.add_argument("--output", default="/Volumes/Expansion/safedrones_marllib_vec")
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--training-seeds", default="1,2,3,4,5")
    args = parser.parse_args()

    output_root = Path(args.output)
    training_seeds = [int(s) for s in args.training_seeds.split(",")]
    curriculum = default_curriculum()
    summary: dict[str, dict] = {}

    for scenario in curriculum:
        scenario_stats = {"n": 0, "reached": 0, "collisions": 0, "per_seed": {}}
        for seed in training_seeds:
            checkpoint = output_root / scenario.name / f"seed{seed}" / "checkpoints" / "final.pt"
            if not checkpoint.exists():
                continue
            result = evaluate_checkpoint(scenario, checkpoint, args.eval_seeds)
            scenario_stats["n"] += 1
            scenario_stats["reached"] += result["reached"]
            scenario_stats["collisions"] += result["collisions"]
            scenario_stats["per_seed"][str(seed)] = result
        if scenario_stats["n"]:
            scenario_stats["completion_rate"] = scenario_stats["reached"] / (scenario_stats["n"] * args.eval_seeds)
            scenario_stats["collision_rate"] = scenario_stats["collisions"] / (scenario_stats["n"] * args.eval_seeds)
        summary[scenario.name] = scenario_stats

    out = output_root / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
