"""MAPPO training entrypoint for the lightweight multi-UAV environment.

Results are written to the external drive by default (per project convention);
code and scripts stay in the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import RewardConfig, default_curriculum
from marllib.envs.multi_uav import MultiUAVEnv
from marllib.policies.mappo import MAPPO, MappoConfig


DEFAULT_OUTPUT = "/Volumes/Expansion/safedrones_marllib"


def _scenario_by_name(name: str, domain_randomize: bool = False):
    for scenario in default_curriculum(domain_randomize=domain_randomize):
        if scenario.name == name:
            return scenario
    raise SystemExit(f"unknown scenario: {name}")


def run_training(
    scenario_name: str,
    seed: int,
    total_steps: int,
    output_root: Path,
    domain_randomize: bool = False,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scenario = _scenario_by_name(scenario_name, domain_randomize)
    reward = RewardConfig()
    env = MultiUAVEnv(scenario, reward, max_steps=200)
    state_dim = scenario.num_agents * 6
    trainer = MAPPO(env.obs_dim, 2, state_dim, scenario.speed_limit, MappoConfig())

    suffix = "_dr" if domain_randomize else ""
    out_dir = output_root / f"{scenario_name}{suffix}" / f"seed{seed}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "num_agents": scenario.num_agents,
                "seed": seed,
                "total_steps": total_steps,
                "mappo": {
                    "lr": trainer.config.lr,
                    "gamma": trainer.config.gamma,
                    "gae_lambda": trainer.config.gae_lambda,
                    "clip_eps": trainer.config.clip_eps,
                    "epochs": trainer.config.epochs,
                },
            },
            indent=2,
        )
        + "\n"
    )

    observations, _ = env.reset(seed)
    global_state = env.global_state()
    step = 0
    episode = 0
    episode_return = 0.0
    episode_steps = 0
    episode_collided = False
    episode_all_reached = False
    recent_returns: list[float] = []

    def checkpoint_path(name: str) -> Path:
        return ckpt_dir / f"{name}.pt"

    def save_checkpoint(name: str) -> None:
        torch.save(trainer.state_dict(), checkpoint_path(name))

    def log_metrics(extra: dict) -> None:
        record = {
            "timestamp_ms": time.time_ns() // 1_000_000,
            "step": step,
            "episode": episode,
            "return": episode_return,
            "episode_steps": episode_steps,
            **extra,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    last_percent = -1
    while step < total_steps:
        actions, log_probs, values = trainer.act_with_values(observations, global_state)
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = {i: terminateds[i] or truncateds[i] for i in env.agent_ids}
        trainer.store(observations, actions, log_probs, rewards, dones, values, global_state)

        for i in env.agent_ids:
            episode_return += rewards[i]
        episode_collided = episode_collided or any(i.get("collided") for i in infos.values())
        episode_all_reached = episode_all_reached or all(i.get("reached") for i in infos.values())
        episode_steps += 1
        step += env.num_agents

        done_all = all(dones.values())
        if done_all:
            recent_returns.append(episode_return / env.num_agents)
            recent_returns = recent_returns[-50:]
            log_metrics(
                {
                    "avg_return": episode_return / env.num_agents,
                    "collided": episode_collided,
                    "all_reached": episode_all_reached,
                }
            )
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            episode_collided = False
            episode_all_reached = False
            observations, _ = env.reset()
        else:
            observations = next_observations
        global_state = env.global_state()

        if trainer.buffer.size >= trainer.config.rollout_steps:
            last_value = trainer.critic(
                torch.from_numpy(global_state).float().unsqueeze(0)
            ).detach().item()
            last_value = np.full(env.num_agents, last_value, dtype=np.float32)
            update_metrics = trainer.update(last_value)
            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            log_metrics({**update_metrics, "mean_recent_return": mean_return})
            print(
                f"step={step} episodes={episode} mean_return={mean_return:.3f} "
                f"policy_loss={update_metrics['policy_loss']:.4f}",
                flush=True,
            )

        percent = int(100 * step / total_steps)
        if percent >= 20 and percent != last_percent and percent in (20, 50, 80, 100):
            save_checkpoint(f"step_{step}")
            last_percent = percent

    save_checkpoint("final")
    print(f"done: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MAPPO on the lightweight multi-UAV env")
    parser.add_argument("--scenario", default="single_uav")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--domain-randomize", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_training(args.scenario, args.seed, args.total_steps, output_root, args.domain_randomize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
