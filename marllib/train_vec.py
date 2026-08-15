"""Vectorized MAPPO training entrypoint with optional CUDA.

Uses ``VectorizedMultiUAVEnv`` for batched rollouts and the same MAPPO trainer,
but steps many environments per Python iteration.  Results go to the external
drive by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import RewardConfig, default_curriculum
from marllib.envs.vectorized import VectorizedMultiUAVEnv
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
    num_envs: int,
    output_root: Path,
    domain_randomize: bool,
    device: str,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scenario = _scenario_by_name(scenario_name, domain_randomize)
    reward = RewardConfig()
    env = VectorizedMultiUAVEnv(scenario, num_envs, reward, max_steps=200)
    state_dim = scenario.num_agents * 6
    trainer = MAPPO(env.obs_dim, 2, state_dim, scenario.speed_limit, MappoConfig(), device=device)

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
                "num_envs": num_envs,
                "device": device,
                "seed": seed,
                "total_steps": total_steps,
                "domain_randomize": domain_randomize,
            },
            indent=2,
        )
        + "\n"
    )

    b, n = num_envs, scenario.num_agents
    steps_per_update = max(1, trainer.config.rollout_steps // num_envs)
    obs = env.reset(seed)  # (b, n, obs_dim)
    global_step = 0
    step = 0
    episode_returns = np.zeros(b, dtype=np.float64)
    episode_steps = np.zeros(b, dtype=np.int64)
    recent_returns: list[float] = []

    def save(name: str) -> None:
        torch.save(trainer.state_dict(), ckpt_dir / f"{name}.pt")

    def log(extra: dict) -> None:
        record = {
            "timestamp_ms": time.time_ns() // 1_000_000,
            "step": step,
            **extra,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def act_batch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_flat = obs.reshape(b * n, -1)
        state = env.global_states()  # (b, state_dim)
        obs_t = torch.from_numpy(obs_flat).float().to(device)
        state_t = torch.from_numpy(state).float().to(device)
        actions, log_probs, _ = trainer.actor.sample(obs_t)
        values = trainer.critic(state_t)  # (b,)
        return (
            actions.detach().cpu().numpy().reshape(b, n, 2),
            log_probs.detach().cpu().numpy().reshape(b, n),
            values.detach().cpu().numpy()[:, None].repeat(n, axis=1),
        )

    last_percent = -1
    while step < total_steps:
        actions, log_probs, values = act_batch()
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        dones = terminated | truncated
        dones_agents = np.repeat(dones[:, None], n, axis=1).astype(np.float32)

        trainer.buffer.obs.append(torch.from_numpy(obs.reshape(b * n, -1)).float())
        trainer.buffer.actions.append(torch.from_numpy(actions.reshape(b * n, 2)).float())
        trainer.buffer.log_probs.append(torch.from_numpy(log_probs.reshape(b * n)).float())
        trainer.buffer.rewards.append(torch.from_numpy(rewards.reshape(b * n)).float())
        trainer.buffer.dones.append(torch.from_numpy(dones_agents.reshape(b * n)))
        trainer.buffer.values.append(torch.from_numpy(values.reshape(b * n)).float())
        trainer.buffer.states.append(
            torch.from_numpy(env.global_states()).float().unsqueeze(1).expand(b, n, state_dim).reshape(b * n, state_dim)
        )

        episode_returns += rewards.sum(axis=-1)
        episode_steps += 1
        if dones.any():
            for bi in np.where(dones)[0]:
                recent_returns.append(float(episode_returns[bi] / n))
                episode_returns[bi] = 0.0
                episode_steps[bi] = 0
            recent_returns = recent_returns[-200:]
            env.reset_done(dones)

        obs = next_obs
        step += num_envs
        global_step += num_envs

        if trainer.buffer.size >= steps_per_update:
            last_state = env.global_states()
            last_value = trainer.critic(torch.from_numpy(last_state).float().to(device)).detach().cpu().numpy()
            last_value = np.repeat(last_value, n)
            metrics = trainer.update(last_value)
            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            log({**metrics, "mean_recent_return": mean_return, "buffer_size": trainer.buffer.size})
            print(
                f"step={step} env_steps={global_step} mean_return={mean_return:.3f} "
                f"policy_loss={metrics['policy_loss']:.4f}",
                flush=True,
            )

        percent = int(100 * step / total_steps)
        if percent >= 20 and percent != last_percent and percent in (20, 50, 80, 100):
            save(f"step_{step}")
            last_percent = percent

    save("final")
    print(f"done: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorized MAPPO training")
    parser.add_argument("--scenario", default="head_on")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_training(
        args.scenario,
        args.seed,
        args.total_steps,
        args.num_envs,
        output_root,
        args.domain_randomize,
        device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
