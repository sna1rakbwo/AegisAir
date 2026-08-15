"""Minimal MAPPO (parameter-sharing CTDE PPO) built on torch.

One shared Gaussian actor emits velocity commands; one centralized critic maps
the concatenated absolute state to a scalar value.  No external RL library is
used so the training loop and hyperparameters stay transparent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class GaussianActor(nn.Module):
    """Shared policy: obs -> bounded Gaussian velocity command."""

    def __init__(self, obs_dim: int, act_dim: int, speed_limit: float, hidden: int = 128) -> None:
        super().__init__()
        self.speed_limit = speed_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.speed_limit * torch.tanh(self.net(obs))
        std = self.log_std.exp().clamp(min=1e-3, max=1.0)
        return mean, std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -self.speed_limit, self.speed_limit)
        return action, log_prob, mean

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.clamp(mean, -self.speed_limit, self.speed_limit)


class CentralizedCritic(nn.Module):
    """State -> scalar value for one shared centralized critic."""

    def __init__(self, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class MappoConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 5
    minibatch_size: int = 256
    rollout_steps: int = 2048
    hidden: int = 128


class RolloutBuffer:
    """Stores one rollout batch for GAE and PPO updates."""

    def __init__(self) -> None:
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.states.clear()

    @property
    def size(self) -> int:
        return len(self.obs)


class MAPPO:
    """Parameter-sharing MAPPO trainer."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
        speed_limit: float,
        config: MappoConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or MappoConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.speed_limit = speed_limit
        self.device = device

        self.actor = GaussianActor(obs_dim, act_dim, speed_limit, self.config.hidden)
        self.critic = CentralizedCritic(state_dim, self.config.hidden)
        self.actor.to(device)
        self.critic.to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def act(self, observations: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], np.ndarray]:
        obs_tensor = torch.from_numpy(np.stack(list(observations.values()))).float()
        actions, _, _ = self.actor.sample(obs_tensor)
        return {i: actions[k].numpy() for k, i in enumerate(observations)}, obs_tensor.numpy()

    @torch.no_grad()
    def act_with_values(
        self, observations: dict[int, np.ndarray], global_state: np.ndarray
    ) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, float]]:
        obs_tensor = torch.from_numpy(np.stack(list(observations.values()))).float().to(self.device)
        state_tensor = torch.from_numpy(global_state).float().unsqueeze(0).to(self.device)
        actions, log_probs, _ = self.actor.sample(obs_tensor)
        values = self.critic(state_tensor)
        actions_dict = {i: actions[k].cpu().numpy() for k, i in enumerate(observations)}
        log_probs_dict = {i: float(log_probs[k]) for k, i in enumerate(observations)}
        team_value = float(values.item())
        values_dict = {i: team_value for i in observations}
        return actions_dict, log_probs_dict, values_dict

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, mean

    def store(
        self,
        obs: dict[int, np.ndarray],
        actions: dict[int, np.ndarray],
        log_probs: dict[int, float],
        rewards: dict[int, float],
        dones: dict[int, bool],
        values: dict[int, float],
        global_state: np.ndarray,
    ) -> None:
        n = len(obs)
        self.buffer.obs.append(torch.from_numpy(np.stack(list(obs.values()))).float())
        self.buffer.actions.append(torch.from_numpy(np.stack(list(actions.values()))).float())
        self.buffer.log_probs.append(torch.tensor(list(log_probs.values()), dtype=torch.float32))
        self.buffer.rewards.append(torch.tensor(list(rewards.values()), dtype=torch.float32))
        self.buffer.dones.append(torch.tensor([bool(dones[i]) for i in obs], dtype=torch.float32))
        self.buffer.values.append(torch.tensor(list(values.values()), dtype=torch.float32))
        self.buffer.states.append(torch.from_numpy(global_state).float().unsqueeze(0).expand(n, -1))

    def compute_returns(self, last_value: np.ndarray) -> torch.Tensor:
        rewards = torch.stack(self.buffer.rewards)
        dones = torch.stack(self.buffer.dones)
        values = torch.stack(self.buffer.values)
        next_values = torch.cat([values[1:], torch.from_numpy(last_value).float().unsqueeze(0)], dim=0)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1], dtype=torch.float32)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.config.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return returns, advantages

    def update(self, last_value: np.ndarray) -> dict[str, float]:
        returns, advantages = self.compute_returns(last_value)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        obs = torch.cat(self.buffer.obs, dim=0).to(self.device)
        actions = torch.cat(self.buffer.actions, dim=0).to(self.device)
        old_log_probs = torch.cat(self.buffer.log_probs, dim=0).to(self.device)
        states = torch.cat(self.buffer.states, dim=0).to(self.device)
        returns_flat = returns.reshape(-1)
        advantages_flat = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_flat = advantages_flat.reshape(-1)

        total_samples = obs.shape[0]
        indices = np.arange(total_samples)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.config.epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, self.config.minibatch_size):
                idx = indices[start : start + self.config.minibatch_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv = advantages_flat[idx]
                mb_returns = returns_flat[idx]
                mb_states = states[idx]

                new_log_prob, entropy, _ = self.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_prob - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.config.entropy_coef * entropy.mean()

                value_pred = self.critic(mb_states)
                value_loss = self.config.value_coef * nn.functional.mse_loss(value_pred, mb_returns)

                self.actor_optim.zero_grad()
                (policy_loss + entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.mean().detach())

        n_updates = max(1, self.config.epochs * max(1, total_samples // self.config.minibatch_size))
        for key in metrics:
            metrics[key] /= n_updates
        self.buffer.clear()
        return metrics

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
