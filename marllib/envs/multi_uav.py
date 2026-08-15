"""2D fixed-altitude multi-UAV environment.

The environment follows the plan.md section 5 kinematics:

    v_{t+1} = v_t + clip(v_cmd - v_t, -a_max*dt, a_max*dt)
    p_{t+1} = p_t + v_{t+1} * dt

It exposes a PettingZoo-style parallel API using numpy arrays and dicts keyed
by agent id.  No gymnasium/pettingzoo dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marllib.config import RewardConfig, ScenarioConfig
from marllib.reward import per_agent_reward


@dataclass
class StepResult:
    observations: dict[int, np.ndarray]
    rewards: dict[int, float]
    terminateds: dict[int, bool]
    truncateds: dict[int, bool]
    infos: dict[int, dict[str, Any]]


class MultiUAVEnv:
    """Parallel multi-UAV velocity-command environment."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        reward: RewardConfig | None = None,
        max_steps: int = 200,
    ) -> None:
        self.scenario = scenario
        self.reward_config = reward or RewardConfig()
        self.max_steps = max_steps
        self.num_agents = scenario.num_agents
        self.agent_ids = list(range(scenario.num_agents))
        self.obs_dim = 4 + 5 * scenario.max_neighbors
        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        n = self.num_agents
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.velocities = np.zeros((n, 2), dtype=np.float64)
        self.goals = np.zeros((n, 2), dtype=np.float64)
        self.prev_goal_distances = np.zeros(n, dtype=np.float64)
        self.prev_actions = np.zeros((n, 2), dtype=np.float64)
        self.reached_flags = np.zeros(n, dtype=bool)
        self.step_count = 0
        self._speed_limit = self.scenario.speed_limit
        self._accel_limit = self.scenario.accel_limit

    @property
    def speed_limit(self) -> float:
        return self._speed_limit

    @property
    def accel_limit(self) -> float:
        return self._accel_limit

    def _sample_positions(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_agents
        cfg = self.scenario
        x0, x1, y0, y1 = cfg.arena
        if cfg.random_start_goal:
            starts = self._rng.uniform((x0 + 0.5, y0 + 0.5), (x1 - 0.5, y1 - 0.5), (n, 2))
            goals = self._rng.uniform((x0 + 0.5, y0 + 0.5), (x1 - 0.5, y1 - 0.5), (n, 2))
        else:
            starts = np.asarray(cfg.starts, dtype=np.float64)
            goals = np.asarray(cfg.goals, dtype=np.float64)
            if cfg.start_noise > 0:
                starts = starts + self._rng.normal(0.0, cfg.start_noise, starts.shape)
            if cfg.goal_noise > 0:
                goals = goals + self._rng.normal(0.0, cfg.goal_noise, goals.shape)
        starts = np.clip(starts, (x0, y0), (x1, y1))
        goals = np.clip(goals, (x0, y0), (x1, y1))
        return starts, goals

    def reset(self, seed: int | None = None) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.positions, self.goals = self._sample_positions()
        self.prev_goal_distances = np.linalg.norm(self.goals - self.positions, axis=1)

        if self.scenario.speed_limit_range is not None:
            lo, hi = self.scenario.speed_limit_range
            self._speed_limit = self._rng.uniform(lo, hi)
        if self.scenario.accel_limit_range is not None:
            lo, hi = self.scenario.accel_limit_range
            self._accel_limit = self._rng.uniform(lo, hi)

        observations = {i: self.observation(i) for i in self.agent_ids}
        infos = {i: {"goal": self.goals[i].copy()} for i in self.agent_ids}
        return observations, infos

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> tuple[dict[int, np.ndarray], dict[int, float], dict[int, bool], dict[int, bool], dict[int, dict[str, Any]]]:
        cfg = self.scenario
        dt = cfg.dt
        n = self.num_agents

        action_matrix = np.stack([np.asarray(actions[i], dtype=np.float64) for i in self.agent_ids])
        # Velocity command is clipped to the speed limit.
        action_matrix = np.clip(
            action_matrix,
            -self._speed_limit,
            self._speed_limit,
        )

        # Acceleration-limited velocity update.
        velocity_delta = np.clip(
            action_matrix - self.velocities,
            -self._accel_limit * dt,
            self._accel_limit * dt,
        )
        new_velocities = self.velocities + velocity_delta
        new_positions = self.positions + new_velocities * dt

        x0, x1, y0, y1 = cfg.arena
        new_positions = np.clip(new_positions, (x0, y0), (x1, y1))

        # Pairwise distances for collision/near-miss.
        collision = np.zeros(n, dtype=bool)
        near_miss_margin = np.zeros(n, dtype=np.float64)
        if n > 1:
            delta = new_positions[:, None, :] - new_positions[None, :, :]
            dist = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(dist, np.inf)
            collision = (dist < cfg.collision_radius).any(axis=1)
            near = np.clip(cfg.near_radius - dist, 0.0, None)
            near_miss_margin = near.sum(axis=1)

        goal_distances = np.linalg.norm(self.goals - new_positions, axis=1)
        reached = goal_distances < cfg.goal_epsilon
        newly_reached = reached & ~self.reached_flags

        action_smoothness = np.sum((action_matrix - self.prev_actions) ** 2, axis=1)

        rewards: dict[int, float] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            rewards[agent_id] = per_agent_reward(
                config=self.reward_config,
                prev_goal_distance=float(self.prev_goal_distances[idx]),
                goal_distance=float(goal_distances[idx]),
                reached_goal=bool(newly_reached[idx]),
                collided=bool(collision[idx]),
                near_miss_margin=float(near_miss_margin[idx]),
                action_smoothness=float(action_smoothness[idx]),
            )

        # Commit the state transition.
        self.positions = new_positions
        self.velocities = new_velocities
        self.prev_goal_distances = goal_distances
        self.prev_actions = action_matrix
        self.reached_flags |= reached
        self.step_count += 1

        any_collision = bool(collision.any())
        all_reached = bool(reached.all())
        truncated = self.step_count >= self.max_steps
        terminated = any_collision or all_reached

        terminateds = {i: terminated for i in self.agent_ids}
        truncateds = {i: (truncated and not terminated) for i in self.agent_ids}
        observations = {i: self.observation(i) for i in self.agent_ids}
        infos: dict[int, dict[str, Any]] = {}
        for idx, agent_id in enumerate(self.agent_ids):
            infos[agent_id] = {
                "collided": bool(collision[idx]),
                "reached": bool(reached[idx]),
                "goal": self.goals[idx].copy(),
                "position": self.positions[idx].copy(),
            }

        return observations, rewards, terminateds, truncateds, infos

    def observation(self, agent_id: int) -> np.ndarray:
        """Flattened relative observation for one agent."""
        cfg = self.scenario
        idx = self.agent_ids.index(agent_id)
        self_vel = self.velocities[idx]
        goal_rel = self.goals[idx] - self.positions[idx]

        obs = [self_vel[0], self_vel[1], goal_rel[0], goal_rel[1]]

        neighbors: list[np.ndarray] = []
        if self.num_agents > 1:
            other = [j for j in self.agent_ids if j != agent_id]
            rel_pos = self.positions[other] - self.positions[idx]
            rel_vel = self.velocities[other] - self.velocities[idx]
            dist = np.linalg.norm(rel_pos, axis=1)
            order = np.argsort(dist)
            for j in order[: cfg.max_neighbors]:
                neighbors.append(
                    np.array(
                        [
                            rel_pos[j, 0],
                            rel_pos[j, 1],
                            rel_vel[j, 0],
                            rel_vel[j, 1],
                            1.0,
                        ]
                    )
                )
        while len(neighbors) < cfg.max_neighbors:
            neighbors.append(np.zeros(5, dtype=np.float64))
        obs.extend(np.concatenate(neighbors).tolist())

        observation = np.asarray(obs, dtype=np.float32)
        if self.scenario.observation_noise > 0:
            observation = observation + self._rng.normal(0.0, self.scenario.observation_noise, observation.shape).astype(np.float32)
        return observation

    def global_state(self) -> np.ndarray:
        """Absolute state concatenation used by a centralized critic."""
        return np.concatenate(
            [
                self.positions.reshape(-1),
                self.velocities.reshape(-1),
                self.goals.reshape(-1),
            ]
        ).astype(np.float32)
