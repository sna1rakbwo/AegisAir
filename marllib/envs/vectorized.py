"""Batched (vectorized) multi-UAV environment for fast rollout collection.

This mirrors ``MultiUAVEnv`` but carries an extra leading batch dimension so
``B`` independent episodes can be stepped with one numpy call, removing the
Python-level per-environment loop that dominates single-env training.
"""

from __future__ import annotations

import numpy as np

from marllib.config import RewardConfig, ScenarioConfig


class VectorizedMultiUAVEnv:
    def __init__(
        self,
        scenario: ScenarioConfig,
        num_envs: int,
        reward: RewardConfig | None = None,
        max_steps: int = 200,
    ) -> None:
        self.scenario = scenario
        self.reward_config = reward or RewardConfig()
        self.num_envs = num_envs
        self.num_agents = scenario.num_agents
        self.max_steps = max_steps
        self.obs_dim = 4 + 5 * scenario.max_neighbors
        self.agent_ids = list(range(scenario.num_agents))
        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        b, n = self.num_envs, self.num_agents
        self.positions = np.zeros((b, n, 2), dtype=np.float64)
        self.velocities = np.zeros((b, n, 2), dtype=np.float64)
        self.goals = np.zeros((b, n, 2), dtype=np.float64)
        self.prev_goal_distances = np.zeros((b, n), dtype=np.float64)
        self.prev_actions = np.zeros((b, n, 2), dtype=np.float64)
        self.reached_flags = np.zeros((b, n), dtype=bool)
        self.step_counts = np.zeros(b, dtype=np.int64)
        self.speed_limits = np.full(b, self.scenario.speed_limit, dtype=np.float64)
        self.accel_limits = np.full(b, self.scenario.accel_limit, dtype=np.float64)

    def _sample_positions_batch(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_agents
        cfg = self.scenario
        x0, x1, y0, y1 = cfg.arena
        if cfg.random_start_goal:
            starts = self._rng.uniform((x0 + 0.5, y0 + 0.5), (x1 - 0.5, y1 - 0.5), (count, n, 2))
            goals = self._rng.uniform((x0 + 0.5, y0 + 0.5), (x1 - 0.5, y1 - 0.5), (count, n, 2))
        else:
            starts = np.broadcast_to(np.asarray(cfg.starts, dtype=np.float64), (count, n, 2)).copy()
            goals = np.broadcast_to(np.asarray(cfg.goals, dtype=np.float64), (count, n, 2)).copy()
            if cfg.start_noise > 0:
                starts += self._rng.normal(0.0, cfg.start_noise, starts.shape)
            if cfg.goal_noise > 0:
                goals += self._rng.normal(0.0, cfg.goal_noise, goals.shape)
        starts = np.clip(starts, (x0, y0), (x1, y1))
        goals = np.clip(goals, (x0, y0), (x1, y1))
        return starts, goals

    def _sample_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._sample_positions_batch(self.num_envs)

    def reset_done(self, done_mask: np.ndarray) -> None:
        idx = np.where(done_mask)[0]
        if len(idx) == 0:
            return
        starts, goals = self._sample_positions_batch(len(idx))
        self.positions[idx] = starts
        self.goals[idx] = goals
        self.velocities[idx] = 0.0
        self.prev_actions[idx] = 0.0
        self.prev_goal_distances[idx] = np.linalg.norm(goals - starts, axis=-1)
        self.reached_flags[idx] = False
        self.step_counts[idx] = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        self.positions, self.goals = self._sample_positions()
        self.prev_goal_distances = np.linalg.norm(self.goals - self.positions, axis=-1)

        if self.scenario.speed_limit_range is not None:
            lo, hi = self.scenario.speed_limit_range
            self.speed_limits = self._rng.uniform(lo, hi, self.num_envs)
        if self.scenario.accel_limit_range is not None:
            lo, hi = self.scenario.accel_limit_range
            self.accel_limits = self._rng.uniform(lo, hi, self.num_envs)
        return self.observations()

    def observations(self) -> np.ndarray:
        b, n = self.num_envs, self.num_agents
        k = self.scenario.max_neighbors
        rel_pos = self.positions[:, None, :, :] - self.positions[:, :, None, :]
        rel_vel = self.velocities[:, None, :, :] - self.velocities[:, :, None, :]
        dist = np.linalg.norm(rel_pos, axis=-1)
        diag = np.arange(n)
        dist[:, diag, diag] = np.inf

        # Pad the neighbor axis to max_neighbors so fixed-size observation is
        # well-defined even when num_agents < max_neighbors.
        pad = max(0, k - n)
        if pad > 0:
            rel_pos = np.pad(rel_pos, ((0, 0), (0, 0), (0, pad), (0, 0)))
            rel_vel = np.pad(rel_vel, ((0, 0), (0, 0), (0, pad), (0, 0)))
            dist = np.pad(dist, ((0, 0), (0, 0), (0, pad)), constant_values=np.inf)

        order = np.argsort(dist, axis=-1)[:, :, :k]
        neighbor_dist = np.take_along_axis(dist, order, axis=-1)
        neighbor_rel_pos = np.take_along_axis(rel_pos, order[..., None], axis=2)
        neighbor_rel_vel = np.take_along_axis(rel_vel, order[..., None], axis=2)
        exists = (neighbor_dist < 1e9).astype(np.float64)[..., None]
        neighbor_rel_pos = neighbor_rel_pos * exists
        neighbor_rel_vel = neighbor_rel_vel * exists

        self_vel = self.velocities
        goal_rel = self.goals - self.positions
        neighbor_features = np.concatenate(
            [neighbor_rel_pos, neighbor_rel_vel, exists],
            axis=-1,
        )
        obs = np.concatenate(
            [
                self_vel,
                goal_rel,
                neighbor_features.reshape(b, n, -1),
            ],
            axis=-1,
        )
        if self.scenario.observation_noise > 0:
            obs = obs + self._rng.normal(0.0, self.scenario.observation_noise, obs.shape)
        return obs.astype(np.float32)

    def global_states(self) -> np.ndarray:
        b, n = self.num_envs, self.num_agents
        return np.concatenate(
            [
                self.positions.reshape(b, -1),
                self.velocities.reshape(b, -1),
                self.goals.reshape(b, -1),
            ],
            axis=-1,
        ).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        b, n = self.num_envs, self.num_agents
        cfg = self.scenario
        dt = cfg.dt

        actions = np.asarray(actions, dtype=np.float64)
        speed = self.speed_limits[:, None, None]
        accel = self.accel_limits[:, None, None]
        actions = np.clip(actions, -speed, speed)

        velocity_delta = np.clip(actions - self.velocities, -accel * dt, accel * dt)
        new_velocities = self.velocities + velocity_delta
        new_positions = self.positions + new_velocities * dt
        x0, x1, y0, y1 = cfg.arena
        new_positions = np.clip(new_positions, (x0, y0), (x1, y1))

        collision = np.zeros((b, n), dtype=bool)
        near_miss_margin = np.zeros((b, n), dtype=np.float64)
        if n > 1:
            delta = new_positions[:, :, None, :] - new_positions[:, None, :, :]
            dist = np.linalg.norm(delta, axis=-1)
            diag = np.arange(n)
            dist[:, diag, diag] = np.inf
            collision = (dist < cfg.collision_radius).any(axis=-1)
            near_miss_margin = np.clip(cfg.near_radius - dist, 0.0, None).sum(axis=-1)

        goal_distances = np.linalg.norm(self.goals - new_positions, axis=-1)
        reached = goal_distances < cfg.goal_epsilon
        newly_reached = reached & ~self.reached_flags
        action_smoothness = np.sum((actions - self.prev_actions) ** 2, axis=-1)

        cfg_r = self.reward_config
        rewards = np.full((b, n), -cfg_r.time_penalty, dtype=np.float64)
        rewards += cfg_r.goal_reward * newly_reached.astype(np.float64)
        rewards += cfg_r.progress_gain * (self.prev_goal_distances - goal_distances)
        rewards -= cfg_r.collision_penalty * collision.astype(np.float64)
        rewards -= cfg_r.near_gain * near_miss_margin**2
        rewards -= cfg_r.smooth_gain * action_smoothness

        self.positions = new_positions
        self.velocities = new_velocities
        self.prev_goal_distances = goal_distances
        self.prev_actions = actions
        self.reached_flags |= reached
        self.step_counts += 1

        any_collision = collision.any(axis=-1)
        all_reached = reached.all(axis=-1)
        truncated = self.step_counts >= self.max_steps
        terminated = any_collision | all_reached

        obs = self.observations()
        infos = {
            "collided": collision,
            "reached": reached,
            "terminated": terminated,
            "truncated": truncated,
        }
        return obs, rewards, terminated, truncated, infos
