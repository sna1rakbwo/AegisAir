"""Phase 1 lightweight MARL environment and reward tests."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from marllib.config import (
    RewardConfig,
    ScenarioConfig,
    head_on,
    perpendicular,
    single_uav,
    with_domain_randomization,
)
from marllib.envs.multi_uav import MultiUAVEnv
from marllib.envs.vectorized import VectorizedMultiUAVEnv
from marllib.policies.mappo import MAPPO, MappoPilot
from marllib.reward import per_agent_reward


def _full_speed_action(env: MultiUAVEnv) -> dict[int, np.ndarray]:
    return {i: np.array([1.0, 0.0], dtype=np.float64) for i in env.agent_ids}


class RewardTest(unittest.TestCase):
    def test_goal_reward_and_progress(self) -> None:
        config = RewardConfig()
        reward = per_agent_reward(
            config=config,
            prev_goal_distance=2.0,
            goal_distance=1.0,
            reached_goal=False,
            collided=False,
            near_miss_margin=0.0,
            action_smoothness=0.0,
        )
        self.assertAlmostEqual(reward, -0.02 + 0.5 * 1.0)

    def test_collision_penalty(self) -> None:
        reward = per_agent_reward(
            config=RewardConfig(),
            prev_goal_distance=1.0,
            goal_distance=1.0,
            reached_goal=False,
            collided=True,
            near_miss_margin=0.0,
            action_smoothness=0.0,
        )
        self.assertAlmostEqual(reward, -0.02 - 20.0)


class EnvironmentTest(unittest.TestCase):
    def test_observation_shape(self) -> None:
        env = MultiUAVEnv(single_uav())
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs[0].shape, (env.obs_dim,))
        self.assertEqual(env.obs_dim, 4 + 5 * 8)

    def test_single_uav_reaches_goal(self) -> None:
        env = MultiUAVEnv(single_uav(), max_steps=100)
        env.reset(seed=0)
        reached = False
        for _ in range(100):
            _, _, terminated, _, infos = env.step({0: np.array([1.0, 0.0], dtype=np.float64)})
            if infos[0]["reached"]:
                reached = True
                break
            if terminated[0]:
                break
        self.assertTrue(reached)

    def test_seed_reproducibility(self) -> None:
        env_a = MultiUAVEnv(head_on())
        env_b = MultiUAVEnv(head_on())
        obs_a, _ = env_a.reset(seed=7)
        obs_b, _ = env_b.reset(seed=7)
        for agent in obs_a:
            np.testing.assert_allclose(obs_a[agent], obs_b[agent])

    def test_head_on_collision_detected(self) -> None:
        # A true head-on scenario with no lateral offset collides head-on.
        scenario = ScenarioConfig(
            name="head_on_zero_offset",
            num_agents=2,
            starts=((-4.0, 0.0), (4.0, 0.0)),
            goals=((4.0, 0.0), (-4.0, 0.0)),
        )
        env = MultiUAVEnv(scenario, max_steps=50)
        env.reset(seed=0)
        # Drive both agents straight at each other.
        actions = {0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])}
        collided = False
        for _ in range(50):
            _, _, terminated, _, infos = env.step(actions)
            if any(i["collided"] for i in infos.values()):
                collided = True
                break
            if terminated[0]:
                break
        self.assertTrue(collided)

    def test_perpendicular_observation_has_two_neighbors_slot(self) -> None:
        env = MultiUAVEnv(perpendicular())
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs[0].shape, (env.obs_dim,))

    def test_domain_randomization_sets_ranges(self) -> None:
        scenario = with_domain_randomization(head_on())
        self.assertIsNotNone(scenario.speed_limit_range)
        self.assertIsNotNone(scenario.accel_limit_range)
        self.assertGreater(scenario.observation_noise, 0.0)

        env = MultiUAVEnv(scenario, max_steps=10)
        env.reset(seed=0)
        self.assertGreaterEqual(env.speed_limit, scenario.speed_limit_range[0])
        self.assertLessEqual(env.speed_limit, scenario.speed_limit_range[1])

    def test_vectorized_env_matches_single_env(self) -> None:
        single = MultiUAVEnv(head_on(), max_steps=50)
        vec = VectorizedMultiUAVEnv(head_on(), 1, max_steps=50)
        single.reset(seed=3)
        vec.reset(seed=3)
        np.testing.assert_allclose(single.positions, vec.positions[0])
        np.testing.assert_allclose(single.goals, vec.goals[0])

        s_obs = {i: single.observation(i) for i in single.agent_ids}
        v_obs = vec.observations()[0]
        for idx, agent in enumerate(single.agent_ids):
            np.testing.assert_allclose(s_obs[agent], v_obs[idx], rtol=1e-5, atol=1e-5)


class MappoPilotTest(unittest.TestCase):
    def test_observation_matches_env_and_actions_bounded(self) -> None:
        scenario = ScenarioConfig(
            name="t",
            num_agents=3,
            starts=((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
            goals=((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
            max_neighbors=2,
        )
        env = MultiUAVEnv(scenario)
        env.reset(seed=0)
        env.step({i: np.array([0.1, 0.2]) for i in env.agent_ids})

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = f"{tmp}/final.pt"
            model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
            torch.save(model.state_dict(), checkpoint)
            pilot = MappoPilot(
                checkpoint,
                obs_dim=env.obs_dim,
                num_agents=env.num_agents,
                speed_limit=scenario.speed_limit,
                max_neighbors=scenario.max_neighbors,
            )

            positions = {i: env.positions[i] for i in env.agent_ids}
            velocities = {i: env.velocities[i] for i in env.agent_ids}
            goals = {i: env.goals[i] for i in env.agent_ids}

            for i in env.agent_ids:
                np.testing.assert_allclose(
                    pilot._observation(i, env.agent_ids, positions, velocities, goals),
                    env.observation(i),
                    rtol=1e-6,
                    atol=1e-6,
                )

            actions = pilot.actions(positions, velocities, goals)
            for i in env.agent_ids:
                self.assertLessEqual(
                    float(np.linalg.norm(actions[i])),
                    scenario.speed_limit + 1e-6,
                )


if __name__ == "__main__":
    unittest.main()
