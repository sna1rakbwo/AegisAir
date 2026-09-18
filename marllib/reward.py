"""Pure reward computation for the lightweight MARL environment.

This module has no environment or numpy dependency so the reward arithmetic can
be unit-tested independently.
"""

from __future__ import annotations

from marllib.config import RewardConfig


def per_agent_reward(
    *,
    config: RewardConfig,
    prev_goal_distance: float,
    goal_distance: float,
    reached_goal: bool,
    collided: bool,
    near_miss_margin: float,
    action_smoothness: float,
) -> float:
    """Compute one agent's scalar reward for a step.

    ``near_miss_margin`` is ``max(0, near_radius - d_ij)`` summed over the
    agent's neighbors; ``action_smoothness`` is ``||a_t - a_{t-1}||^2``.
    """

    reward = -config.time_penalty
    if reached_goal:
        reward += config.goal_reward
    reward += config.progress_gain * (prev_goal_distance - goal_distance)
    if collided:
        reward -= config.collision_penalty
    reward -= config.near_gain * near_miss_margin**2
    reward -= config.smooth_gain * action_smoothness
    return reward
