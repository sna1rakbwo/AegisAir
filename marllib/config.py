"""Scenario and reward configuration for the lightweight MARL environment.

All geometry is 2D fixed-altitude FLU: ``x`` forward, ``y`` left, in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights from plan.md section 6."""

    goal_reward: float = 10.0
    progress_gain: float = 0.5
    collision_penalty: float = 20.0
    near_gain: float = 2.0
    smooth_gain: float = 0.05
    time_penalty: float = 0.02


@dataclass(frozen=True)
class ScenarioConfig:
    """One curriculum scenario."""

    name: str
    num_agents: int
    starts: tuple[tuple[float, float], ...]
    goals: tuple[tuple[float, float], ...]
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)
    speed_limit: float = 1.5
    accel_limit: float = 2.0
    dt: float = 0.1
    collision_radius: float = 0.25
    near_radius: float = 0.8
    goal_epsilon: float = 0.5
    max_neighbors: int = 8
    random_start_goal: bool = False
    # Domain randomization overrides applied by the env at reset.
    start_noise: float = 0.0
    goal_noise: float = 0.0
    speed_limit_range: tuple[float, float] | None = None
    accel_limit_range: tuple[float, float] | None = None
    observation_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if len(self.starts) != self.num_agents or len(self.goals) != self.num_agents:
            raise ValueError("starts/goals must match num_agents")
        if self.collision_radius >= self.near_radius:
            raise ValueError("collision_radius must be smaller than near_radius")


def single_uav() -> ScenarioConfig:
    return ScenarioConfig(
        name="single_uav",
        num_agents=1,
        starts=((-4.0, 0.0),),
        goals=((4.0, 0.0),),
        start_noise=0.3,
        goal_noise=0.3,
    )


def head_on() -> ScenarioConfig:
    return ScenarioConfig(
        name="head_on",
        num_agents=2,
        starts=((-4.0, 0.5), (4.0, -0.5)),
        goals=((4.0, -0.5), (-4.0, 0.5)),
        start_noise=0.25,
        goal_noise=0.25,
    )


def perpendicular() -> ScenarioConfig:
    return ScenarioConfig(
        name="perpendicular",
        num_agents=2,
        starts=((-4.0, 0.5), (0.5, -4.0)),
        goals=((4.0, -0.5), (-0.5, 4.0)),
        start_noise=0.25,
        goal_noise=0.25,
    )


def diagonal() -> ScenarioConfig:
    return ScenarioConfig(
        name="diagonal",
        num_agents=2,
        starts=((-4.0, -4.0), (-4.0, 4.0)),
        goals=((4.0, 4.0), (4.0, -4.0)),
        start_noise=0.25,
        goal_noise=0.25,
    )


def randomized(num_agents: int) -> ScenarioConfig:
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    # Random starts/goals are sampled by the env from the arena at reset.
    starts = tuple((0.0, 0.0) for _ in range(num_agents))
    goals = tuple((0.0, 0.0) for _ in range(num_agents))
    return ScenarioConfig(
        name=f"randomized_{num_agents}",
        num_agents=num_agents,
        starts=starts,
        goals=goals,
        random_start_goal=True,
    )


def default_curriculum(domain_randomize: bool = False) -> list[ScenarioConfig]:
    """The Phase 1 training curriculum from plan.md section 7."""
    curriculum = [
        single_uav(),
        head_on(),
        perpendicular(),
        diagonal(),
        randomized(2),
        randomized(4),
        randomized(8),
    ]
    if domain_randomize:
        return [with_domain_randomization(s) for s in curriculum]
    return curriculum


def with_domain_randomization(scenario: ScenarioConfig) -> ScenarioConfig:
    """Enable the plan.md section 8 randomized dynamics/observation ranges."""
    return replace(
        scenario,
        speed_limit_range=(1.0, 2.0),
        accel_limit_range=(1.5, 3.0),
        observation_noise=0.05,
    )
