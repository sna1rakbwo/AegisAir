"""Recovery-command execution semantics for the lightweight 2D environment.

Commands never reach the actuators directly.  They only produce high-level
intents (goal overrides, velocity scaling, aborts, priorities) that are then
passed back through the MARL pilot and, critically, through the Runtime
Assurance CBF filter, which retains the final safety veto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm.geometry import Vector3
from swarm.interfaces import RecoveryPlan


@dataclass
class ActiveRecovery:
    """Mutable recovery intents currently applied to one drone."""

    priority: str = "normal"
    velocity_scale: float = 1.0
    velocity_scale_expiry: float | None = None
    goal_override: Vector3 | None = None
    goal_override_expiry: float | None = None
    aborted: bool = False


@dataclass
class RecoveryOverrides:
    """Per-step intents that the rollout loop applies to the nominal pilot."""

    velocity_scale: dict[int, float] = field(default_factory=dict)
    goal_override: dict[int, Vector3] = field(default_factory=dict)
    aborted: dict[int, bool] = field(default_factory=dict)
    priority: dict[int, str] = field(default_factory=dict)


def default_active(drones: list[int]) -> dict[int, ActiveRecovery]:
    return {drone: ActiveRecovery() for drone in drones}


def tick(active: dict[int, ActiveRecovery], t: float) -> None:
    """Revert transient intents whose TTL has expired."""
    for state in active.values():
        if state.velocity_scale_expiry is not None and t >= state.velocity_scale_expiry:
            state.velocity_scale = 1.0
            state.velocity_scale_expiry = None
        if state.goal_override_expiry is not None and t >= state.goal_override_expiry:
            state.goal_override = None
            state.goal_override_expiry = None


def apply_plan(
    active: dict[int, ActiveRecovery],
    plan: RecoveryPlan,
    t: float,
    base_goals: dict[int, Vector3],
) -> None:
    """Mutate ``active`` in place from a validated recovery plan."""
    for command in plan.commands:
        state = active[command.drone]
        if command.action == "HOLD":
            state.velocity_scale = 0.0
            state.velocity_scale_expiry = t + command.ttl_sec
        elif command.action == "YIELD":
            state.velocity_scale = min(state.velocity_scale, 0.4)
            state.velocity_scale_expiry = t + command.ttl_sec
        elif command.action == "REROUTE":
            state.goal_override = command.waypoint
            state.goal_override_expiry = t + command.ttl_sec
        elif command.action == "REASSIGN":
            state.goal_override = command.waypoint
            state.goal_override_expiry = None
        elif command.action == "RETURN":
            state.goal_override = base_goals[command.drone]
            state.goal_override_expiry = None
        elif command.action == "CHANGE_PRIORITY":
            state.priority = command.priority
        elif command.action == "ABORT":
            state.aborted = True
            state.velocity_scale = 0.0


def overrides(active: dict[int, ActiveRecovery]) -> RecoveryOverrides:
    result = RecoveryOverrides()
    for drone, state in active.items():
        result.velocity_scale[drone] = state.velocity_scale
        result.priority[drone] = state.priority
        if state.goal_override is not None:
            result.goal_override[drone] = state.goal_override
        if state.aborted:
            result.aborted[drone] = True
    return result
