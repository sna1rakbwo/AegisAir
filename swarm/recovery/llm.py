"""Deterministic mission-recovery planners used by the paper experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from swarm.geometry import Vector3
from swarm.interfaces import RecoveryCommand, RecoveryPlan
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class RecoveryContext:
    """Everything the recovery client needs to produce a structured plan."""

    event: str
    agent_i: int
    agent_j: int
    current_margin: float
    predicted_min_margin: float | None
    margin_degradation: float | None
    intervention_count: int
    cause: str
    severity: str
    snapshots: dict[int, DroneSnapshot]
    current_goals: dict[int, Vector3]
    base_goals: dict[int, Vector3]
    priorities: dict[int, str]
    timestamp_ms: int
    mission_change: dict[str, Any] | None = None


@dataclass
class LLMRecoveryResult:
    """Normalised result from any recovery backend."""

    plan: RecoveryPlan | None
    raw: dict[str, Any] | None
    latency_s: float
    timeout: bool
    valid: bool
    errors: list[str]
    backend: str


class RecoveryLLMError(RuntimeError):
    """Raised when a local LLM backend cannot be constructed or used."""


RECOVERY_SYSTEM_PROMPT = (
    "You are the Semantic Mission Manager for a multi-UAV mission. "
    "When the current mission plan becomes invalid under safety, environment, or "
    "coordination changes, replan at the mission level. "
    "Return only one compact JSON object with fields 'action' and optional "
    "'agent', 'high', 'low'. Do not output waypoints, ttl, command_id, or "
    "schema fields; those are filled deterministically. "
    "Whitelisted actions: HOLD, YIELD, REROUTE, REASSIGN, "
    "CHANGE_PRIORITY, ABORT, RETURN. "
    "For HOLD/YIELD/REROUTE/ABORT/RETURN set 'agent' to the affected drone id. "
    "For REASSIGN when a drone failed, set 'agent' to the healthy drone that "
    "takes over the failed drone's task. HARD CONSTRAINT: never reassign a "
    "drone whose priority is 'critical' or 'safety' — such a drone must keep "
    "its own mission; choose a normal-priority healthy drone instead. "
    "For CHANGE_PRIORITY set 'high' and 'low' to drone ids. "
    "If mission_change.kind is 'fail_drone', respond with action REASSIGN. "
    "If mission_change.kind is 'block_corridor', respond with action REROUTE. "
    "If mission_change.kind is 'priority_change', respond with action CHANGE_PRIORITY. "
    "If cause is 'COORDINATION_DEGRADATION' (a symmetric deadlock with no "
    "goal progress), set the optional 'priority_order' to the right-of-way "
    "order according to mission semantics (urgent missions first), and choose "
    "action REROUTE or YIELD for the affected drones; never emit velocity or "
    "acceleration commands. "
    "Never modify d0, CBF constraints, safety thresholds, or actuator limits. "
    "Never output velocity, acceleration, or turn commands. "
    "Do not include reasoning or markdown; output JSON only."
)


def _context_payload(context: RecoveryContext) -> dict[str, Any]:
    return {
        "event": context.event,
        "agent_i": context.agent_i,
        "agent_j": context.agent_j,
        "current_margin": context.current_margin,
        "predicted_min_margin": context.predicted_min_margin,
        "margin_degradation": context.margin_degradation,
        "intervention_count": context.intervention_count,
        "cause": context.cause,
        "severity": context.severity,
        "priorities": context.priorities,
        "current_goals": context.current_goals,
        "positions": {
            i: list(snap.position) for i, snap in context.snapshots.items()
        },
        "mission_change": context.mission_change,
    }


class LLMRecoveryClient:
    """Protocol implemented by every semantic-recovery backend."""

    name: str = "base"

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        raise NotImplementedError


def _lateral_waypoint(
    actor_pos: Vector3,
    other_pos: Vector3,
    actor_goal: Vector3,
    offset_m: float = 2.0,
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
) -> Vector3:
    """A short detour perpendicular to the actor->other direction."""
    dx = other_pos[0] - actor_pos[0]
    dy = other_pos[1] - actor_pos[1]
    length = math.hypot(dx, dy)
    if length == 0:
        perp_x, perp_y = 0.0, 1.0
    else:
        perp_x, perp_y = -dy / length, dx / length
    # Prefer the side that keeps the detour roughly toward the goal.
    toward_goal_x = actor_goal[0] - actor_pos[0]
    toward_goal_y = actor_goal[1] - actor_pos[1]
    if perp_x * toward_goal_x + perp_y * toward_goal_y < 0:
        perp_x, perp_y = -perp_x, -perp_y
    x0, x1, y0, y1 = arena
    waypoint = (
        float(min(max(actor_pos[0] + perp_x * offset_m, x0), x1)),
        float(min(max(actor_pos[1] + perp_y * offset_m, y0), y1)),
        0.0,
    )
    return waypoint


class DeterministicRecoveryClient(LLMRecoveryClient):
    """Deterministic, rule-based stand-in for the local LLM.

    It emits the same whitelisted ``REROUTE`` mission change the local model
    would be asked to produce, so R0/R1/R2 differ only in the trigger policy
    rather than the backend.  ``plan_latency_s`` is optional and lets a smoke
    test exercise the timeout/fallback path without loading a model.
    """

    name = "deterministic"

    def __init__(
        self,
        *,
        arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0),
        plan_latency_s: float = 0.0,
    ) -> None:
        self.arena = arena
        self.plan_latency_s = plan_latency_s

    def _choose_actor(self, context: RecoveryContext) -> int:
        """Lower-priority drone yields; ties break toward the higher id."""
        pair = (context.agent_i, context.agent_j)
        order = sorted(
            pair,
            key=lambda drone: (
                0 if context.priorities.get(drone, "normal") == "normal" else 1,
                drone,
            ),
        )
        return order[0]

    def _build_plan(self, context: RecoveryContext) -> RecoveryPlan:
        if context.agent_i == context.agent_j:
            actor = context.agent_i
            command = RecoveryCommand(
                drone=actor,
                action="RETURN",
                waypoint=None,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"det-{context.timestamp_ms}-{actor}",
            )
            return RecoveryPlan(
                schema_version=1,
                intent_text=f"rejoin original mission for drone {actor}",
                commands=[command],
                constraints=None,
                rationale="deterministic return to base goal after mission invalidation",
                timestamp_ms=context.timestamp_ms,
            )

        actor = self._choose_actor(context)
        other = context.agent_i if actor == context.agent_j else context.agent_j
        waypoint = _lateral_waypoint(
            context.snapshots[actor].position,
            context.snapshots[other].position,
            context.base_goals[actor],
            arena=self.arena,
        )
        command_id = f"det-{context.timestamp_ms}-{actor}"
        command = RecoveryCommand(
            drone=actor,
            action="REROUTE",
            waypoint=waypoint,
            priority="normal",
            ttl_sec=1.0,
            command_id=command_id,
        )
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"recover pair {actor}-{other} after {context.event}",
            commands=[command],
            constraints=None,
            rationale=f"deterministic lateral detour for lower-priority drone {actor}",
            timestamp_ms=context.timestamp_ms,
        )

    def generate(self, context: RecoveryContext) -> LLMRecoveryResult:
        started = time.perf_counter()
        if self.plan_latency_s:
            time.sleep(self.plan_latency_s)
        plan = self._build_plan(context)
        return LLMRecoveryResult(
            plan=plan,
            raw=plan.model_dump(mode="json"),
            latency_s=time.perf_counter() - started,
            timeout=False,
            valid=True,
            errors=[],
            backend=self.name,
        )


class RuleMissionPlanner(DeterministicRecoveryClient):
    """Deterministic semantic mission planner for the three Phase-7 scenarios.

    This is a stand-in for the local LLM: it turns a structured mission change
    into the correct whitelisted high-level actions (REROUTE / REASSIGN /
    CHANGE_PRIORITY / ABORT) without ever emitting velocity commands.
    """

    name = "rule-mission-planner"

    def _build_plan(self, context: RecoveryContext) -> RecoveryPlan:
        change = context.mission_change
        if change and change.get("kind") == "fail_drone":
            return self._fail_drone_plan(context, change)
        if change and change.get("kind") == "block_corridor":
            return self._block_corridor_plan(context, change)
        if change and change.get("kind") == "priority_change":
            return self._priority_change_plan(context, change)
        if context.cause == "COORDINATION_DEGRADATION":
            return self._coordination_degradation_plan(context)
        return super()._build_plan(context)

    def _coordination_degradation_plan(
        self, context: RecoveryContext
    ) -> RecoveryPlan:
        """Deterministic deadlock breaker for a symmetric stand-off.

        The stalled drone yields briefly and reroutes to a lateral waypoint
        whose sign depends on the drone id.  This breaks the symmetric
        "everyone waits, nobody moves" state while HOCBF keeps the maneuver
        collision-free.
        """
        drone = context.agent_i
        snap = context.snapshots[drone]
        goal = context.base_goals[drone]
        delta_x = goal[0] - snap.position[0]
        delta_y = goal[1] - snap.position[1]
        length = math.hypot(delta_x, delta_y)
        if length < 1e-6:
            perp_x, perp_y = 0.0, 1.0
        else:
            ux, uy = delta_x / length, delta_y / length
            perp_x, perp_y = -uy, ux
        sign = 1.0 if drone % 2 == 0 else -1.0
        waypoint = (
            snap.position[0] + perp_x * sign * 2.0,
            snap.position[1] + perp_y * sign * 2.0,
            0.0,
        )
        commands = [
            RecoveryCommand(
                drone=drone,
                action="YIELD",
                waypoint=None,
                priority="normal",
                ttl_sec=2.0,
                command_id=f"yield-{context.timestamp_ms}-{drone}",
            ),
            RecoveryCommand(
                drone=drone,
                action="REROUTE",
                waypoint=waypoint,
                priority="normal",
                ttl_sec=4.0,
                command_id=f"reroute-{context.timestamp_ms}-{drone}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"break symmetric deadlock for drone {drone}",
            commands=commands,
            constraints=None,
            rationale="deterministic lateral yield for coordination degradation",
            timestamp_ms=context.timestamp_ms,
        )

    def _fail_drone_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        failed = int(change["drone"])
        target = context.base_goals[failed]
        healthy = min(
            (d for d in context.snapshots if d != failed),
            key=lambda d: _distance_2d(context.snapshots[d].position, target),
        )
        commands = [
            RecoveryCommand(
                drone=failed,
                action="ABORT",
                waypoint=None,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"abort-{context.timestamp_ms}-{failed}",
            ),
            RecoveryCommand(
                drone=healthy,
                action="REASSIGN",
                waypoint=target,
                priority="normal",
                ttl_sec=1.0,
                command_id=f"reassign-{context.timestamp_ms}-{healthy}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"reassign task of failed drone {failed} to drone {healthy}",
            commands=commands,
            constraints=None,
            rationale=f"drone {failed} failed; nearest healthy drone {healthy} takes over",
            timestamp_ms=context.timestamp_ms,
        )

    def _block_corridor_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        drone = int(change["drone"])
        x0, x1, y0, y1 = change["zone"]
        goal = context.base_goals[drone]
        corners = [
            (x0, y0),
            (x1, y0),
            (x0, y1),
            (x1, y1),
        ]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        corner = min(corners, key=lambda p: _distance_2d(p, goal))
        waypoint = (
            corner[0] + math.copysign(2.0, corner[0] - cx),
            corner[1] + math.copysign(2.0, corner[1] - cy),
            0.0,
        )
        command = RecoveryCommand(
            drone=drone,
            action="REROUTE",
            waypoint=waypoint,
            priority="normal",
            ttl_sec=5.0,
            command_id=f"reroute-{context.timestamp_ms}-{drone}",
        )
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"reroute drone {drone} around blocked corridor",
            commands=[command],
            constraints=None,
            rationale=f"corridor {change['zone']} became unavailable",
            timestamp_ms=context.timestamp_ms,
        )

    def _priority_change_plan(
        self, context: RecoveryContext, change: dict[str, Any]
    ) -> RecoveryPlan:
        high = int(change["high"])
        low = int(change["low"])
        commands = [
            RecoveryCommand(
                drone=high,
                action="CHANGE_PRIORITY",
                waypoint=None,
                priority="safety",
                ttl_sec=1.0,
                command_id=f"prio-{context.timestamp_ms}-{high}",
            ),
            RecoveryCommand(
                drone=low,
                action="YIELD",
                waypoint=None,
                priority="normal",
                ttl_sec=2.0,
                command_id=f"yield-{context.timestamp_ms}-{low}",
            ),
        ]
        return RecoveryPlan(
            schema_version=1,
            intent_text=f"prioritize drone {high} over drone {low}",
            commands=commands,
            constraints=None,
            rationale=f"mission priority change: {high} high, {low} normal",
            timestamp_ms=context.timestamp_ms,
        )


def _distance_2d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
