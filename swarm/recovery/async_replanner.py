"""Asynchronous Semantic Mission Manager (System 2).

Implements ``docs/decisions/llm_async_mission_replanning.md``:

    System 1 - Runtime Assurance / CBF keeps the current instant safe.
    System 2 - the LLM replans the mission when the current plan is no longer
               consistent with safety / mission / coordination constraints.

The trigger is a Mission Validity Monitor:

    E_replan = E_safety OR E_mission OR E_coord

The LLM request is submitted on a background thread and committed only after it
is ready and validated; the CBF filter still sees every action.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time

from swarm.geometry import Vector3
from swarm.interfaces import RecoveryPlan
from swarm.ra.runtime_assurance import FilterResult
from swarm.recovery.executor import (
    ActiveRecovery,
    RecoveryOverrides,
    apply_plan,
    default_active,
    overrides,
    tick,
)
from swarm.recovery.decision import expand_decision, parse_decision, semantic_errors
from swarm.recovery.llm import (
    DeterministicRecoveryClient,
    LLMRecoveryClient,
    RecoveryContext,
)
from swarm.recovery.validator import RecoveryValidator
from swarm.safety import DroneSnapshot


_MISSION_ACTIONS = {"REROUTE", "REASSIGN", "CHANGE_PRIORITY", "ABORT", "RETURN"}


def _cross_track_distance(
    point: Vector3,
    start: Vector3,
    goal: Vector3,
) -> float:
    """2D distance from ``point`` to the segment ``start -> goal``."""
    px, py = point[0] - start[0], point[1] - start[1]
    gx, gy = goal[0] - start[0], goal[1] - start[1]
    length_sq = gx * gx + gy * gy
    if length_sq == 0:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * gx + py * gy) / length_sq))
    cx, cy = start[0] + t * gx, start[1] + t * gy
    return math.hypot(point[0] - cx, point[1] - cy)


@dataclass(frozen=True)
class ReplanConfig:
    route_deviation_threshold_m: float = 1.5
    stall_window_s: float = 3.0
    stall_epsilon_m: float = 0.05
    replan_cooldown_s: float = 3.0
    replan_timeout_s: float = 10.0


@dataclass
class ReplanCounters:
    llm_calls: int = 0
    llm_syntactic_invalid: int = 0
    llm_schema_invalid: int = 0
    llm_semantic_invalid: int = 0
    llm_execution_invalid: int = 0
    llm_stale_invalid: int = 0
    llm_timeouts: int = 0
    triggers: int = 0
    trigger_causes: list[str] = field(default_factory=list)
    plans_committed: int = 0
    llm_plans_committed: int = 0
    fallback_plans_committed: int = 0
    mission_changes: int = 0
    replan_latencies_s: list[float] = field(default_factory=list)


@dataclass
class _PendingReplan:
    drone: int
    submitted_at: float
    submitted_wall: float
    future: Future
    context: RecoveryContext
    request_id: int


class AsyncMissionReplanner:
    """Non-blocking System-2 mission-validity trigger machine."""

    def __init__(
        self,
        *,
        client: LLMRecoveryClient | None = None,
        fallback: LLMRecoveryClient | None = None,
        validator: RecoveryValidator | None = None,
        config: ReplanConfig | None = None,
        dt: float = 0.1,
        blocking: bool = False,
    ) -> None:
        self.client = client or DeterministicRecoveryClient()
        self.fallback = fallback or DeterministicRecoveryClient()
        self.validator = validator or RecoveryValidator()
        self.config = config or ReplanConfig()
        self.dt = dt
        self.blocking = blocking
        self.counters = ReplanCounters()
        self.active: dict[int, ActiveRecovery] = {}
        self._base_starts: dict[int, Vector3] = {}
        self._mission_goals: dict[int, Vector3] = {}
        self._best_goal_dist: dict[int, float] = {}
        self._no_progress_since: dict[int, float] = {}
        self._last_replan_t: dict[int, float] = {}
        self._handled_drones: set[int] = set()
        self._pending: _PendingReplan | None = None
        self._request_id = 0
        self._executor = ThreadPoolExecutor(max_workers=1)

    # -- public -------------------------------------------------------------

    def step(
        self,
        *,
        t: float,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int | None = None,
        mission_change: dict | None = None,
    ) -> RecoveryOverrides:
        if not self.active:
            self.active = default_active(list(snapshots))
        if not self._base_starts:
            self._base_starts = {
                i: snapshots[i].position for i in snapshots
            }
            self._mission_goals = dict(base_goals)
            self._best_goal_dist = {
                i: _goal_dist(snapshots[i].position, base_goals[i])
                for i in snapshots
            }
            self._no_progress_since = {i: t for i in snapshots}

        tick(self.active, t)
        self._poll_pending(t, base_goals)

        trigger = self._evaluate_trigger(t, snapshots, mission_change)
        if trigger is not None and self._pending is None:
            drone, cause = trigger
            if t - self._last_replan_t.get(drone, float("-inf")) >= self.config.replan_cooldown_s:
                self._submit(
                    t=t,
                    drone=drone,
                    cause=cause,
                    snapshots=snapshots,
                    results=results,
                    current_goals=current_goals,
                    base_goals=base_goals,
                    timestamp_ms=timestamp_ms,
                    mission_change=mission_change,
                )
        return overrides(self.active)

    def priorities(self) -> dict[int, str]:
        return {drone: state.priority for drone, state in self.active.items()}

    # -- Mission Validity Monitor ------------------------------------------

    def _evaluate_trigger(
        self,
        t: float,
        snapshots: dict[int, DroneSnapshot],
        mission_change: dict | None,
    ) -> tuple[int, str] | None:
        # E_mission: explicit external change wins first.
        if mission_change is not None:
            drone = int(mission_change.get("drone", 0))
            return drone, "MISSION_CHANGE"

        # E_safety: cross-track route deviation.
        for i, snap in snapshots.items():
            if (
                self.active[i].aborted
                or self.active[i].goal_override is not None
                or i in self._handled_drones
            ):
                continue
            deviation = _cross_track_distance(
                snap.position,
                self._base_starts[i],
                self._mission_goals[i],
            )
            if deviation > self.config.route_deviation_threshold_m:
                return i, "ROUTE_DEVIATION"

        # E_coord: stalled goal progress.
        for i, snap in snapshots.items():
            if (
                self.active[i].aborted
                or self.active[i].goal_override is not None
                or i in self._handled_drones
            ):
                continue
            dist = _goal_dist(snap.position, self._mission_goals[i])
            if dist < self._best_goal_dist[i] - self.config.stall_epsilon_m:
                self._best_goal_dist[i] = dist
                self._no_progress_since[i] = t
            elif t - self._no_progress_since[i] >= self.config.stall_window_s:
                return i, "COORDINATION_DEGRADATION"

        return None

    # -- LLM interaction ----------------------------------------------------

    def _make_context(
        self,
        *,
        drone: int,
        cause: str,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int,
        mission_change: dict | None,
    ) -> RecoveryContext:
        result_for_margin = results.get(drone)
        return RecoveryContext(
            event="MISSION_PLAN_INVALIDATED",
            agent_i=drone,
            agent_j=drone,
            current_margin=(
                result_for_margin.safety_margin if result_for_margin is not None else 0.0
            ),
            predicted_min_margin=(
                result_for_margin.predicted_margin if result_for_margin is not None else None
            ),
            margin_degradation=(
                result_for_margin.degradation if result_for_margin is not None else None
            ),
            intervention_count=0,
            cause=cause,
            severity="MEDIUM",
            snapshots=snapshots,
            current_goals=current_goals,
            base_goals=base_goals,
            priorities=self.priorities(),
            timestamp_ms=timestamp_ms,
            mission_change=mission_change,
        )

    def _submit(
        self,
        *,
        t: float,
        drone: int,
        cause: str,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int | None,
        mission_change: dict | None,
    ) -> None:
        self.counters.triggers += 1
        self.counters.trigger_causes.append(cause)
        self.counters.llm_calls += 1
        self._request_id += 1
        context = self._make_context(
            drone=drone,
            cause=cause,
            snapshots=snapshots,
            results=results,
            current_goals=current_goals,
            base_goals=self._mission_goals,
            timestamp_ms=timestamp_ms if timestamp_ms is not None else int(time.time() * 1000),
            mission_change=mission_change,
        )
        future = self._executor.submit(self.client.generate, context)
        pending = _PendingReplan(
            drone=drone,
            submitted_at=t,
            submitted_wall=time.perf_counter(),
            future=future,
            context=context,
            request_id=self._request_id,
        )
        self._last_replan_t[drone] = t

        if self.blocking:
            self._wait_and_commit(pending, t, base_goals)
        else:
            self._pending = pending

    def _poll_pending(self, t: float, base_goals: dict[int, Vector3]) -> None:
        if self._pending is None:
            return
        pending = self._pending
        if t - pending.submitted_at > self.config.replan_timeout_s:
            self.counters.llm_timeouts += 1
            self._commit(
                self.fallback.generate(pending.context).plan,
                pending,
                t,
                base_goals,
                source="fallback",
            )
            self._pending = None
            return
        if not pending.future.done():
            return
        result = pending.future.result()
        plan, source = self._plan_from_llm(result, pending.context)
        self._commit(plan, pending, t, base_goals, source=source)
        self._pending = None

    def _wait_and_commit(
        self,
        pending: _PendingReplan,
        t: float,
        base_goals: dict[int, Vector3],
    ) -> None:
        result = pending.future.result()
        plan, source = self._plan_from_llm(result, pending.context)
        self._commit(plan, pending, t, base_goals, source=source)

    def _plan_from_llm(self, result, context) -> tuple[RecoveryPlan | None, str]:
        if result.raw is None:
            self.counters.llm_syntactic_invalid += 1
            return self.fallback.generate(context).plan, "fallback"

        # Deterministic stand-ins emit a full RecoveryPlan; validate directly.
        full = self.validator.validate(result.raw, now_ms=int(time.time() * 1000))
        if full.valid:
            return full.plan, "llm"
        if any(error.startswith("expired recovery command") for error in full.errors):
            self.counters.llm_stale_invalid += 1
            return self.fallback.generate(context).plan, "fallback"

        decision, _ = parse_decision(result.raw)
        if decision is None:
            self.counters.llm_schema_invalid += 1
            return self.fallback.generate(context).plan, "fallback"
        if semantic_errors(decision, context):
            self.counters.llm_semantic_invalid += 1
            return self.fallback.generate(context).plan, "fallback"
        try:
            expanded = expand_decision(decision, context)
        except ValueError:
            self.counters.llm_semantic_invalid += 1
            return self.fallback.generate(context).plan, "fallback"
        validation = self.validator.validate(
            expanded.model_dump(mode="json"), now_ms=int(time.time() * 1000)
        )
        if validation.valid:
            return validation.plan, "llm"
        if any(error.startswith("expired recovery command") for error in validation.errors):
            self.counters.llm_stale_invalid += 1
        else:
            self.counters.llm_execution_invalid += 1
        return self.fallback.generate(context).plan, "fallback"

    def _commit(
        self,
        plan: RecoveryPlan | None,
        pending: _PendingReplan,
        t: float,
        base_goals: dict[int, Vector3],
        source: str = "llm",
    ) -> None:
        if plan is None:
            return
        positions = {
            drone: snap.position
            for drone, snap in pending.context.snapshots.items()
        }
        apply_plan(self.active, plan, t, self._mission_goals, positions)
        self._handled_drones.update(command.drone for command in plan.commands)
        for command in plan.commands:
            if command.action == "REASSIGN" and command.waypoint is not None:
                self._mission_goals[command.drone] = command.waypoint
        self.counters.plans_committed += 1
        if source == "llm":
            self.counters.llm_plans_committed += 1
        else:
            self.counters.fallback_plans_committed += 1
        if any(c.action in _MISSION_ACTIONS for c in plan.commands):
            self.counters.mission_changes += 1
        self.counters.replan_latencies_s.append(time.perf_counter() - pending.submitted_wall)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _goal_dist(position: Vector3, goal: Vector3) -> float:
    return math.hypot(position[0] - goal[0], position[1] - goal[1])
