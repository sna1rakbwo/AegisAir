"""Three-layer semantic-recovery orchestrator (R0 / R1 / R2).

This implements the ``predictor_architecture`` decision:

    Level 1 - Predictive Pre-alert (speculative candidate, no mission change)
    Level 2 - Runtime Confirmation (commit a validated recovery plan)
    Level 3 - Hard CBF (always independent, applied by Runtime Assurance)

The three comparison modes are:

    R0 - no predictor; runtime confirmation triggers the LLM directly.
    R1 - predictor directly triggers LLM execution (speculation is committed).
    R2 - predictor speculative planning + runtime confirmation (main design).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
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
from swarm.recovery.llm import (
    DeterministicRecoveryClient,
    LLMRecoveryClient,
    LLMRecoveryResult,
    RecoveryContext,
)
from swarm.recovery.validator import RecoveryValidator
from swarm.safety import DroneSnapshot


class RecoveryMode(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


@dataclass(frozen=True)
class RecoveryConfig:
    rho_warn: float = 0.2
    g_th: float = 0.3
    n_th: int = 1
    intervention_window_s: float = 0.5
    latency_budget_s: float = 0.6
    candidate_ttl_s: float = 2.0
    confirmation_window_s: float = 1.0


@dataclass
class RecoveryCounters:
    llm_calls: int = 0
    llm_timeouts: int = 0
    llm_invalid: int = 0
    plans_executed: int = 0
    mission_changes: int = 0
    unnecessary_mission_changes: int = 0
    speculative_plans_generated: int = 0
    speculative_plans_discarded: int = 0
    recovery_latencies_s: list[float] = field(default_factory=list)
    candidate_lead_times_s: list[float] = field(default_factory=list)


@dataclass
class _PendingConfirmation:
    pair: tuple[int, int]
    created_s: float
    due_s: float
    counted: bool = False


_MISSION_ACTIONS = {"REROUTE", "REASSIGN", "CHANGE_PRIORITY", "ABORT", "RETURN"}


def _normalize_pair(i: int, j: int) -> tuple[int, int]:
    return (min(i, j), max(i, j))


class SemanticRecovery:
    """Stateful semantic-recovery trigger machine."""

    def __init__(
        self,
        *,
        mode: RecoveryMode | str,
        client: LLMRecoveryClient | None = None,
        validator: RecoveryValidator | None = None,
        config: RecoveryConfig | None = None,
        dt: float = 0.1,
    ) -> None:
        self.mode = RecoveryMode(mode)
        self.client = client or DeterministicRecoveryClient()
        self.fallback = DeterministicRecoveryClient()
        self.validator = validator or RecoveryValidator()
        self.config = config or RecoveryConfig()
        self.dt = dt
        self.counters = RecoveryCounters()
        self.active: dict[int, ActiveRecovery] = {}
        self._intervention_window: deque[tuple[float, tuple[int, int]]] = deque()
        self._pending: list[_PendingConfirmation] = []
        self._candidate: RecoveryPlan | None = None
        self._candidate_t: float | None = None
        self._candidate_pair: tuple[int, int] | None = None
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
    ) -> RecoveryOverrides:
        if not self.active:
            self.active = default_active(list(snapshots))
        tick(self.active, t)
        self._record_interventions(t, results)

        proactive = self._proactive_pair(results)
        confirmed = self._confirmed_pair(results, t)
        timestamp_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        if self.mode is RecoveryMode.R1:
            if proactive is not None:
                self._execute(
                    t=t,
                    drone=proactive[0],
                    other=proactive[1],
                    event="PREDICTED_CONFLICT",
                    snapshots=snapshots,
                    results=results,
                    current_goals=current_goals,
                    base_goals=base_goals,
                    timestamp_ms=timestamp_ms,
                    needs_confirmation=True,
                )

        elif self.mode is RecoveryMode.R0:
            if confirmed is not None:
                self._execute(
                    t=t,
                    drone=confirmed[0],
                    other=confirmed[1],
                    event="SAFETY_MARGIN_DEGRADATION",
                    snapshots=snapshots,
                    results=results,
                    current_goals=current_goals,
                    base_goals=base_goals,
                    timestamp_ms=timestamp_ms,
                    needs_confirmation=False,
                )

        else:  # R2
            if proactive is not None and self._candidate is None:
                self._generate_candidate(
                    t=t,
                    drone=proactive[0],
                    other=proactive[1],
                    snapshots=snapshots,
                    results=results,
                    current_goals=current_goals,
                    base_goals=base_goals,
                    timestamp_ms=timestamp_ms,
                )

            if confirmed is not None:
                if (
                    self._candidate is not None
                    and self._candidate_pair == _normalize_pair(*confirmed)
                ):
                    lead = t - self._candidate_t if self._candidate_t is not None else 0.0
                    self.counters.candidate_lead_times_s.append(lead)
                    self._apply_valid_plan(self._candidate, t, base_goals)
                    self.counters.recovery_latencies_s.append(self.dt)
                    self._candidate = None
                    self._candidate_t = None
                    self._candidate_pair = None
                else:
                    self._discard_candidate()
                    self._execute(
                        t=t,
                        drone=confirmed[0],
                        other=confirmed[1],
                        event="SAFETY_MARGIN_DEGRADATION",
                        snapshots=snapshots,
                        results=results,
                        current_goals=current_goals,
                        base_goals=base_goals,
                        timestamp_ms=timestamp_ms,
                        needs_confirmation=False,
                    )

            if self._candidate is not None and self._candidate_t is not None:
                if t - self._candidate_t > self.config.candidate_ttl_s:
                    self._discard_candidate()

        self._resolve_pending_confirmations(t, confirmed)
        return overrides(self.active)

    def priorities(self) -> dict[int, str]:
        return {drone: state.priority for drone, state in self.active.items()}

    # -- trigger detection --------------------------------------------------

    def _proactive_pair(self, results: dict[int, FilterResult]) -> tuple[int, int] | None:
        worst = min(
            results.values(),
            key=lambda r: (
                r.predicted_margin
                if r.predicted_margin is not None
                else float("inf")
            ),
        )
        if not worst.proactive or worst.worst_pair is None:
            return None
        return worst.drone, worst.worst_pair

    def _confirmed_pair(
        self, results: dict[int, FilterResult], t: float
    ) -> tuple[int, int] | None:
        worst = min(results.values(), key=lambda r: r.safety_margin)
        if worst.worst_pair is None:
            return None
        n_cbf = self._pair_intervention_count(worst.drone, worst.worst_pair, t)
        confirmed = (
            worst.safety_margin < self.config.rho_warn
            and (
                worst.degradation > self.config.g_th
                or n_cbf > self.config.n_th
            )
        )
        if not confirmed:
            return None
        return worst.drone, worst.worst_pair

    def _record_interventions(self, t: float, results: dict[int, FilterResult]) -> None:
        pairs: set[tuple[int, int]] = set()
        for result in results.values():
            if result.intervened and result.worst_pair is not None:
                pairs.add(_normalize_pair(result.drone, result.worst_pair))
        for pair in pairs:
            self._intervention_window.append((t, pair))
        horizon = t - self.config.confirmation_window_s
        while self._intervention_window and self._intervention_window[0][0] < horizon:
            self._intervention_window.popleft()

    def _pair_intervention_count(self, drone: int, other: int, t: float) -> int:
        pair = _normalize_pair(drone, other)
        horizon = t - self.config.intervention_window_s
        return sum(
            1
            for p_t, p in self._intervention_window
            if p == pair and p_t >= horizon
        )

    # -- LLM interaction ----------------------------------------------------

    def _make_context(
        self,
        *,
        drone: int,
        other: int,
        event: str,
        t: float,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int,
    ) -> RecoveryContext:
        result_for_margin = results.get(drone)
        return RecoveryContext(
            event=event,
            agent_i=drone,
            agent_j=other,
            current_margin=(
                result_for_margin.safety_margin if result_for_margin is not None else 0.0
            ),
            predicted_min_margin=(
                result_for_margin.predicted_margin if result_for_margin is not None else None
            ),
            margin_degradation=(
                result_for_margin.degradation if result_for_margin is not None else None
            ),
            intervention_count=self._pair_intervention_count(drone, other, t),
            cause=(
                "TRAJECTORY_CONFLICT"
                if event == "PREDICTED_CONFLICT"
                else "DYNAMICS"
            ),
            severity=self._severity(event, result_for_margin),
            snapshots=snapshots,
            current_goals=current_goals,
            base_goals=base_goals,
            priorities=self.priorities(),
            timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _severity(event: str, result: FilterResult | None) -> str:
        if event == "PREDICTED_CONFLICT":
            margin = result.predicted_margin if result is not None else None
            if margin is not None and margin < -0.3:
                return "CRITICAL"
            return "HIGH" if margin is not None and margin < 0.0 else "MEDIUM"
        if result is None:
            return "HIGH"
        return "CRITICAL" if result.safety_margin < 0.0 else "HIGH"

    def _request_plan(self, context: RecoveryContext) -> LLMRecoveryResult:
        self.counters.llm_calls += 1
        try:
            future = self._executor.submit(self.client.generate, context)
            result = future.result(timeout=self.config.latency_budget_s)
        except FutureTimeout:
            self.counters.llm_timeouts += 1
            return LLMRecoveryResult(
                plan=None,
                raw=None,
                latency_s=self.config.latency_budget_s,
                timeout=True,
                valid=False,
                errors=["latency budget exceeded"],
                backend=self.client.name,
            )

        validation = self.validator.validate(result.raw)
        if not validation.valid:
            self.counters.llm_invalid += 1
            return LLMRecoveryResult(
                plan=None,
                raw=result.raw,
                latency_s=result.latency_s,
                timeout=result.timeout,
                valid=False,
                errors=validation.errors,
                backend=result.backend,
            )
        return LLMRecoveryResult(
            plan=validation.plan,
            raw=result.raw,
            latency_s=result.latency_s,
            timeout=result.timeout,
            valid=True,
            errors=[],
            backend=result.backend,
        )

    def _generate_candidate(
        self,
        *,
        t: float,
        drone: int,
        other: int,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int,
    ) -> None:
        context = self._make_context(
            drone=drone,
            other=other,
            event="PREDICTED_CONFLICT",
            t=t,
            snapshots=snapshots,
            results=results,
            current_goals=current_goals,
            base_goals=base_goals,
            timestamp_ms=timestamp_ms,
        )
        result = self._request_plan(context)
        if result.plan is not None:
            self._candidate = result.plan
            self._candidate_t = t
            self._candidate_pair = _normalize_pair(drone, other)
            self.counters.speculative_plans_generated += 1

    def _execute(
        self,
        *,
        t: float,
        drone: int,
        other: int,
        event: str,
        snapshots: dict[int, DroneSnapshot],
        results: dict[int, FilterResult],
        current_goals: dict[int, Vector3],
        base_goals: dict[int, Vector3],
        timestamp_ms: int,
        needs_confirmation: bool,
    ) -> None:
        context = self._make_context(
            drone=drone,
            other=other,
            event=event,
            t=t,
            snapshots=snapshots,
            results=results,
            current_goals=current_goals,
            base_goals=base_goals,
            timestamp_ms=timestamp_ms,
        )
        result = self._request_plan(context)
        plan = result.plan
        if plan is None:
            # Assume AI can fail: a slow or malformed model falls back to the
            # deterministic safe plan instead of stalling recovery forever.
            plan = self.fallback.generate(context).plan

        if plan is None:
            return

        is_mission_change = self._apply_valid_plan(plan, t, base_goals)
        self.counters.recovery_latencies_s.append(result.latency_s + self.dt)

        if needs_confirmation and is_mission_change:
            self._pending.append(
                _PendingConfirmation(
                    pair=_normalize_pair(drone, other),
                    created_s=t,
                    due_s=t + self.config.confirmation_window_s,
                )
            )

    def _apply_valid_plan(
        self, plan: RecoveryPlan, t: float, base_goals: dict[int, Vector3]
    ) -> bool:
        apply_plan(self.active, plan, t, base_goals)
        self.counters.plans_executed += 1
        is_mission_change = any(c.action in _MISSION_ACTIONS for c in plan.commands)
        if is_mission_change:
            self.counters.mission_changes += 1
        return is_mission_change

    def _discard_candidate(self) -> None:
        if self._candidate is not None:
            self.counters.speculative_plans_discarded += 1
        self._candidate = None
        self._candidate_t = None
        self._candidate_pair = None

    def _resolve_pending_confirmations(
        self, t: float, confirmed: tuple[int, int] | None
    ) -> None:
        confirmed_pair = _normalize_pair(*confirmed) if confirmed is not None else None
        for entry in self._pending:
            if entry.counted:
                continue
            later_intervention = any(
                p == entry.pair and p_t >= entry.created_s
                for p_t, p in self._intervention_window
            )
            runtime_evidence = (
                entry.pair == confirmed_pair or later_intervention
            )
            if runtime_evidence:
                entry.counted = True
            elif t >= entry.due_s:
                entry.counted = True
                self.counters.unnecessary_mission_changes += 1

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
