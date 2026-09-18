"""C3 空时预约：确定性任务时隙在 RA 否决时只允许后移。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from swarm.interfaces import (
    CoordinationSpaceTimeReservationDecision,
    SpaceTimeReservationWindow,
)
from swarm.recovery.reservation import (
    C3ReservationCoordinator,
    ReservationConfig,
    ReservationDirectives,
    Vector3,
    _segment_distance,
)
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class SpaceTimeReservationConfig(ReservationConfig):
    """C3-STR 冻结参数；所有时长在触发时转换为离散控制步。"""

    rate_hz: float = 20.0
    staging_window_s: float = 8.0
    service_window_s: float = 10.0
    final_return_window_s: float = 8.0
    veto_delay_steps: int = 10
    return_compatibility_clearance_m: float = 1.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz 必须为正")
        if min(
            self.staging_window_s,
            self.service_window_s,
            self.final_return_window_s,
        ) <= 0.0:
            raise ValueError("预约窗口时长必须为正")
        if self.veto_delay_steps < 1:
            raise ValueError("veto_delay_steps 必须为正")
        if self.return_compatibility_clearance_m <= 0.0:
            raise ValueError("return_compatibility_clearance_m 必须为正")


@dataclass
class _Slot:
    slot_id: str
    phase: str
    drone_ids: tuple[int, ...]
    planned_open_step: int
    planned_close_step: int
    actual_open_step: int | None = None
    status: str = "scheduled"
    delay_steps: int = 0

    def interface(self) -> SpaceTimeReservationWindow:
        return SpaceTimeReservationWindow(
            slot_id=self.slot_id,
            phase=self.phase,
            drone_ids=list(self.drone_ids),
            planned_open_step=self.planned_open_step,
            planned_close_step=self.planned_close_step,
            actual_open_step=self.actual_open_step,
            status=self.status,
            delay_steps=self.delay_steps,
        )


@dataclass(frozen=True)
class SpaceTimeReservationDirectives:
    goal_overrides: dict[int, Vector3]
    velocity_scales: dict[int, float]
    decision: CoordinationSpaceTimeReservationDecision


def _segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    def cross(left: np.ndarray, right: np.ndarray) -> float:
        return float(left[0] * right[1] - left[1] * right[0])

    ab = b - a
    cd = d - c
    denominator = cross(ab, cd)
    if abs(denominator) <= 1e-12:
        return False
    t = cross(c - a, cd) / denominator
    u = cross(c - a, ab) / denominator
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _segment_segment_distance(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _segment_distance(a, c, d),
        _segment_distance(b, c, d),
        _segment_distance(c, a, b),
        _segment_distance(d, a, b),
    )


class C3SpaceTimeReservationCoordinator(C3ReservationCoordinator):
    """在原 C3 生命周期上增加可延迟时隙和兼容归位组。"""

    MODES = tuple(
        mode for mode in C3ReservationCoordinator.MODES if mode != "revoke_hold"
    ) + ("slot_wait", "slot_delay")

    def __init__(
        self,
        config: SpaceTimeReservationConfig,
        drone_ids: list[int],
        base_starts: Mapping[int, Vector3],
        base_goals: Mapping[int, Vector3],
    ) -> None:
        super().__init__(config, drone_ids, base_starts, base_goals)
        self.config = config
        self.slots: list[_Slot] = []
        self.final_groups: list[tuple[int, ...]] = []
        self.final_group_index = 0
        self.final_settle_counters = {drone: 0 for drone in self.drone_ids}
        self.schedule_revision = 0
        self.total_delay_steps = 0
        self.delay_remaining_steps = 0
        self.resume_mode: str | None = None
        self.trigger_step: int | None = None
        self.last_str_decision: CoordinationSpaceTimeReservationDecision | None = None

    def _compatible_return(self, left: int, right: int) -> bool:
        left_start = np.asarray(self.clearance_goals[left][:2], dtype=np.float64)
        left_goal = np.asarray(self.base_goals[left][:2], dtype=np.float64)
        right_start = np.asarray(self.clearance_goals[right][:2], dtype=np.float64)
        right_goal = np.asarray(self.base_goals[right][:2], dtype=np.float64)
        return (
            _segment_segment_distance(left_start, left_goal, right_start, right_goal)
            >= self.config.return_compatibility_clearance_m
        )

    def _build_final_groups(self) -> list[tuple[int, ...]]:
        groups: list[list[int]] = []
        for drone in self.final_order:
            placed = False
            for group in groups:
                if all(self._compatible_return(drone, other) for other in group):
                    group.append(drone)
                    placed = True
                    break
            if not placed:
                groups.append([drone])
        return [tuple(group) for group in groups]

    def _freeze_schedule(self, trigger_step: int) -> None:
        if self.slots:
            return
        self.trigger_step = trigger_step
        self.final_groups = self._build_final_groups()
        cursor = trigger_step + round(self.config.staging_window_s * self.config.rate_hz)
        service_steps = round(self.config.service_window_s * self.config.rate_hz)
        final_steps = round(self.config.final_return_window_s * self.config.rate_hz)
        for index, drone in enumerate(self.order):
            self.slots.append(
                _Slot(
                    slot_id=f"service-{index:02d}-uav{drone}",
                    phase="service",
                    drone_ids=(drone,),
                    planned_open_step=cursor,
                    planned_close_step=cursor + service_steps - 1,
                )
            )
            cursor += service_steps
        for index, group in enumerate(self.final_groups):
            suffix = "-".join(f"uav{drone}" for drone in group)
            self.slots.append(
                _Slot(
                    slot_id=f"return-{index:02d}-{suffix}",
                    phase="final_return",
                    drone_ids=group,
                    planned_open_step=cursor,
                    planned_close_step=cursor + final_steps - 1,
                )
            )
            cursor += final_steps

    def _service_slot(self) -> _Slot | None:
        active = self._active_service()
        if active is None:
            return None
        return next(
            (
                slot
                for slot in self.slots
                if slot.phase == "service" and slot.drone_ids == (active,)
            ),
            None,
        )

    def _final_slot(self) -> _Slot | None:
        if self.final_group_index >= len(self.final_groups):
            return None
        group = self.final_groups[self.final_group_index]
        return next(
            (
                slot
                for slot in self.slots
                if slot.phase == "final_return" and slot.drone_ids == group
            ),
            None,
        )

    def _active_slot(self) -> _Slot | None:
        if self.resume_mode == "final_return" or self.mode == "final_return":
            return self._final_slot()
        if self.resume_mode in {"pass", "clear", "release"} or self.mode in {
            "pass",
            "clear",
            "release",
        }:
            return self._service_slot()
        return None

    def _shift_incomplete_slots(self, delay_steps: int) -> None:
        if delay_steps <= 0:
            return
        active = self._active_slot()
        for slot in self.slots:
            if slot.status == "completed":
                continue
            if slot is active and slot.actual_open_step is not None:
                slot.planned_close_step += delay_steps
            else:
                slot.planned_open_step += delay_steps
                slot.planned_close_step += delay_steps
            slot.delay_steps += delay_steps
        self.total_delay_steps += delay_steps
        self.schedule_revision += 1

    def _mark_schedule(self, step: int) -> None:
        for slot in self.slots:
            if slot.phase == "service" and all(
                drone in self.cleared for drone in slot.drone_ids
            ):
                slot.status = "completed"
            elif slot.phase == "final_return" and all(
                drone in self.final_returned for drone in slot.drone_ids
            ):
                slot.status = "completed"
        active = self._active_slot()
        if active is not None and active.status != "completed":
            if active.actual_open_step is None:
                active.actual_open_step = step
            active.status = "delayed" if self.mode == "slot_delay" else "open"
            if step > active.planned_close_step:
                self._shift_incomplete_slots(step - active.planned_close_step + 1)

    def _committed_waypoints(self) -> dict[int, list[Vector3]]:
        active_service = self._active_service()
        if self.mode == "pass" and active_service is not None:
            return {
                active_service: [
                    self.base_goals[active_service],
                    self.clearance_goals[active_service],
                ]
            }
        if self.mode == "clear" and active_service is not None:
            return {active_service: [self.clearance_goals[active_service]]}
        if self.mode == "final_return":
            slot = self._final_slot()
            return (
                {drone: [self.base_goals[drone]] for drone in slot.drone_ids}
                if slot is not None
                else {}
            )
        if self.mode == "slot_delay":
            return {
                drone: [goal] for drone, goal in self.fallback_goals.items()
            }
        return {}

    def _decision(
        self,
        *,
        step: int,
        timestamp_ms: int,
        authorized: list[int],
        held: list[int],
        frozen_goals: Mapping[int, Vector3],
        authority_source: str = "c3_rule",
    ) -> CoordinationSpaceTimeReservationDecision:
        self._mark_schedule(step)
        active_slot = self._active_slot()
        decision = CoordinationSpaceTimeReservationDecision(
            step=step,
            coordination_mode=self.mode,
            admission_triggers=list(
                self.last_triggers if self.mode not in {"normal", "complete"} else []
            ),
            authorized_drone_ids=authorized,
            held_drone_ids=held,
            frozen_hold_goals=dict(frozen_goals),
            committed_waypoints=self._committed_waypoints(),
            reservation_windows=[slot.interface() for slot in self.slots],
            active_slot_id=active_slot.slot_id if active_slot is not None else None,
            service_latched_drone_ids=sorted(self.service_latched),
            cleared_drone_ids=sorted(self.cleared),
            final_returned_drone_ids=sorted(self.final_returned),
            conflict_zone_id=(
                self.config.conflict_zone_id
                if self.mode not in {"normal", "complete"}
                else None
            ),
            ra_veto_streak=self.ra_veto_streak,
            schedule_revision=self.schedule_revision,
            total_delay_steps=self.total_delay_steps,
            authority_source=authority_source,
            timestamp_ms=timestamp_ms,
        )
        self.last_decision = decision
        self.last_str_decision = decision
        return decision

    def _delay_directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
    ) -> SpaceTimeReservationDirectives:
        goals: dict[int, Vector3] = {}
        scales: dict[int, float] = {}
        for drone in self.drone_ids:
            goal = self.fallback_goals[drone]
            if self._settled(drone, goal, snapshots):
                scales[drone] = 0.0
            else:
                goals[drone] = goal
                scales[drone] = self.config.transit_scale
        decision = self._decision(
            step=step,
            timestamp_ms=timestamp_ms,
            authorized=[],
            held=list(self.drone_ids),
            frozen_goals=self.fallback_goals,
            authority_source="fallback",
        )
        self.mode_steps[self.mode] += 1
        self.delay_remaining_steps -= 1
        return SpaceTimeReservationDirectives(goals, scales, decision)

    def _slot_wait_directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
        resume_mode: str,
    ) -> SpaceTimeReservationDirectives:
        goals: dict[int, Vector3] = {}
        scales: dict[int, float] = {}
        active_service = self._active_service()
        active_final = set((self._final_slot().drone_ids if self._final_slot() else ()))
        for drone in self.drone_ids:
            if resume_mode == "final_return":
                goal = self.clearance_goals[drone]
                should_hold = drone not in self.final_returned
            elif drone == active_service:
                goal = self.hold_goals[drone]
                should_hold = True
            elif drone in self.cleared:
                goal = self.clearance_goals[drone]
                should_hold = True
            else:
                goal = self.hold_goals[drone]
                should_hold = True
            if not should_hold:
                scales[drone] = 0.0
            elif self._settled(drone, goal, snapshots):
                scales[drone] = 0.0
            else:
                goals[drone] = goal
                scales[drone] = self.config.transit_scale
        original_mode = self.mode
        self.mode = "slot_wait"
        slot = self._final_slot() if resume_mode == "final_return" else self._service_slot()
        frozen = {
            drone: goals.get(
                drone,
                self.clearance_goals[drone]
                if resume_mode == "final_return" or drone in self.cleared
                else self.hold_goals[drone],
            )
            for drone in self.drone_ids
            if drone not in self.final_returned
        }
        decision = self._decision(
            step=step,
            timestamp_ms=timestamp_ms,
            authorized=[],
            held=[drone for drone in self.drone_ids if drone not in self.final_returned],
            frozen_goals=frozen,
        )
        decision = decision.model_copy(
            update={"active_slot_id": slot.slot_id if slot is not None else None}
        )
        self.last_decision = decision
        self.last_str_decision = decision
        self.mode_steps["slot_wait"] += 1
        self.mode = original_mode
        return SpaceTimeReservationDirectives(goals, scales, decision)

    def _final_return_directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
    ) -> SpaceTimeReservationDirectives:
        slot = self._final_slot()
        if slot is not None and step < slot.planned_open_step:
            return self._slot_wait_directives(
                step=step,
                timestamp_ms=timestamp_ms,
                snapshots=snapshots,
                resume_mode="final_return",
            )
        if slot is None:
            self.mode = "complete"
            active_group: tuple[int, ...] = ()
        else:
            active_group = slot.drone_ids
            for drone in active_group:
                if self._settled(drone, self.base_goals[drone], snapshots):
                    self.final_settle_counters[drone] += 1
                else:
                    self.final_settle_counters[drone] = 0
            if all(
                self.final_settle_counters[drone] >= self.config.settle_steps
                for drone in active_group
            ):
                self.final_returned.update(active_group)
                self.final_group_index += 1
                self.ra_veto_streak = 0
                if self.final_group_index >= len(self.final_groups):
                    self.mode = "complete"
                    active_group = ()
                else:
                    active_group = self.final_groups[self.final_group_index]

        next_slot = self._final_slot()
        if (
            self.mode == "final_return"
            and next_slot is not None
            and step < next_slot.planned_open_step
        ):
            return self._slot_wait_directives(
                step=step,
                timestamp_ms=timestamp_ms,
                snapshots=snapshots,
                resume_mode="final_return",
            )

        goals: dict[int, Vector3] = {}
        scales: dict[int, float] = {}
        for drone in self.drone_ids:
            if drone in active_group:
                scales[drone] = self.config.transit_scale
            elif drone in self.final_returned or self.mode == "complete":
                scales[drone] = 0.0
            else:
                goal = self.clearance_goals[drone]
                if self._settled(drone, goal, snapshots):
                    scales[drone] = 0.0
                else:
                    goals[drone] = goal
                    scales[drone] = self.config.transit_scale
        held = [drone for drone in self.drone_ids if drone not in active_group]
        frozen = {
            drone: self.clearance_goals[drone]
            for drone in held
            if drone not in self.final_returned
        }
        decision = self._decision(
            step=step,
            timestamp_ms=timestamp_ms,
            authorized=list(active_group),
            held=held,
            frozen_goals=frozen,
        )
        self.mode_steps[self.mode] += 1
        return SpaceTimeReservationDirectives(goals, scales, decision)

    def directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        previous_results: Mapping[int, Any] | None,
    ) -> SpaceTimeReservationDirectives:
        if (
            self.last_str_decision is not None
            and self.last_str_decision.ra_vetoed
            and self.mode != "slot_delay"
        ):
            self.resume_mode = self.mode
            self._freeze_fallback_goals(snapshots)
            self._shift_incomplete_slots(self.config.veto_delay_steps)
            self.delay_remaining_steps = self.config.veto_delay_steps
            self.mode = "slot_delay"
            self.ra_veto_streak = 0

        if self.mode == "slot_delay":
            if self.delay_remaining_steps > 0:
                return self._delay_directives(
                    step=step,
                    timestamp_ms=timestamp_ms,
                    snapshots=snapshots,
                )
            self.mode = self.resume_mode or "staging"
            self.resume_mode = None
            self.fallback_goals = {}

        if self.mode == "final_return":
            return self._final_return_directives(
                step=step,
                timestamp_ms=timestamp_ms,
                snapshots=snapshots,
            )

        if self.mode == "pass":
            service_slot = self._service_slot()
            if service_slot is not None and step < service_slot.planned_open_step:
                return self._slot_wait_directives(
                    step=step,
                    timestamp_ms=timestamp_ms,
                    snapshots=snapshots,
                    resume_mode="pass",
                )

        base: ReservationDirectives = super().directives(
            step=step,
            timestamp_ms=timestamp_ms,
            snapshots=snapshots,
            nominal=nominal,
            previous_results=previous_results,
        )
        if self.order and not self.slots:
            self._freeze_schedule(step)
        if self.mode == "final_return":
            return self._final_return_directives(
                step=step,
                timestamp_ms=timestamp_ms,
                snapshots=snapshots,
            )
        if self.mode == "pass":
            service_slot = self._service_slot()
            if service_slot is not None and step < service_slot.planned_open_step:
                return self._slot_wait_directives(
                    step=step,
                    timestamp_ms=timestamp_ms,
                    snapshots=snapshots,
                    resume_mode="pass",
                )
        old = base.decision
        decision = self._decision(
            step=step,
            timestamp_ms=timestamp_ms,
            authorized=list(old.authorized_drone_ids),
            held=list(old.held_drone_ids),
            frozen_goals=old.frozen_hold_goals,
            authority_source=old.authority_source,
        )
        return SpaceTimeReservationDirectives(
            dict(base.goal_overrides),
            dict(base.velocity_scales),
            decision,
        )

    def observe_ra(
        self, results: Mapping[int, Any]
    ) -> CoordinationSpaceTimeReservationDecision | None:
        if self.last_str_decision is None:
            return None
        vetoed = any(
            results[drone].feasible is False
            for drone in self.last_str_decision.authorized_drone_ids
        )
        if vetoed:
            self.ra_veto_count += 1
            self.ra_veto_streak += 1
        else:
            self.ra_veto_streak = 0
        decision = self.last_str_decision.model_copy(
            update={"ra_vetoed": vetoed, "ra_veto_streak": self.ra_veto_streak}
        )
        self.last_decision = decision
        self.last_str_decision = decision
        return decision

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(
            {
                "schedule_revision": self.schedule_revision,
                "total_delay_steps": self.total_delay_steps,
                "final_return_groups": [list(group) for group in self.final_groups],
                "reservation_windows": [
                    slot.interface().model_dump(mode="json") for slot in self.slots
                ],
                "complete": self.mode == "complete",
                "active_at_end": self.mode not in {"normal", "complete"},
            }
        )
        return base
