"""终点占用感知的 C3 时空预约；所有意图仍由 RA 最终过滤。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from swarm.interfaces import CoordinationReservationDecision
from swarm.safety import DroneSnapshot, safe_holding_point


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ReservationConfig:
    conflict_zone_id: str
    conflict_zone_center: tuple[float, float]
    conflict_zone_radius_m: float = 2.0
    admission_horizon_s: float = 3.0
    reserve_threshold: float = 1.0
    predicted_rho_warning: float = 0.2
    fallback_d_safe_m: float = 1.6
    hold_offset_m: float = 1.5
    clearance_offset_m: float = 2.2
    position_epsilon_m: float = 0.35
    speed_epsilon_mps: float = 0.15
    settle_steps: int = 5
    ra_veto_streak_limit: int = 3
    transit_scale: float = 0.6
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)

    def __post_init__(self) -> None:
        if self.conflict_zone_radius_m <= 0.0:
            raise ValueError("conflict_zone_radius_m 必须为正")
        if not 2.0 <= self.admission_horizon_s <= 3.0:
            raise ValueError("admission_horizon_s 必须在 [2, 3] 秒")
        if self.settle_steps < 1 or self.ra_veto_streak_limit < 1:
            raise ValueError("settle/veto 步数必须为正")
        if not 0.0 < self.transit_scale <= 1.0:
            raise ValueError("transit_scale 必须在 (0, 1]")


@dataclass(frozen=True)
class ReservationDirectives:
    goal_overrides: dict[int, Vector3]
    velocity_scales: dict[int, float]
    decision: CoordinationReservationDecision


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    length_sq = float(delta @ delta)
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(((point - start) @ delta) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * delta)))


class C3ReservationCoordinator:
    """安全等待、目标清空和最终归位组成的确定性预约状态机。"""

    MODES = (
        "normal",
        "admission_pending",
        "staging",
        "pass",
        "clear",
        "release",
        "final_return",
        "revoke_hold",
        "complete",
    )

    def __init__(
        self,
        config: ReservationConfig,
        drone_ids: list[int],
        base_starts: Mapping[int, Vector3],
        base_goals: Mapping[int, Vector3],
    ) -> None:
        self.config = config
        self.drone_ids = tuple(sorted(drone_ids))
        self.base_starts = {drone: tuple(base_starts[drone]) for drone in self.drone_ids}
        self.base_goals = {drone: tuple(base_goals[drone]) for drone in self.drone_ids}
        self.mode = "normal"
        self.order: list[int] = []
        self.index = 0
        self.final_order: list[int] = []
        self.final_index = 0
        self.hold_goals: dict[int, Vector3] = {}
        self.clearance_goals: dict[int, Vector3] = {}
        self.fallback_goals: dict[int, Vector3] = {}
        self.service_latched: set[int] = set()
        self.cleared: set[int] = set()
        self.final_returned: set[int] = set()
        self.settle_counter = 0
        self.ra_veto_streak = 0
        self.ra_veto_count = 0
        self.trigger_count = 0
        self.revoke_count = 0
        self.last_triggers: list[str] = []
        self.last_nominal: dict[int, np.ndarray] = {}
        self.last_decision: CoordinationReservationDecision | None = None
        self.mode_steps = {mode: 0 for mode in self.MODES}

    def _entry_time(self, snapshot: DroneSnapshot, velocity: np.ndarray) -> float | None:
        position = np.asarray(snapshot.position[:2], dtype=np.float64)
        center = np.asarray(self.config.conflict_zone_center, dtype=np.float64)
        relative = position - center
        radius = self.config.conflict_zone_radius_m
        if float(relative @ relative) <= radius * radius:
            return 0.0
        a = float(velocity @ velocity)
        if a <= 1e-12:
            return None
        b = 2.0 * float(relative @ velocity)
        c = float(relative @ relative) - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None
        roots = (
            (-b - math.sqrt(discriminant)) / (2.0 * a),
            (-b + math.sqrt(discriminant)) / (2.0 * a),
        )
        valid = [value for value in roots if 0.0 <= value <= self.config.admission_horizon_s]
        return min(valid) if valid else None

    def _predicted_min_rho(
        self,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        d_safe: float,
    ) -> float:
        minimum = float("inf")
        for left_index, left in enumerate(self.drone_ids):
            for right in self.drone_ids[left_index + 1 :]:
                position = np.asarray(snapshots[left].position[:2]) - np.asarray(
                    snapshots[right].position[:2]
                )
                velocity = np.asarray(nominal[left]) - np.asarray(nominal[right])
                for value in np.linspace(0.0, self.config.admission_horizon_s, 31):
                    distance = float(np.linalg.norm(position + velocity * value))
                    minimum = min(minimum, (distance - d_safe) / d_safe)
        return minimum

    @staticmethod
    def _reserve_low(previous_results: Mapping[int, Any] | None, threshold: float) -> bool:
        reserves = [
            float(result.feasibility_reserve)
            for result in (previous_results or {}).values()
            if result.feasibility_reserve is not None
        ]
        return bool(reserves) and min(reserves) < threshold

    def _freeze_stage_goals(self, snapshots: Mapping[int, DroneSnapshot]) -> None:
        positions = {drone: snapshots[drone].position for drone in self.drone_ids}
        self.hold_goals = {
            drone: (
                *safe_holding_point(drone, positions, offset=self.config.hold_offset_m)[:2],
                snapshots[drone].position[2],
            )
            for drone in self.drone_ids
        }

    def _clearance_goal(self, drone: int) -> Vector3:
        start = np.asarray(self.base_starts[drone][:2], dtype=np.float64)
        goal = np.asarray(self.base_goals[drone][:2], dtype=np.float64)
        direction = goal - start
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else np.asarray((1.0, 0.0))
        perpendicular = np.asarray((-direction[1], direction[0]))
        candidates = [
            goal + sign * perpendicular * self.config.clearance_offset_m
            for sign in (-1.0, 1.0)
        ]
        x_min, x_max, y_min, y_max = self.config.arena
        candidates = [
            np.asarray((np.clip(value[0], x_min, x_max), np.clip(value[1], y_min, y_max)))
            for value in candidates
        ]
        other_paths = [
            (
                np.asarray(self.base_starts[other][:2], dtype=np.float64),
                np.asarray(self.base_goals[other][:2], dtype=np.float64),
            )
            for other in self.drone_ids
            if other != drone
        ]
        other_holds = [
            np.asarray(self.hold_goals[other][:2], dtype=np.float64)
            for other in self.drone_ids
            if other != drone
        ]

        def score(candidate: np.ndarray) -> tuple[float, float]:
            path_clearance = min(
                (_segment_distance(candidate, start_value, goal_value) for start_value, goal_value in other_paths),
                default=float("inf"),
            )
            hold_clearance = min(
                (float(np.linalg.norm(candidate - hold)) for hold in other_holds),
                default=float("inf"),
            )
            zone_clearance = float(
                np.linalg.norm(candidate - np.asarray(self.config.conflict_zone_center))
            )
            return min(path_clearance, hold_clearance), zone_clearance

        selected = max(candidates, key=score)
        return (float(selected[0]), float(selected[1]), self.base_goals[drone][2])

    def _freeze_clearance_goals(self) -> None:
        self.clearance_goals = {
            drone: self._clearance_goal(drone) for drone in self.drone_ids
        }

    def _freeze_fallback_goals(self, snapshots: Mapping[int, DroneSnapshot]) -> None:
        positions = {drone: snapshots[drone].position for drone in self.drone_ids}
        self.fallback_goals = {
            drone: (
                *safe_holding_point(drone, positions, offset=self.config.hold_offset_m)[:2],
                snapshots[drone].position[2],
            )
            for drone in self.drone_ids
        }

    def _settled(self, drone: int, goal: Vector3, snapshots: Mapping[int, DroneSnapshot]) -> bool:
        position_error = float(
            np.linalg.norm(np.asarray(snapshots[drone].position[:2]) - np.asarray(goal[:2]))
        )
        speed = float(np.linalg.norm(np.asarray(snapshots[drone].velocity or (0.0, 0.0, 0.0))[:2]))
        return position_error <= self.config.position_epsilon_m and speed <= self.config.speed_epsilon_mps

    def _all_settled(
        self,
        goals: Mapping[int, Vector3],
        drones: list[int],
        snapshots: Mapping[int, DroneSnapshot],
    ) -> bool:
        return all(self._settled(drone, goals[drone], snapshots) for drone in drones)

    def _active_service(self) -> int | None:
        return self.order[self.index] if self.index < len(self.order) else None

    def _active_final(self) -> int | None:
        return self.final_order[self.final_index] if self.final_index < len(self.final_order) else None

    def record_nominal(self, nominal: Mapping[int, np.ndarray]) -> None:
        self.last_nominal = {
            drone: np.asarray(value, dtype=np.float64) for drone, value in nominal.items()
        }

    def directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        previous_results: Mapping[int, Any] | None,
    ) -> ReservationDirectives:
        if self.mode in {"pass", "clear", "final_return"} and (
            self.ra_veto_streak >= self.config.ra_veto_streak_limit
        ):
            self.mode = "revoke_hold"
            self.revoke_count += 1
            self._freeze_fallback_goals(snapshots)

        entry_times = {
            drone: value
            for drone in self.drone_ids
            if (value := self._entry_time(snapshots[drone], np.asarray(nominal[drone])))
            is not None
        }
        d_safe = max(
            [
                float(result.d_safe)
                for result in (previous_results or {}).values()
                if result.d_safe is not None
            ],
            default=self.config.fallback_d_safe_m,
        )
        predicted_low = self._predicted_min_rho(snapshots, nominal, d_safe) <= self.config.predicted_rho_warning
        reserve_low = self._reserve_low(previous_results, self.config.reserve_threshold)

        if self.mode == "normal" and len(entry_times) >= 2 and (predicted_low or reserve_low):
            detected = sorted(entry_times, key=lambda drone: (entry_times[drone], drone))
            self.order = detected + [drone for drone in self.drone_ids if drone not in detected]
            self.final_order = list(reversed(self.order))
            self._freeze_stage_goals(snapshots)
            self._freeze_clearance_goals()
            self.last_triggers = ["conflict_zone"]
            if reserve_low:
                self.last_triggers.append("reserve_low")
            if predicted_low:
                self.last_triggers.append("predicted_rho_low")
            self.mode = "admission_pending"
            self.trigger_count += 1
        elif self.mode == "admission_pending":
            self.mode = "staging"
        elif self.mode == "staging":
            if self._all_settled(self.hold_goals, list(self.order), snapshots):
                self.settle_counter += 1
                if self.settle_counter >= self.config.settle_steps:
                    self.mode = "pass"
                    self.settle_counter = 0
            else:
                self.settle_counter = 0
        elif self.mode == "pass":
            active = self._active_service()
            if active is not None and self._settled(active, self.base_goals[active], snapshots):
                self.service_latched.add(active)
                self.mode = "clear"
                self.settle_counter = 0
        elif self.mode == "clear":
            active = self._active_service()
            if active is not None and self._settled(active, self.clearance_goals[active], snapshots):
                self.settle_counter += 1
                if self.settle_counter >= self.config.settle_steps:
                    self.cleared.add(active)
                    self.mode = "release"
                    self.settle_counter = 0
            else:
                self.settle_counter = 0
        elif self.mode == "release":
            self.index += 1
            self.ra_veto_streak = 0
            if self.index >= len(self.order):
                self.mode = "final_return"
            else:
                self.mode = "pass"
        elif self.mode == "final_return":
            active = self._active_final()
            if active is not None and self._settled(active, self.base_goals[active], snapshots):
                self.settle_counter += 1
                if self.settle_counter >= self.config.settle_steps:
                    self.final_returned.add(active)
                    self.final_index += 1
                    self.ra_veto_streak = 0
                    self.settle_counter = 0
                    if self.final_index >= len(self.final_order):
                        self.mode = "complete"
            else:
                self.settle_counter = 0

        active_service = self._active_service()
        active_final = self._active_final()
        active = (
            active_service
            if self.mode in {"pass", "clear"}
            else active_final if self.mode == "final_return" else None
        )
        goals: dict[int, Vector3] = {}
        scales: dict[int, float] = {}

        def stage_or_stop(drone: int, goal: Vector3) -> None:
            if self._settled(drone, goal, snapshots):
                scales[drone] = 0.0
            else:
                goals[drone] = goal
                scales[drone] = self.config.transit_scale

        if self.mode in {"admission_pending", "staging"}:
            for drone in self.drone_ids:
                stage_or_stop(drone, self.hold_goals[drone])
        elif self.mode in {"pass", "clear", "release"}:
            for drone in self.drone_ids:
                if drone == active_service:
                    if self.mode == "clear":
                        stage_or_stop(drone, self.clearance_goals[drone])
                    else:
                        scales[drone] = 1.0
                elif drone in self.cleared:
                    stage_or_stop(drone, self.clearance_goals[drone])
                else:
                    stage_or_stop(drone, self.hold_goals[drone])
        elif self.mode == "final_return":
            for drone in self.drone_ids:
                if drone == active_final:
                    scales[drone] = self.config.transit_scale
                elif drone in self.final_returned:
                    scales[drone] = 0.0
                else:
                    stage_or_stop(drone, self.clearance_goals[drone])
        elif self.mode == "revoke_hold":
            for drone in self.drone_ids:
                stage_or_stop(drone, self.fallback_goals[drone])
        elif self.mode == "complete":
            for drone in self.drone_ids:
                scales[drone] = 0.0

        held = [drone for drone in self.drone_ids if scales.get(drone) == 0.0 or drone in goals]
        clearance_drone = active_service if self.mode == "clear" else None
        decision = CoordinationReservationDecision(
            step=step,
            coordination_mode=self.mode,
            admission_triggers=list(self.last_triggers if self.mode not in {"normal", "complete"} else []),
            authorized_drone_ids=[active] if active is not None else [],
            held_drone_ids=[drone for drone in held if drone != active],
            frozen_hold_goals={
                drone: (
                    self.fallback_goals[drone]
                    if self.mode == "revoke_hold"
                    else self.clearance_goals[drone]
                    if drone in self.cleared or self.mode == "final_return"
                    else self.hold_goals[drone]
                )
                for drone in held
                if drone != active
            },
            clearance_drone_id=clearance_drone,
            clearance_goal=(self.clearance_goals[clearance_drone] if clearance_drone is not None else None),
            service_latched_drone_ids=sorted(self.service_latched),
            cleared_drone_ids=sorted(self.cleared),
            final_returned_drone_ids=sorted(self.final_returned),
            conflict_zone_id=(self.config.conflict_zone_id if self.mode not in {"normal", "complete"} else None),
            ra_veto_streak=self.ra_veto_streak,
            authority_source="fallback" if self.mode == "revoke_hold" else "c3_rule",
            timestamp_ms=timestamp_ms,
        )
        self.mode_steps[self.mode] += 1
        self.last_decision = decision
        return ReservationDirectives(goals, scales, decision)

    def observe_ra(self, results: Mapping[int, Any]) -> CoordinationReservationDecision | None:
        if self.last_decision is None:
            return None
        vetoed = False
        for drone in self.last_decision.authorized_drone_ids:
            result = results[drone]
            # ``safe_action`` may temporarily oppose a small goal command while
            # a feasible HOCBF command brakes PX4 overshoot.  That is filtering,
            # not an explicit veto.  Only the frozen feasibility flag constitutes
            # the clear RA rejection that may revoke a reservation.
            vetoed = vetoed or result.feasible is False
        if vetoed:
            self.ra_veto_count += 1
            self.ra_veto_streak += 1
        else:
            self.ra_veto_streak = 0
        self.last_decision = self.last_decision.model_copy(
            update={"ra_vetoed": vetoed, "ra_veto_streak": self.ra_veto_streak}
        )
        return self.last_decision

    def summary(self) -> dict[str, Any]:
        return {
            "trigger_count": self.trigger_count,
            "ra_veto_count": self.ra_veto_count,
            "revoke_count": self.revoke_count,
            "mode_steps": dict(self.mode_steps),
            "service_latched": sorted(self.service_latched),
            "cleared": sorted(self.cleared),
            "final_returned": sorted(self.final_returned),
            "complete": self.mode == "complete",
            "active_at_end": self.mode not in {"normal", "complete"},
            "clearance_goals": {drone: goal for drone, goal in self.clearance_goals.items()},
        }
