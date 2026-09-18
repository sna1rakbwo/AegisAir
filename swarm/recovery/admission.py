"""事件触发的 C3 冲突区通行准入；不承担低层安全控制。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from swarm.interfaces import CoordinationAdmissionDecision
from swarm.safety import DroneSnapshot, safe_holding_point


@dataclass(frozen=True)
class AdmissionConfig:
    conflict_zone_id: str
    conflict_zone_center: tuple[float, float]
    conflict_zone_radius_m: float
    admission_horizon_s: float
    reserve_threshold: float
    predicted_rho_warning: float
    fallback_d_safe_m: float
    release_margin_m: float = 0.5
    hold_offset_m: float = 1.5
    hold_goal_epsilon_m: float = 0.35

    def __post_init__(self) -> None:
        if self.conflict_zone_radius_m <= 0.0:
            raise ValueError("conflict_zone_radius_m 必须为正")
        if not 2.0 <= self.admission_horizon_s <= 3.0:
            raise ValueError("admission_horizon_s 必须预注册在 [2, 3] 秒")
        if self.fallback_d_safe_m <= 0.0:
            raise ValueError("fallback_d_safe_m 必须为正")


@dataclass(frozen=True)
class AdmissionDirectives:
    goal_overrides: dict[int, tuple[float, float, float]]
    velocity_scales: dict[int, float]
    decision: CoordinationAdmissionDecision


class C3AdmissionCoordinator:
    """确定性、事件触发、RA 不可绕过的共享冲突区准入状态机。"""

    def __init__(self, config: AdmissionConfig, drone_ids: list[int]) -> None:
        self.config = config
        self.drone_ids = tuple(sorted(drone_ids))
        self.mode = "normal"
        self.order: list[int] = []
        self.index = 0
        self.frozen_hold_goals: dict[int, tuple[float, float, float]] = {}
        self.hold_enter_steps: dict[int, int] = {}
        self.passed: set[int] = set()
        self.authorized_was_inside = False
        self.last_triggers: list[str] = []
        self.last_decision: CoordinationAdmissionDecision | None = None
        self.last_nominal: dict[int, np.ndarray] = {}
        self.trigger_count = 0
        self.ra_veto_count = 0
        self.mode_steps = {name: 0 for name in (
            "normal", "admission_pending", "hold", "pass", "release"
        )}

    def _inside(self, position: tuple[float, float, float], margin: float = 0.0) -> bool:
        center = np.asarray(self.config.conflict_zone_center, dtype=np.float64)
        return float(np.linalg.norm(np.asarray(position[:2]) - center)) <= (
            self.config.conflict_zone_radius_m + margin
        )

    def _entry_time(
        self, snapshot: DroneSnapshot, velocity: np.ndarray
    ) -> float | None:
        position = np.asarray(snapshot.position[:2], dtype=np.float64)
        center = np.asarray(self.config.conflict_zone_center, dtype=np.float64)
        rel = position - center
        radius = self.config.conflict_zone_radius_m
        if float(rel @ rel) <= radius * radius:
            return 0.0
        a = float(velocity @ velocity)
        if a <= 1e-12:
            return None
        b = 2.0 * float(rel @ velocity)
        c = float(rel @ rel) - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        roots = [(-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a)]
        candidates = [value for value in roots if 0.0 <= value <= self.config.admission_horizon_s]
        return min(candidates) if candidates else None

    def _predicted_min_rho(
        self,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        d_safe: float,
    ) -> float:
        minimum = float("inf")
        times = np.linspace(0.0, self.config.admission_horizon_s, 26)
        for left_index, left in enumerate(self.drone_ids):
            for right in self.drone_ids[left_index + 1 :]:
                rel_position = np.asarray(snapshots[left].position[:2]) - np.asarray(
                    snapshots[right].position[:2]
                )
                rel_velocity = np.asarray(nominal[left]) - np.asarray(nominal[right])
                for value in times:
                    distance = float(np.linalg.norm(rel_position + rel_velocity * value))
                    minimum = min(minimum, (distance - d_safe) / d_safe)
        return minimum

    @staticmethod
    def _reserve_low(previous_results: Mapping[int, Any] | None, threshold: float) -> bool:
        if not previous_results:
            return False
        reserves = [
            float(result.feasibility_reserve)
            for result in previous_results.values()
            if result.feasibility_reserve is not None
        ]
        return bool(reserves) and min(reserves) < threshold

    def _freeze_holds(
        self, snapshots: Mapping[int, DroneSnapshot], step: int
    ) -> None:
        positions = {drone: snapshots[drone].position for drone in self.drone_ids}
        self.frozen_hold_goals = {
            drone: (
                *safe_holding_point(
                    drone, positions, offset=self.config.hold_offset_m
                )[:2],
                snapshots[drone].position[2],
            )
            for drone in self.drone_ids
        }
        self.hold_enter_steps = {drone: step for drone in self.drone_ids}

    def _active_authorized(self) -> int | None:
        if self.index >= len(self.order):
            return None
        return self.order[self.index]

    def directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        previous_results: Mapping[int, Any] | None,
    ) -> AdmissionDirectives:
        self.last_nominal = {
            drone: np.asarray(value, dtype=np.float64) for drone, value in nominal.items()
        }
        entry_times = {
            drone: value
            for drone in self.drone_ids
            if (value := self._entry_time(snapshots[drone], self.last_nominal[drone]))
            is not None
        }
        d_safe_values = [
            float(result.d_safe)
            for result in (previous_results or {}).values()
            if result.d_safe is not None
        ]
        d_safe = max(d_safe_values, default=self.config.fallback_d_safe_m)
        predicted_rho_low = (
            self._predicted_min_rho(snapshots, self.last_nominal, d_safe)
            <= self.config.predicted_rho_warning
        )
        reserve_low = self._reserve_low(previous_results, self.config.reserve_threshold)

        if self.mode == "normal" and len(entry_times) >= 2 and (
            reserve_low or predicted_rho_low
        ):
            self.order = sorted(entry_times, key=lambda drone: (entry_times[drone], drone))
            self.index = 0
            self.passed.clear()
            self.authorized_was_inside = False
            self._freeze_holds(snapshots, step)
            self.last_triggers = ["conflict_zone"]
            if reserve_low:
                self.last_triggers.append("reserve_low")
            if predicted_rho_low:
                self.last_triggers.append("predicted_rho_low")
            self.mode = "admission_pending"
            self.trigger_count += 1
        elif self.mode == "admission_pending":
            self.mode = "hold"
        elif self.mode == "hold":
            self.mode = "pass"
        elif self.mode == "pass":
            authorized = self._active_authorized()
            if authorized is not None:
                inside = self._inside(snapshots[authorized].position)
                self.authorized_was_inside = self.authorized_was_inside or inside
                reached_goal = float(
                    np.linalg.norm(
                        np.asarray(snapshots[authorized].position[:2])
                        - np.asarray(snapshots[authorized].target[:2])
                    )
                ) < 0.5 if snapshots[authorized].target is not None else False
                # 本 calibration 采用更保守的“穿越目标完成后交棒”。仅以首次
                # 离开冲突区永久放行会漏掉随后被 RA 反向推回的飞行器。
                if reached_goal:
                    self.passed.add(authorized)
                    self.mode = "release"
        elif self.mode == "release":
            self.index += 1
            self.authorized_was_inside = False
            if self.index >= len(self.order):
                self.mode = "normal"
                self.order = []
                self.frozen_hold_goals = {}
                self.hold_enter_steps = {}
                self.last_triggers = []
            else:
                self.mode = "pass"

        authorized = self._active_authorized() if self.mode == "pass" else None
        if self.mode in {"admission_pending", "hold", "release"}:
            held = [drone for drone in self.order if drone not in self.passed]
        elif self.mode == "pass":
            held = [
                drone
                for drone in self.order
                if drone != authorized and drone not in self.passed
            ]
        else:
            held = []
        goal_overrides: dict[int, tuple[float, float, float]] = {}
        velocity_scales: dict[int, float] = {}
        for drone in held:
            goal = self.frozen_hold_goals[drone]
            distance = float(
                np.linalg.norm(
                    np.asarray(snapshots[drone].position[:2]) - np.asarray(goal[:2])
                )
            )
            if distance <= self.config.hold_goal_epsilon_m:
                velocity_scales[drone] = 0.0
            else:
                goal_overrides[drone] = goal
                velocity_scales[drone] = 0.6
        decision = CoordinationAdmissionDecision(
            step=step,
            coordination_mode=self.mode,
            admission_triggers=list(self.last_triggers if self.mode != "normal" else []),
            authorized_drone_ids=[authorized] if authorized is not None else [],
            held_drone_ids=held,
            frozen_hold_goals={drone: self.frozen_hold_goals[drone] for drone in held},
            hold_enter_steps={drone: self.hold_enter_steps[drone] for drone in held},
            conflict_zone_id=(self.config.conflict_zone_id if self.mode != "normal" else None),
            authority_source="c3_rule",
            timestamp_ms=timestamp_ms,
        )
        self.mode_steps[self.mode] += 1
        self.last_decision = decision
        return AdmissionDirectives(goal_overrides, velocity_scales, decision)

    def observe_ra(self, results: Mapping[int, Any]) -> CoordinationAdmissionDecision | None:
        if self.last_decision is None:
            return None
        vetoed = False
        for drone in self.last_decision.authorized_drone_ids:
            result = results[drone]
            nominal = self.last_nominal.get(drone, np.zeros(2))
            safe = np.asarray(result.safe_action, dtype=np.float64)
            vetoed = vetoed or result.feasible is False or (
                float(np.linalg.norm(nominal)) > 0.1 and float(safe @ nominal) <= 0.0
            )
        if vetoed:
            self.ra_veto_count += 1
        self.last_decision = self.last_decision.model_copy(update={"ra_vetoed": vetoed})
        return self.last_decision

    def summary(self) -> dict[str, Any]:
        return {
            "trigger_count": self.trigger_count,
            "ra_veto_count": self.ra_veto_count,
            "mode_steps": dict(self.mode_steps),
            "active_at_end": self.mode != "normal",
        }
