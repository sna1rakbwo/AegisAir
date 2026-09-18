"""中等密度四机通行的确定性 C3 组时隙协调器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from swarm.interfaces import CoordinationAdmissionDecision
from swarm.safety import DroneSnapshot


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class GroupSlotConfig:
    """冻结的任务层组时隙；不包含任何执行器命令。"""

    conflict_zone_id: str
    conflict_zone_center: tuple[float, float]
    group_order: tuple[tuple[int, ...], ...]
    position_epsilon_m: float = 0.35
    speed_epsilon_mps: float = 0.15
    settle_steps: int = 5
    transit_scale: float = 0.8

    def __post_init__(self) -> None:
        if not self.group_order or any(not group for group in self.group_order):
            raise ValueError("group_order 必须包含至少一个非空组")
        flattened = [drone for group in self.group_order for drone in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("每架无人机只能属于一个通行组")
        if self.position_epsilon_m <= 0.0 or self.speed_epsilon_mps <= 0.0:
            raise ValueError("稳定阈值必须为正")
        if self.settle_steps < 1 or not 0.0 < self.transit_scale <= 1.0:
            raise ValueError("settle_steps/transit_scale 非法")


@dataclass(frozen=True)
class GroupSlotDirectives:
    goal_overrides: dict[int, Vector3]
    velocity_scales: dict[int, float]
    decision: CoordinationAdmissionDecision


class C3GroupSlotCoordinator:
    """先冻结等待点，再按兼容航路组依次释放的轻量C3。"""

    MODES = ("normal", "admission_pending", "hold", "pass", "release")

    def __init__(
        self,
        config: GroupSlotConfig,
        drone_ids: list[int],
        base_goals: Mapping[int, Vector3],
    ) -> None:
        self.config = config
        self.drone_ids = tuple(sorted(drone_ids))
        expected = {drone for group in config.group_order for drone in group}
        if set(self.drone_ids) != expected:
            raise ValueError("group_order 必须恰好覆盖 drone_ids")
        self.base_goals = {drone: tuple(base_goals[drone]) for drone in self.drone_ids}
        self.mode = "normal"
        self.group_index = 0
        self.hold_goals: dict[int, Vector3] = {}
        self.hold_enter_steps: dict[int, int] = {}
        self.completed: set[int] = set()
        self.finished = False
        self.settle_counts = {drone: 0 for drone in self.drone_ids}
        self.last_decision: CoordinationAdmissionDecision | None = None
        self.trigger_count = 0
        self.ra_veto_count = 0
        self.mode_steps = {mode: 0 for mode in self.MODES}

    def _active_group(self) -> tuple[int, ...]:
        if self.group_index >= len(self.config.group_order):
            return ()
        return self.config.group_order[self.group_index]

    def _settled(self, drone: int, snapshots: Mapping[int, DroneSnapshot]) -> bool:
        position_error = float(
            np.linalg.norm(
                np.asarray(snapshots[drone].position[:2])
                - np.asarray(self.base_goals[drone][:2])
            )
        )
        speed = float(
            np.linalg.norm(
                np.asarray(snapshots[drone].velocity or (0.0, 0.0, 0.0))[:2]
            )
        )
        return (
            position_error <= self.config.position_epsilon_m
            and speed <= self.config.speed_epsilon_mps
        )

    def _freeze_holds(self, snapshots: Mapping[int, DroneSnapshot], step: int) -> None:
        self.hold_goals = {
            drone: tuple(snapshots[drone].position) for drone in self.drone_ids
        }
        self.hold_enter_steps = {drone: step for drone in self.drone_ids}

    def directives(
        self,
        *,
        step: int,
        timestamp_ms: int,
        snapshots: Mapping[int, DroneSnapshot],
        nominal: Mapping[int, np.ndarray],
        previous_results: Mapping[int, Any] | None,
    ) -> GroupSlotDirectives:
        del nominal, previous_results
        if self.mode == "normal" and not self.finished:
            self._freeze_holds(snapshots, step)
            self.mode = "admission_pending"
            self.trigger_count += 1
        elif self.mode == "admission_pending":
            self.mode = "hold"
        elif self.mode == "hold":
            self.mode = "pass"
        elif self.mode == "pass":
            active = self._active_group()
            for drone in active:
                if self._settled(drone, snapshots):
                    self.settle_counts[drone] += 1
                else:
                    self.settle_counts[drone] = 0
            if active and all(
                self.settle_counts[drone] >= self.config.settle_steps
                for drone in active
            ):
                self.completed.update(active)
                self.mode = "release"
        elif self.mode == "release":
            self.group_index += 1
            if self.group_index >= len(self.config.group_order):
                self.mode = "normal"
                self.finished = True
            else:
                self.mode = "pass"

        active = self._active_group() if self.mode == "pass" else ()
        held = [drone for drone in self.drone_ids if drone not in active and drone not in self.completed]
        goals: dict[int, Vector3] = {}
        scales: dict[int, float] = {}
        for drone in self.drone_ids:
            if drone in active:
                scales[drone] = self.config.transit_scale
            elif drone in self.completed:
                scales[drone] = 0.0
            else:
                goals[drone] = self.hold_goals[drone]
                scales[drone] = 0.0
        decision = CoordinationAdmissionDecision(
            step=step,
            coordination_mode=self.mode,
            admission_triggers=(
                ["conflict_zone"] if self.mode != "normal" else []
            ),
            authorized_drone_ids=list(active),
            held_drone_ids=held,
            frozen_hold_goals={drone: self.hold_goals[drone] for drone in held},
            hold_enter_steps={drone: self.hold_enter_steps[drone] for drone in held},
            conflict_zone_id=(
                self.config.conflict_zone_id if self.mode != "normal" else None
            ),
            authority_source="c3_rule",
            timestamp_ms=timestamp_ms,
        )
        self.mode_steps[self.mode] += 1
        self.last_decision = decision
        return GroupSlotDirectives(goals, scales, decision)

    def observe_ra(
        self, results: Mapping[int, Any]
    ) -> CoordinationAdmissionDecision | None:
        if self.last_decision is None:
            return None
        vetoed = any(
            results[drone].feasible is False
            for drone in self.last_decision.authorized_drone_ids
        )
        if vetoed:
            self.ra_veto_count += 1
        self.last_decision = self.last_decision.model_copy(
            update={"ra_vetoed": vetoed}
        )
        return self.last_decision

    def summary(self) -> dict[str, Any]:
        return {
            "trigger_count": self.trigger_count,
            "ra_veto_count": self.ra_veto_count,
            "mode_steps": dict(self.mode_steps),
            "group_order": [list(group) for group in self.config.group_order],
            "completed": sorted(self.completed),
            "complete": self.finished,
            "active_at_end": self.mode != "normal",
        }
