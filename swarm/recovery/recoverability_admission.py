"""Commit 前的失效任务可恢复性准入与 fail-closed hold。

本模块只产生任务级目标/航路意图，最终速度仍必须经过 Runtime Assurance。
准入器将失效机的制动轨迹视为障碍包络，在冻结候选集合中寻找与该包络
保持声明间距的最短折线路径。没有候选通过时，不提交 orphan goal，而是
冻结健康机当前位置作为安全 hold。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from swarm.geometry import Vector3
from swarm.recovery.executor import RecoveryOverrides
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class RecoverabilityAdmissionConfig:
    clearance_m: float = 2.4
    braking_accel_mps2: float = 2.0
    ring_extra_m: tuple[float, ...] = (0.4, 0.8, 1.2)
    ring_samples: int = 32
    obstacle_samples: int = 9
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)
    waypoint_epsilon_m: float = 0.55
    max_route_length_m: float = 18.0

    def __post_init__(self) -> None:
        if self.clearance_m <= 0.0:
            raise ValueError("clearance_m 必须为正")
        if self.braking_accel_mps2 <= 0.0:
            raise ValueError("braking_accel_mps2 必须为正")
        if self.ring_samples < 8:
            raise ValueError("ring_samples 至少为 8")
        if self.obstacle_samples < 2:
            raise ValueError("obstacle_samples 至少为 2")
        if not self.ring_extra_m or any(value <= 0.0 for value in self.ring_extra_m):
            raise ValueError("ring_extra_m 必须包含正数")


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


class RecoverabilityAdmissionCoordinator:
    """一次失效事件对应一次确定性 admission/hold 决策。"""

    def __init__(self, config: RecoverabilityAdmissionConfig | None = None) -> None:
        self.config = config or RecoverabilityAdmissionConfig()
        self.state = "idle"
        self.failed_drone: int | None = None
        self.healthy_drone: int | None = None
        self.trigger_step: int | None = None
        self.route: list[Vector3] = []
        self.route_index = 0
        self.hold_goal: Vector3 | None = None
        self.admission_count = 0
        self.rejection_count = 0
        self.plans_committed = 0
        self.unsafe_commit_count = 0
        self.candidate_count = 0
        self.predicted_min_clearance_m: float | None = None
        self.rejection_reason: str | None = None
        self.completed = False

    def _obstacle_centers(self, snapshot: DroneSnapshot) -> list[np.ndarray]:
        position = np.asarray(snapshot.position[:2], dtype=np.float64)
        velocity = np.asarray(
            snapshot.velocity[:2] if snapshot.velocity else (0.0, 0.0),
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(velocity))
        if speed <= 1e-9:
            return [position.copy() for _ in range(self.config.obstacle_samples)]
        stop_time = speed / self.config.braking_accel_mps2
        times = np.linspace(0.0, stop_time, self.config.obstacle_samples)
        direction = velocity / speed
        return [
            position
            + direction
            * max(
                0.0,
                speed * value
                - 0.5 * self.config.braking_accel_mps2 * value * value,
            )
            for value in times
        ]

    def _route_clearance(
        self, points: list[np.ndarray], obstacle_centers: list[np.ndarray]
    ) -> float:
        minimum = float("inf")
        for left, right in zip(points, points[1:]):
            for center in obstacle_centers:
                minimum = min(minimum, _segment_distance(center, left, right))
        return minimum

    def _inside_arena(self, point: np.ndarray) -> bool:
        x0, x1, y0, y1 = self.config.arena
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    def _candidate_routes(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_centers: list[np.ndarray],
    ) -> list[list[np.ndarray]]:
        routes: list[list[np.ndarray]] = [[start, goal]]
        anchor = obstacle_centers[-1]
        for extra in self.config.ring_extra_m:
            radius = self.config.clearance_m + extra
            for index in range(self.config.ring_samples):
                angle = 2.0 * math.pi * index / self.config.ring_samples
                waypoint = anchor + radius * np.asarray(
                    (math.cos(angle), math.sin(angle)), dtype=np.float64
                )
                if self._inside_arena(waypoint):
                    routes.append([start, waypoint, goal])
        return routes

    @staticmethod
    def _route_length(points: list[np.ndarray]) -> float:
        return sum(
            float(np.linalg.norm(right - left))
            for left, right in zip(points, points[1:])
        )

    def _admit(
        self,
        *,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_centers: list[np.ndarray],
        altitude: float,
    ) -> None:
        if min(
            float(np.linalg.norm(goal - center)) for center in obstacle_centers
        ) < self.config.clearance_m:
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.rejection_reason = "orphan_goal_inside_failed_vehicle_envelope"
            return
        candidates = self._candidate_routes(start, goal, obstacle_centers)
        self.candidate_count = len(candidates)
        admissible: list[tuple[float, float, list[np.ndarray]]] = []
        for points in candidates:
            clearance = self._route_clearance(points, obstacle_centers)
            length = self._route_length(points)
            if (
                clearance + 1e-9 >= self.config.clearance_m
                and length <= self.config.max_route_length_m
            ):
                admissible.append((length, -clearance, points))
        if not admissible:
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.rejection_reason = "no_candidate_inside_recoverability_envelope"
            return
        _, negative_clearance, selected = min(
            admissible, key=lambda item: (item[0], item[1])
        )
        self.predicted_min_clearance_m = -negative_clearance
        if self.predicted_min_clearance_m + 1e-9 < self.config.clearance_m:
            self.unsafe_commit_count += 1
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.rejection_reason = "internal_clearance_guard"
            return
        self.route = [
            (float(point[0]), float(point[1]), float(altitude))
            for point in selected[1:]
        ]
        self.route_index = 0
        self.state = "admitted"
        self.admission_count += 1
        self.plans_committed += 1

    def step(
        self,
        *,
        step: int,
        snapshots: Mapping[int, DroneSnapshot],
        base_goals: Mapping[int, Vector3],
        mission_change: dict | None,
    ) -> RecoveryOverrides:
        if mission_change is not None and self.state == "idle":
            if mission_change.get("kind") != "fail_drone":
                raise ValueError("recoverability admission 仅处理 fail_drone")
            failed = int(mission_change["drone"])
            healthy = [drone for drone in sorted(snapshots) if drone != failed]
            if len(healthy) != 1:
                raise ValueError("本 calibration 的任务准入要求恰好一架健康机")
            self.failed_drone = failed
            self.healthy_drone = healthy[0]
            self.trigger_step = step
            healthy_snapshot = snapshots[self.healthy_drone]
            self.hold_goal = tuple(
                float(value) for value in healthy_snapshot.position
            )
            self._admit(
                start=np.asarray(healthy_snapshot.position[:2], dtype=np.float64),
                goal=np.asarray(base_goals[failed][:2], dtype=np.float64),
                obstacle_centers=self._obstacle_centers(snapshots[failed]),
                altitude=healthy_snapshot.position[2],
            )

        result = RecoveryOverrides()
        if self.healthy_drone is None:
            return result
        if self.state == "rejected_hold":
            assert self.hold_goal is not None
            result.goal_override[self.healthy_drone] = self.hold_goal
            result.velocity_scale[self.healthy_drone] = 0.0
            return result
        if self.state in {"admitted", "complete"}:
            snapshot = snapshots[self.healthy_drone]
            while self.route_index < len(self.route) - 1:
                target = self.route[self.route_index]
                if float(
                    np.linalg.norm(
                        np.asarray(snapshot.position[:2]) - np.asarray(target[:2])
                    )
                ) > self.config.waypoint_epsilon_m:
                    break
                self.route_index += 1
            target = self.route[self.route_index]
            result.goal_override[self.healthy_drone] = target
            result.velocity_scale[self.healthy_drone] = 1.0
            if self.route_index == len(self.route) - 1 and float(
                np.linalg.norm(
                    np.asarray(snapshot.position[:2]) - np.asarray(target[:2])
                )
            ) <= self.config.waypoint_epsilon_m:
                self.state = "complete"
                self.completed = True
            return result
        return result

    def summary(self) -> dict[str, object]:
        return {
            "state": self.state,
            "failed_drone": self.failed_drone,
            "healthy_drone": self.healthy_drone,
            "trigger_step": self.trigger_step,
            "candidate_count": self.candidate_count,
            "admission_count": self.admission_count,
            "rejection_count": self.rejection_count,
            "plans_committed": self.plans_committed,
            "unsafe_commit_count": self.unsafe_commit_count,
            "predicted_min_clearance_m": self.predicted_min_clearance_m,
            "clearance_threshold_m": self.config.clearance_m,
            "route": [list(point) for point in self.route],
            "route_index": self.route_index,
            "hold_goal": (
                list(self.hold_goal) if self.hold_goal is not None else None
            ),
            "rejection_reason": self.rejection_reason,
            "completed": self.completed,
        }


class RAOnlySafeHoldCoordinator:
    """比较条件：权限撤销后不做任务准入，只冻结健康机安全 hold。"""

    def __init__(self) -> None:
        self.plans_committed = 0
        self.trigger_step: int | None = None
        self.healthy_drone: int | None = None
        self.hold_goal: Vector3 | None = None

    def step(
        self,
        *,
        step: int,
        snapshots: Mapping[int, DroneSnapshot],
        base_goals: Mapping[int, Vector3],
        mission_change: dict | None,
    ) -> RecoveryOverrides:
        del base_goals
        if mission_change is not None and self.hold_goal is None:
            failed = int(mission_change["drone"])
            healthy = [drone for drone in sorted(snapshots) if drone != failed]
            if len(healthy) != 1:
                raise ValueError("RA-only hold 要求恰好一架健康机")
            self.healthy_drone = healthy[0]
            self.hold_goal = tuple(
                float(value) for value in snapshots[self.healthy_drone].position
            )
            self.trigger_step = step
        result = RecoveryOverrides()
        if self.healthy_drone is not None and self.hold_goal is not None:
            result.goal_override[self.healthy_drone] = self.hold_goal
            result.velocity_scale[self.healthy_drone] = 0.0
        return result

    def summary(self) -> dict[str, object]:
        return {
            "state": "ra_only_hold" if self.hold_goal is not None else "idle",
            "trigger_step": self.trigger_step,
            "healthy_drone": self.healthy_drone,
            "hold_goal": list(self.hold_goal) if self.hold_goal is not None else None,
            "plans_committed": 0,
            "admission_count": 0,
            "rejection_count": 0,
            "unsafe_commit_count": 0,
        }
