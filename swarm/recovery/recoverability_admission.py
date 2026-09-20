"""Commit 前的失效任务可恢复性准入与 fail-closed hold。

本模块只产生任务级目标/航路意图，最终速度仍必须经过 Runtime Assurance。
准入器将失效机的制动轨迹视为连续障碍包络，在冻结候选集合中先做几何
净空检查，再按健康机当前速度、输入限制和执行响应进行时间对齐 rollout。
没有候选通过时，不提交 orphan goal，而是冻结健康机当前位置作为安全
hold。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping

import numpy as np

from swarm.geometry import Vector3
from swarm.recovery.executor import RecoveryOverrides
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class RecoverabilityAdmissionConfig:
    clearance_m: float = 2.4
    braking_accel_mps2: float = 2.0
    braking_accel_uncertainty_mps2: float = 0.25
    ring_extra_m: tuple[float, ...] = (0.4, 0.8, 1.2)
    ring_samples: int = 32
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)
    waypoint_epsilon_m: float = 0.55
    max_route_length_m: float = 18.0
    dt_s: float = 0.05
    rollout_horizon_s: float = 16.0
    reaction_delay_s: float = 0.05
    goal_gain_s_inv: float = 1.5
    velocity_gain_s_inv: float = 2.0
    healthy_velocity_limit_mps: float = 1.5
    healthy_acceleration_limit_mps2: float = 2.0
    execution_tau_s: float = 0.7
    command_feedforward_tau_s: float = 0.7
    terminal_speed_mps: float = 0.15
    tracking_error_buffer_m: float = 0.20

    def __post_init__(self) -> None:
        if self.clearance_m <= 0.0:
            raise ValueError("clearance_m 必须为正")
        if self.braking_accel_mps2 <= 0.0:
            raise ValueError("braking_accel_mps2 必须为正")
        if not 0.0 <= self.braking_accel_uncertainty_mps2 < self.braking_accel_mps2:
            raise ValueError("制动不确定性必须非负且小于名义制动能力")
        if self.ring_samples < 8:
            raise ValueError("ring_samples 至少为 8")
        if not self.ring_extra_m or any(value <= 0.0 for value in self.ring_extra_m):
            raise ValueError("ring_extra_m 必须包含正数")
        if self.dt_s <= 0.0 or self.rollout_horizon_s <= 0.0:
            raise ValueError("rollout 时间参数必须为正")
        if self.reaction_delay_s < 0.0:
            raise ValueError("reaction_delay_s 必须非负")
        if self.goal_gain_s_inv <= 0.0 or self.velocity_gain_s_inv <= 0.0:
            raise ValueError("健康机控制增益必须为正")
        if (
            self.healthy_velocity_limit_mps <= 0.0
            or self.healthy_acceleration_limit_mps2 <= 0.0
        ):
            raise ValueError("健康机速度和加速度限制必须为正")
        if self.execution_tau_s <= 0.0 or self.command_feedforward_tau_s <= 0.0:
            raise ValueError("执行响应参数必须为正")
        if self.terminal_speed_mps < 0.0 or self.tracking_error_buffer_m < 0.0:
            raise ValueError("终端速度和跟踪误差预算必须非负")

    @property
    def effective_clearance_m(self) -> float:
        return self.clearance_m + self.tracking_error_buffer_m

    @property
    def conservative_braking_accel_mps2(self) -> float:
        return self.braking_accel_mps2 - self.braking_accel_uncertainty_mps2


@dataclass(frozen=True)
class _RouteRollout:
    feasible: bool
    completed: bool
    min_clearance_m: float
    completion_time_s: float | None
    terminal_speed_mps: float
    rejection_reason: str | None


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(delta @ delta)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    denominator = _cross_2d(first_delta, second_delta)
    offset = second_start - first_start
    if abs(denominator) <= 1e-12:
        if abs(_cross_2d(offset, first_delta)) > 1e-12:
            return False
        first_norm = float(first_delta @ first_delta)
        if first_norm <= 1e-12:
            return _segment_distance(first_start, second_start, second_end) <= 1e-12
        lower = float((offset @ first_delta) / first_norm)
        upper = lower + float((second_delta @ first_delta) / first_norm)
        return max(min(lower, upper), 0.0) <= min(max(lower, upper), 1.0) + 1e-12
    first_fraction = _cross_2d(offset, second_delta) / denominator
    second_fraction = _cross_2d(offset, first_delta) / denominator
    return (
        -1e-12 <= first_fraction <= 1.0 + 1e-12
        and -1e-12 <= second_fraction <= 1.0 + 1e-12
    )


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _segment_distance(first_start, second_start, second_end),
        _segment_distance(first_end, second_start, second_end),
        _segment_distance(second_start, first_start, first_end),
        _segment_distance(second_end, first_start, first_end),
    )


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
        self.geometric_min_clearance_m: float | None = None
        self.predicted_completion_time_s: float | None = None
        self.predicted_terminal_speed_mps: float | None = None
        self.dynamic_rejection_count = 0
        self.geometric_rejection_count = 0
        self.route_length_rejection_count = 0
        self.rollout_candidate_count = 0
        self.dynamic_rejection_reasons: dict[str, int] = {}
        self.decision_latency_ms: float | None = None
        self.rejection_reason: str | None = None
        self.completed = False

    def _failed_trajectory(self, snapshot: DroneSnapshot) -> list[np.ndarray]:
        position = np.asarray(snapshot.position[:2], dtype=np.float64)
        velocity = np.asarray(
            snapshot.velocity[:2] if snapshot.velocity else (0.0, 0.0),
            dtype=np.float64,
        )
        points = [position.copy()]
        steps = int(math.ceil(self.config.rollout_horizon_s / self.config.dt_s))
        delay_steps = int(math.ceil(self.config.reaction_delay_s / self.config.dt_s))
        decay = 1.0 - math.exp(-self.config.dt_s / self.config.execution_tau_s)
        for step in range(steps):
            speed = float(np.linalg.norm(velocity))
            next_velocity = velocity.copy()
            if step >= delay_steps and speed > 1e-9:
                speed_reduction = min(
                    decay * speed,
                    self.config.conservative_braking_accel_mps2
                    * self.config.dt_s,
                )
                next_velocity *= max(0.0, 1.0 - speed_reduction / speed)
            position = position + 0.5 * self.config.dt_s * (
                velocity + next_velocity
            )
            velocity = next_velocity
            points.append(position.copy())
        return points

    def _route_clearance(
        self, points: list[np.ndarray], failed_trajectory: list[np.ndarray]
    ) -> float:
        # The revoked vehicle's zero-command braking model preserves its
        # measured velocity direction, so the complete swept centerline is the
        # segment from the first to the final predicted point. Using that
        # segment is continuous and avoids a sampling-dependent O(H) scan for
        # every route candidate.
        obstacle_start = failed_trajectory[0]
        obstacle_end = failed_trajectory[-1]
        minimum = float("inf")
        for left, right in zip(points, points[1:]):
            minimum = min(
                minimum,
                _segment_segment_distance(
                    left,
                    right,
                    obstacle_start,
                    obstacle_end,
                ),
            )
        return minimum

    def _inside_arena(self, point: np.ndarray) -> bool:
        x0, x1, y0, y1 = self.config.arena
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    def _candidate_routes(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        failed_trajectory: list[np.ndarray],
    ) -> list[list[np.ndarray]]:
        routes: list[list[np.ndarray]] = [[start, goal]]
        anchor = failed_trajectory[-1]
        for extra in self.config.ring_extra_m:
            radius = self.config.effective_clearance_m + extra
            for index in range(self.config.ring_samples):
                angle = 2.0 * math.pi * index / self.config.ring_samples
                waypoint = anchor + radius * np.asarray(
                    (math.cos(angle), math.sin(angle)), dtype=np.float64
                )
                if self._inside_arena(waypoint):
                    routes.append([start, waypoint, goal])
        return routes

    def _rollout_route(
        self,
        *,
        points: list[np.ndarray],
        initial_velocity: np.ndarray,
        failed_trajectory: list[np.ndarray],
    ) -> _RouteRollout:
        position = points[0].copy()
        velocity = initial_velocity.copy()
        target_index = 1
        minimum_clearance = float(
            np.linalg.norm(position - failed_trajectory[0])
        )
        initial_speed = float(np.linalg.norm(velocity))
        if np.any(
            np.abs(velocity) > self.config.healthy_velocity_limit_mps + 1e-9
        ):
            return _RouteRollout(
                feasible=False,
                completed=False,
                min_clearance_m=minimum_clearance,
                completion_time_s=None,
                terminal_speed_mps=initial_speed,
                rejection_reason="initial_velocity_limit_violation",
            )
        delay_steps = int(math.ceil(self.config.reaction_delay_s / self.config.dt_s))
        execution_fraction = 1.0 - math.exp(
            -self.config.dt_s / self.config.execution_tau_s
        )
        control_scale = (
            execution_fraction * self.config.command_feedforward_tau_s
        )
        steps = int(math.ceil(self.config.rollout_horizon_s / self.config.dt_s))

        for step in range(steps):
            while target_index < len(points) - 1 and float(
                np.linalg.norm(points[target_index] - position)
            ) <= self.config.waypoint_epsilon_m:
                target_index += 1
            target = points[target_index]
            velocity_command = np.clip(
                self.config.goal_gain_s_inv * (target - position),
                -self.config.healthy_velocity_limit_mps,
                self.config.healthy_velocity_limit_mps,
            )
            acceleration = np.zeros(2, dtype=np.float64)
            if step >= delay_steps:
                acceleration = np.clip(
                    self.config.velocity_gain_s_inv * (velocity_command - velocity),
                    -self.config.healthy_acceleration_limit_mps2,
                    self.config.healthy_acceleration_limit_mps2,
                )
            next_velocity = velocity + control_scale * acceleration
            next_speed = float(np.linalg.norm(next_velocity))
            if np.any(
                np.abs(next_velocity)
                > self.config.healthy_velocity_limit_mps + 1e-9
            ):
                return _RouteRollout(
                    feasible=False,
                    completed=False,
                    min_clearance_m=minimum_clearance,
                    completion_time_s=None,
                    terminal_speed_mps=next_speed,
                    rejection_reason="velocity_limit_violation",
                )
            next_position = position + 0.5 * self.config.dt_s * (
                velocity + next_velocity
            )
            if not self._inside_arena(next_position):
                return _RouteRollout(
                    feasible=False,
                    completed=False,
                    min_clearance_m=minimum_clearance,
                    completion_time_s=None,
                    terminal_speed_mps=float(np.linalg.norm(next_velocity)),
                    rejection_reason="rollout_left_arena",
                )
            obstacle_index = min(step, len(failed_trajectory) - 2)
            relative_start = position - failed_trajectory[obstacle_index]
            relative_end = next_position - failed_trajectory[obstacle_index + 1]
            interval_clearance = _segment_distance(
                np.zeros(2, dtype=np.float64), relative_start, relative_end
            )
            minimum_clearance = min(minimum_clearance, interval_clearance)
            position = next_position
            velocity = next_velocity
            if minimum_clearance + 1e-9 < self.config.effective_clearance_m:
                return _RouteRollout(
                    feasible=False,
                    completed=False,
                    min_clearance_m=minimum_clearance,
                    completion_time_s=None,
                    terminal_speed_mps=float(np.linalg.norm(velocity)),
                    rejection_reason="dynamic_clearance_violation",
                )
            if (
                target_index == len(points) - 1
                and float(np.linalg.norm(target - position))
                <= self.config.waypoint_epsilon_m
                and float(np.linalg.norm(velocity))
                <= self.config.terminal_speed_mps
            ):
                return _RouteRollout(
                    feasible=True,
                    completed=True,
                    min_clearance_m=minimum_clearance,
                    completion_time_s=(step + 1) * self.config.dt_s,
                    terminal_speed_mps=float(np.linalg.norm(velocity)),
                    rejection_reason=None,
                )
        return _RouteRollout(
            feasible=False,
            completed=False,
            min_clearance_m=minimum_clearance,
            completion_time_s=None,
            terminal_speed_mps=float(np.linalg.norm(velocity)),
            rejection_reason="rollout_horizon_exhausted",
        )

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
        initial_velocity: np.ndarray,
        goal: np.ndarray,
        failed_trajectory: list[np.ndarray],
        altitude: float,
    ) -> None:
        initial_speed = float(np.linalg.norm(initial_velocity))
        if np.any(
            np.abs(initial_velocity)
            > self.config.healthy_velocity_limit_mps + 1e-9
        ):
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.predicted_terminal_speed_mps = initial_speed
            self.rejection_reason = "initial_velocity_limit_violation"
            return
        goal_clearance = min(
            _segment_distance(goal, left, right)
            for left, right in zip(failed_trajectory, failed_trajectory[1:])
        )
        if goal_clearance < self.config.effective_clearance_m:
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.rejection_reason = "orphan_goal_inside_failed_vehicle_envelope"
            return
        candidates = self._candidate_routes(start, goal, failed_trajectory)
        self.candidate_count = len(candidates)
        geometric_candidates: list[tuple[float, float, list[np.ndarray]]] = []
        for points in candidates:
            clearance = self._route_clearance(points, failed_trajectory)
            length = self._route_length(points)
            if clearance + 1e-9 < self.config.effective_clearance_m:
                self.geometric_rejection_count += 1
                continue
            if length > self.config.max_route_length_m:
                self.route_length_rejection_count += 1
                continue
            geometric_candidates.append((length, clearance, points))

        # Dynamic feasibility is evaluated in increasing route length order.
        # Once a route is feasible, every later route is longer and cannot win
        # the primary deterministic ranking; routes at the same length remain
        # eligible for completion-time and clearance tie breaking.
        geometric_candidates.sort(key=lambda item: item[0])
        admissible: list[
            tuple[float, float, float, list[np.ndarray], _RouteRollout]
        ] = []
        for length, _geometric_clearance, points in geometric_candidates:
            if admissible and length > admissible[0][0] + 1e-9:
                break
            self.rollout_candidate_count += 1
            rollout = self._rollout_route(
                points=points,
                initial_velocity=initial_velocity,
                failed_trajectory=failed_trajectory,
            )
            if rollout.feasible:
                assert rollout.completion_time_s is not None
                admissible.append(
                    (
                        length,
                        rollout.completion_time_s,
                        -rollout.min_clearance_m,
                        points,
                        rollout,
                    )
                )
            else:
                self.dynamic_rejection_count += 1
                reason = rollout.rejection_reason or "unknown"
                self.dynamic_rejection_reasons[reason] = (
                    self.dynamic_rejection_reasons.get(reason, 0) + 1
                )
        if not admissible:
            self.state = "rejected_hold"
            self.rejection_count += 1
            self.rejection_reason = "no_dynamically_recoverable_candidate"
            return
        _, _, negative_clearance, selected, rollout = min(
            admissible, key=lambda item: (item[0], item[1], item[2])
        )
        self.geometric_min_clearance_m = self._route_clearance(
            selected, failed_trajectory
        )
        self.predicted_min_clearance_m = -negative_clearance
        self.predicted_completion_time_s = rollout.completion_time_s
        self.predicted_terminal_speed_mps = rollout.terminal_speed_mps
        if (
            self.predicted_min_clearance_m + 1e-9
            < self.config.effective_clearance_m
        ):
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
            started = time.perf_counter()
            self._admit(
                start=np.asarray(healthy_snapshot.position[:2], dtype=np.float64),
                initial_velocity=np.asarray(
                    healthy_snapshot.velocity[:2]
                    if healthy_snapshot.velocity is not None
                    else (0.0, 0.0),
                    dtype=np.float64,
                ),
                goal=np.asarray(base_goals[failed][:2], dtype=np.float64),
                failed_trajectory=self._failed_trajectory(snapshots[failed]),
                altitude=healthy_snapshot.position[2],
            )
            self.decision_latency_ms = 1_000.0 * (time.perf_counter() - started)

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
            "effective_clearance_threshold_m": self.config.effective_clearance_m,
            "tracking_error_buffer_m": self.config.tracking_error_buffer_m,
            "healthy_velocity_limit_mps": self.config.healthy_velocity_limit_mps,
            "healthy_acceleration_limit_mps2": (
                self.config.healthy_acceleration_limit_mps2
            ),
            "terminal_speed_limit_mps": self.config.terminal_speed_mps,
            "conservative_braking_accel_mps2": (
                self.config.conservative_braking_accel_mps2
            ),
            "geometric_min_clearance_m": self.geometric_min_clearance_m,
            "predicted_completion_time_s": self.predicted_completion_time_s,
            "predicted_terminal_speed_mps": self.predicted_terminal_speed_mps,
            "dynamic_rejection_count": self.dynamic_rejection_count,
            "rollout_candidate_count": self.rollout_candidate_count,
            "geometric_rejection_count": self.geometric_rejection_count,
            "route_length_rejection_count": self.route_length_rejection_count,
            "dynamic_rejection_reasons": dict(self.dynamic_rejection_reasons),
            "decision_latency_ms": self.decision_latency_ms,
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
