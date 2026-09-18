"""Deterministic C1 PX4 admission and brake-latch supervisor.

This layer is intentionally independent from the sampled-data barrier.  It
prevents a delayed-peer estimator from admitting a symmetric head-on crossing
before its history is mature, serializes the crossing through a lateral staging
point, and latches a zero-velocity command when the barrier becomes marginal or
infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from swarm.ra.margins import RuntimeAssuranceParams, dynamic_safety_boundary
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class C1Px4SupervisorConfig:
    estimator_delay_s: float = 0.3
    warmup_extra_cycles: int = 2
    right_of_way_drone: int = 2
    barrier_tau_px4_s: float = 0.2
    admission_execution_tau_s: float = 0.7
    command_feedforward_tau_s: float = 0.7
    latch_enter_rho: float = 0.2
    latch_release_rho: float = 0.5
    latch_release_cycles: int = 5
    stage_tolerance_m: float = 0.35
    stage_speed_tolerance_mps: float = 0.25
    stage_clearance_m: float = 0.2
    arena: tuple[float, float, float, float] = (-6.0, 6.0, -6.0, 6.0)

    def __post_init__(self) -> None:
        if self.estimator_delay_s < 0.0:
            raise ValueError("estimator_delay_s must be non-negative")
        if self.warmup_extra_cycles < 0:
            raise ValueError("warmup_extra_cycles must be non-negative")
        if self.admission_execution_tau_s <= 0.0:
            raise ValueError("admission_execution_tau_s must be positive")
        if self.barrier_tau_px4_s < 0.0:
            raise ValueError("barrier_tau_px4_s must be non-negative")
        if self.command_feedforward_tau_s <= 0.0:
            raise ValueError("command_feedforward_tau_s must be positive")
        if self.latch_release_rho <= self.latch_enter_rho:
            raise ValueError("latch release rho must exceed enter rho")
        if self.latch_release_cycles < 1:
            raise ValueError("latch_release_cycles must be positive")


class C1Px4AdmissionSupervisor:
    """State machine for a two-UAV delayed-peer head-on mission."""

    def __init__(
        self,
        *,
        config: C1Px4SupervisorConfig,
        drone_ids: list[int],
        base_goals: dict[int, tuple[float, float, float]],
        reset_starts: dict[int, tuple[float, float, float]],
        rate_hz: float,
        ra_params: RuntimeAssuranceParams,
        v_max: float,
    ) -> None:
        if len(drone_ids) != 2:
            raise ValueError("C1 PX4 admission supervisor requires exactly two UAVs")
        if config.right_of_way_drone not in drone_ids:
            raise ValueError("right_of_way_drone must be in drone_ids")
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.config = config
        self.drone_ids = tuple(drone_ids)
        self.primary = config.right_of_way_drone
        self.yielder = next(i for i in drone_ids if i != self.primary)
        self.base_goals = dict(base_goals)
        self.reset_starts = dict(reset_starts)
        self.warmup_steps = (
            math.ceil(config.estimator_delay_s * rate_hz)
            + config.warmup_extra_cycles
        )
        # During admission only one UAV is allowed to translate.  The staging
        # clearance therefore uses one-UAV maximum closing speed, the full
        # estimator AoI margin, and the empirically identified execution lag.
        admission_params = RuntimeAssuranceParams(
            **{
                **ra_params.__dict__,
                "tau_ctrl": ra_params.tau_ctrl
                + config.admission_execution_tau_s,
            }
        )
        self.required_stage_clearance_m = dynamic_safety_boundary(
            closing_speed=v_max,
            perception_sigma_i=0.0,
            perception_sigma_j=0.0,
            aoi=config.estimator_delay_s,
            params=admission_params,
        ) + config.stage_clearance_m
        start = reset_starts[self.yielder]
        ymin, ymax = config.arena[2], config.arena[3]
        positive_room = ymax - start[1]
        negative_room = start[1] - ymin
        sign = 1.0 if positive_room >= negative_room else -1.0
        stage_y = start[1] + sign * self.required_stage_clearance_m
        if not ymin <= stage_y <= ymax:
            raise ValueError(
                "arena cannot fit the execution-aware C1 staging clearance"
            )
        self.stage_goal = (start[0], stage_y, start[2])
        primary_goal = base_goals[self.primary]
        clear_y = primary_goal[1] - sign * self.required_stage_clearance_m
        if not ymin <= clear_y <= ymax:
            raise ValueError(
                "arena cannot fit the execution-aware C1 primary clearance"
            )
        self.primary_clear_goal = (primary_goal[0], clear_y, primary_goal[2])
        self.phase = "WARMUP"
        self.brake_latched = False
        self.clean_cycles = 0
        self.latch_trip_count = 0
        self.latch_release_count = 0
        self.phase_changes = 0

    @staticmethod
    def _speed(snapshot: DroneSnapshot) -> float:
        return float(np.linalg.norm(snapshot.velocity or (0.0, 0.0, 0.0)))

    def _settled(
        self,
        snapshot: DroneSnapshot,
        target: tuple[float, float, float],
        tolerance_m: float,
    ) -> bool:
        return (
            float(
                np.linalg.norm(
                    np.asarray(snapshot.position[:2]) - np.asarray(target[:2])
                )
            )
            <= tolerance_m
            and self._speed(snapshot) <= self.config.stage_speed_tolerance_mps
        )

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_changes += 1

    def directives(
        self,
        *,
        step: int,
        snapshots: dict[int, DroneSnapshot],
        goal_epsilon: float,
    ) -> tuple[dict[int, tuple[float, float, float]], dict[int, float]]:
        """Return deterministic goal overrides and velocity scales."""
        if self.phase == "WARMUP" and step >= self.warmup_steps:
            self._set_phase("STAGE_YIELDER")
        if self.phase == "STAGE_YIELDER" and self._settled(
            snapshots[self.yielder], self.stage_goal, self.config.stage_tolerance_m
        ):
            self._set_phase("PASS_PRIMARY")
        if self.phase == "PASS_PRIMARY" and self._settled(
            snapshots[self.primary], self.base_goals[self.primary], goal_epsilon
        ):
            self._set_phase("CLEAR_PRIMARY")
        if self.phase == "CLEAR_PRIMARY" and self._settled(
            snapshots[self.primary],
            self.primary_clear_goal,
            self.config.stage_tolerance_m,
        ):
            self._set_phase("PASS_YIELDER")
        if self.phase == "PASS_YIELDER" and self._settled(
            snapshots[self.yielder], self.base_goals[self.yielder], goal_epsilon
        ):
            self._set_phase("RETURN_PRIMARY")
        if self.phase == "RETURN_PRIMARY" and self._settled(
            snapshots[self.primary], self.base_goals[self.primary], goal_epsilon
        ):
            self._set_phase("COMPLETE")

        goals: dict[int, tuple[float, float, float]] = {}
        scales = {i: 0.0 for i in self.drone_ids}
        if self.phase == "STAGE_YIELDER":
            goals[self.yielder] = self.stage_goal
            scales[self.yielder] = 1.0
        elif self.phase == "PASS_PRIMARY":
            scales[self.primary] = 1.0
            goals[self.yielder] = self.stage_goal
        elif self.phase == "CLEAR_PRIMARY":
            goals[self.primary] = self.primary_clear_goal
            goals[self.yielder] = self.stage_goal
            scales[self.primary] = 1.0
        elif self.phase == "PASS_YIELDER":
            goals[self.primary] = self.primary_clear_goal
            scales[self.yielder] = 1.0
        elif self.phase == "RETURN_PRIMARY":
            scales[self.primary] = 1.0
        return goals, scales

    def observe_ra(self, results: dict[int, Any]) -> None:
        """Trip/release the global brake latch with explicit hysteresis."""
        feasible = all(result.feasible is not False for result in results.values())
        min_rho = min(float(result.safety_margin) for result in results.values())
        bad = (not feasible) or min_rho <= self.config.latch_enter_rho
        clean = feasible and min_rho >= self.config.latch_release_rho
        if bad and not self.brake_latched:
            self.brake_latched = True
            self.clean_cycles = 0
            self.latch_trip_count += 1
        elif self.brake_latched:
            self.clean_cycles = self.clean_cycles + 1 if clean else 0
            if self.clean_cycles >= self.config.latch_release_cycles:
                self.brake_latched = False
                self.clean_cycles = 0
                self.latch_release_count += 1

    @property
    def complete(self) -> bool:
        return self.phase == "COMPLETE"

    def summary(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "complete": self.complete,
            "warmup_steps": self.warmup_steps,
            "stage_goal": list(self.stage_goal),
            "primary_clear_goal": list(self.primary_clear_goal),
            "required_stage_clearance_m": self.required_stage_clearance_m,
            "brake_latched_at_end": self.brake_latched,
            "latch_trip_count": self.latch_trip_count,
            "latch_release_count": self.latch_release_count,
            "phase_changes": self.phase_changes,
            "barrier_tau_px4_s": self.config.barrier_tau_px4_s,
            "command_feedforward_tau_s": self.config.command_feedforward_tau_s,
            "admission_execution_tau_s": self.config.admission_execution_tau_s,
        }
