"""Execution-model conformance monitor and deterministic backup controller.

The sampled-data barrier is only as trustworthy as the short-horizon execution
model used by the filter.  This module treats that model as an untrusted
component: it checks the previously issued velocity command against the next
measured PX4 state and latches a deterministic backup when the calibrated
residual envelope, QP feasibility, solver deadline, or telemetry-age contract
is violated.

Thresholds are deliberately supplied by a frozen calibration artifact.  The
module does not learn or relax safety thresholds online.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TelemetryFreshnessConfig:
    """Input-age contract for commands issued by the central controller."""

    max_age_s: float = 0.15
    release_samples: int = 3

    def __post_init__(self) -> None:
        if self.max_age_s <= 0.0:
            raise ValueError("max telemetry age must be positive")
        if self.release_samples < 1:
            raise ValueError("release samples must be positive")


@dataclass(frozen=True)
class TelemetryFreshnessDecision:
    """Whether normal control may use the current set of input samples."""

    fresh: bool
    active: bool
    reasons: tuple[str, ...]
    stale_drones: tuple[int, ...]
    missing_drones: tuple[int, ...]
    consecutive_fresh: int
    newly_tripped: bool = False
    newly_released: bool = False


class TelemetryFreshnessGate:
    """Immediately blocks stale inputs and releases with fresh-sample hysteresis."""

    def __init__(self, config: TelemetryFreshnessConfig | None = None) -> None:
        self.config = config or TelemetryFreshnessConfig()
        self.active = False
        self.consecutive_fresh = 0
        self.trip_count = 0
        self.release_count = 0
        self.active_steps = 0
        self.stale_steps = 0
        self.max_observed_age_s = 0.0
        self.reason_counts: dict[str, int] = {}

    def assess(
        self,
        *,
        required_drones: tuple[int, ...] | list[int],
        telemetry_ages_s: Mapping[int, float],
    ) -> TelemetryFreshnessDecision:
        required = tuple(sorted(required_drones))
        missing = tuple(drone for drone in required if drone not in telemetry_ages_s)
        stale = tuple(
            drone
            for drone in required
            if drone in telemetry_ages_s
            and (
                not math.isfinite(float(telemetry_ages_s[drone]))
                or float(telemetry_ages_s[drone]) > self.config.max_age_s
            )
        )
        finite_ages = [
            float(telemetry_ages_s[drone])
            for drone in required
            if drone in telemetry_ages_s
            and math.isfinite(float(telemetry_ages_s[drone]))
        ]
        if finite_ages:
            self.max_observed_age_s = max(self.max_observed_age_s, max(finite_ages))

        reasons: list[str] = []
        if missing:
            reasons.append("TELEMETRY_MISSING")
        if stale:
            reasons.append("TELEMETRY_STALE")
        fresh = not reasons
        newly_tripped = False
        newly_released = False
        if not fresh:
            self.stale_steps += 1
            self.consecutive_fresh = 0
            for reason in reasons:
                self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            if not self.active:
                self.active = True
                self.trip_count += 1
                newly_tripped = True
        elif self.active:
            self.consecutive_fresh += 1
            if self.consecutive_fresh >= self.config.release_samples:
                self.active = False
                self.release_count += 1
                newly_released = True
                reasons.append("TELEMETRY_FRESH")
            else:
                reasons.append("FRESHNESS_HYSTERESIS")
        else:
            self.consecutive_fresh = 0
            reasons.append("TELEMETRY_FRESH")

        if self.active:
            self.active_steps += 1
        return TelemetryFreshnessDecision(
            fresh=fresh,
            active=self.active,
            reasons=tuple(reasons),
            stale_drones=stale,
            missing_drones=missing,
            consecutive_fresh=self.consecutive_fresh,
            newly_tripped=newly_tripped,
            newly_released=newly_released,
        )


@dataclass(frozen=True)
class ExecutionSupervisorConfig:
    """Frozen C2-prime runtime contract.

    ``velocity_residual_limit_mps`` and ``position_residual_limit_m`` must come
    from calibration data disjoint from validation/final trials.
    """

    tau_s: float
    velocity_residual_limit_mps: float
    position_residual_limit_m: float
    max_telemetry_age_s: float = 0.15
    solve_deadline_s: float = 0.05
    trip_samples: int = 2
    release_samples: int = 10
    backup_speed_mps: float = 0.6
    enable_residual_gate: bool = True
    enable_qp_gate: bool = True
    enable_predictive_gate: bool = False
    safe_distance_m: float = 1.6
    response_delay_s: float = 0.30
    braking_deceleration_mps2: float = 2.0
    recoverability_buffer_m: float = 0.20

    def __post_init__(self) -> None:
        if self.tau_s < 0.0:
            raise ValueError("tau_s must be non-negative")
        if self.velocity_residual_limit_mps <= 0.0:
            raise ValueError("velocity residual limit must be positive")
        if self.position_residual_limit_m <= 0.0:
            raise ValueError("position residual limit must be positive")
        if self.max_telemetry_age_s <= 0.0:
            raise ValueError("max telemetry age must be positive")
        if self.solve_deadline_s <= 0.0:
            raise ValueError("solve deadline must be positive")
        if self.trip_samples < 1 or self.release_samples < 1:
            raise ValueError("trip/release samples must be positive")
        if self.backup_speed_mps <= 0.0:
            raise ValueError("backup speed must be positive")
        if self.safe_distance_m <= 0.0:
            raise ValueError("safe distance must be positive")
        if self.response_delay_s < 0.0:
            raise ValueError("response delay must be non-negative")
        if self.braking_deceleration_mps2 <= 0.0:
            raise ValueError("braking deceleration must be positive")
        if self.recoverability_buffer_m < 0.0:
            raise ValueError("recoverability buffer must be non-negative")


@dataclass(frozen=True)
class ExecutionResidual:
    drone: int
    dt_s: float
    velocity_mps: float
    position_m: float


@dataclass(frozen=True)
class RecoverabilityPair:
    """Conservative pairwise stopping-distance reserve for C2 predictive gate."""

    drones: tuple[int, int]
    separation_m: float
    closing_speed_mps: float
    required_distance_m: float
    margin_m: float


@dataclass(frozen=True)
class ExecutionSupervisorDecision:
    active: bool
    reasons: tuple[str, ...]
    residuals: dict[int, ExecutionResidual]
    consecutive_bad: int
    consecutive_clean: int
    recoverability: tuple[RecoverabilityPair, ...] = ()
    backup_drones: tuple[int, ...] = ()
    newly_tripped: bool = False
    newly_released: bool = False


@dataclass
class _IssuedCommand:
    position: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    timestamp_ms: int
    issued_t_s: float


class ExecutionConformanceSupervisor:
    """Stateful conformance monitor with latched, hysteretic backup switching."""

    def __init__(self, config: ExecutionSupervisorConfig) -> None:
        self.config = config
        self._issued: dict[int, _IssuedCommand] = {}
        self.active = False
        self.consecutive_bad = 0
        self.consecutive_clean = 0
        self.trip_count = 0
        self.release_count = 0
        self.active_steps = 0
        self.reason_counts: dict[str, int] = {}
        self._backup_drones: tuple[int, ...] = ()

    def observe(
        self,
        snapshots: Mapping[int, object],
        *,
        now_t_s: float,
    ) -> dict[int, ExecutionResidual]:
        """Compare the last command prediction with the newly observed state."""
        residuals: dict[int, ExecutionResidual] = {}
        for drone, snap in snapshots.items():
            previous = self._issued.get(drone)
            velocity_raw = getattr(snap, "velocity", None)
            if previous is None or velocity_raw is None:
                continue
            timestamp_ms = int(getattr(snap, "timestamp_ms", 0))
            timestamp_dt = (timestamp_ms - previous.timestamp_ms) / 1000.0
            wall_dt = now_t_s - previous.issued_t_s
            dt = timestamp_dt if timestamp_dt > 1e-4 else wall_dt
            if dt <= 1e-4:
                continue
            # A delayed sample can span more than one control period.  The
            # exact ZOH prediction remains well defined for the measured dt.
            tau = self.config.tau_s
            alpha = 1.0 - math.exp(-dt / tau) if tau > 0.0 else 1.0
            beta = dt - tau * alpha if tau > 0.0 else dt
            delta_u = previous.command - previous.velocity
            predicted_v = previous.velocity + alpha * delta_u
            predicted_p = previous.position + previous.velocity * dt + beta * delta_u
            observed_v = np.asarray(velocity_raw[:2], dtype=np.float64)
            observed_p = np.asarray(getattr(snap, "position")[:2], dtype=np.float64)
            residuals[drone] = ExecutionResidual(
                drone=drone,
                dt_s=float(dt),
                velocity_mps=float(np.linalg.norm(observed_v - predicted_v)),
                position_m=float(np.linalg.norm(observed_p - predicted_p)),
            )
        return residuals

    def assess(
        self,
        *,
        residuals: Mapping[int, ExecutionResidual],
        qp_feasible: bool,
        solve_elapsed_s: float,
        telemetry_ages_s: Mapping[int, float],
        snapshots: Mapping[int, object] | None = None,
    ) -> ExecutionSupervisorDecision:
        reasons: list[str] = []
        recoverability = self.recoverability(snapshots or {})
        predictive_pairs = tuple(pair for pair in recoverability if pair.margin_m <= 0.0)
        backup_drones: tuple[int, ...] = ()
        if self.config.enable_predictive_gate and predictive_pairs:
            reasons.append("PREDICTIVE_RECOVERABILITY")
            # Deterministic right-of-way: the higher numeric ID yields.  The
            # other vehicle retains RA-filtered motion, allowing a safe pass
            # rather than having both vehicles retreat into a deadlock.
            backup_drones = tuple(sorted({max(pair.drones) for pair in predictive_pairs}))
            self._backup_drones = backup_drones
        if self.config.enable_residual_gate:
            if any(
                r.velocity_mps > self.config.velocity_residual_limit_mps
                for r in residuals.values()
            ):
                reasons.append("VELOCITY_RESIDUAL")
            if any(
                r.position_m > self.config.position_residual_limit_m
                for r in residuals.values()
            ):
                reasons.append("POSITION_RESIDUAL")
        if self.config.enable_qp_gate and not qp_feasible:
            reasons.append("QP_INFEASIBLE")
        if solve_elapsed_s > self.config.solve_deadline_s:
            reasons.append("SOLVE_DEADLINE")
        if any(age > self.config.max_telemetry_age_s for age in telemetry_ages_s.values()):
            reasons.append("TELEMETRY_STALE")

        newly_tripped = False
        newly_released = False
        if reasons:
            self.consecutive_bad += 1
            self.consecutive_clean = 0
            for reason in set(reasons):
                self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            if not self.active and self.consecutive_bad >= self.config.trip_samples:
                self.active = True
                self.trip_count += 1
                newly_tripped = True
        else:
            self.consecutive_bad = 0
            self.consecutive_clean += 1
            if self.active and self.consecutive_clean >= self.config.release_samples:
                self.active = False
                self.release_count += 1
                newly_released = True
                self._backup_drones = ()

        if self.active:
            self.active_steps += 1
            if not reasons:
                reasons.append("HYSTERESIS_HOLD")
        elif not reasons:
            reasons.append("MODEL_CONFORMANT" if residuals else "NO_HISTORY")

        selective_backup = ()
        if self.active and self._backup_drones and set(reasons).issubset(
            {"PREDICTIVE_RECOVERABILITY", "HYSTERESIS_HOLD"}
        ):
            selective_backup = self._backup_drones

        return ExecutionSupervisorDecision(
            active=self.active,
            reasons=tuple(reasons),
            residuals=dict(residuals),
            consecutive_bad=self.consecutive_bad,
            consecutive_clean=self.consecutive_clean,
            recoverability=tuple(recoverability),
            backup_drones=selective_backup if self.active else backup_drones,
            newly_tripped=newly_tripped,
            newly_released=newly_released,
        )

    def recoverability(self, snapshots: Mapping[int, object]) -> tuple[RecoverabilityPair, ...]:
        """Return frozen stopping-distance reserves for all observed pairs.

        The reserve budgets the relative closing during the response delay and
        a one-sided effective braking deceleration.  It is deliberately
        conservative: only the yielding UAV is required to brake, so it does
        not credit unverified cooperative braking by the right-of-way UAV.
        """
        if not self.config.enable_predictive_gate:
            return ()
        pairs: list[RecoverabilityPair] = []
        ids = sorted(snapshots)
        for index, first in enumerate(ids):
            for second in ids[index + 1 :]:
                a = snapshots[first]
                b = snapshots[second]
                pa = np.asarray(getattr(a, "position")[:2], dtype=np.float64)
                pb = np.asarray(getattr(b, "position")[:2], dtype=np.float64)
                delta = pb - pa
                separation = float(np.linalg.norm(delta))
                if separation <= 1e-9:
                    closing = 2.0 * self.config.backup_speed_mps
                else:
                    unit = delta / separation
                    va = np.asarray(getattr(a, "velocity")[:2], dtype=np.float64)
                    vb = np.asarray(getattr(b, "velocity")[:2], dtype=np.float64)
                    closing = max(0.0, float(np.dot(va - vb, unit)))
                required = (
                    self.config.safe_distance_m
                    + self.config.recoverability_buffer_m
                    + closing * self.config.response_delay_s
                    + closing * closing / (2.0 * self.config.braking_deceleration_mps2)
                )
                pairs.append(
                    RecoverabilityPair(
                        drones=(first, second),
                        separation_m=separation,
                        closing_speed_mps=closing,
                        required_distance_m=required,
                        margin_m=separation - required,
                    )
                )
        return tuple(pairs)

    def record_commands(
        self,
        snapshots: Mapping[int, object],
        commands: Mapping[int, np.ndarray | tuple[float, float]],
        *,
        now_t_s: float,
    ) -> None:
        for drone, command in commands.items():
            snap = snapshots[drone]
            velocity = getattr(snap, "velocity", None)
            if velocity is None:
                continue
            self._issued[drone] = _IssuedCommand(
                position=np.asarray(getattr(snap, "position")[:2], dtype=np.float64),
                velocity=np.asarray(velocity[:2], dtype=np.float64),
                command=np.asarray(command[:2], dtype=np.float64),
                timestamp_ms=int(getattr(snap, "timestamp_ms", 0)),
                issued_t_s=float(now_t_s),
            )


def deterministic_backup_velocity(
    *,
    drone: int,
    snapshots: Mapping[int, object],
    dt_s: float,
    a_max: float,
    v_max: float,
    retreat_speed_mps: float,
) -> np.ndarray:
    """Acceleration-limited retreat away from the other-agent centroid.

    For a head-on encounter this first applies maximum available braking and
    then continues toward the already-used safe-holding direction.  The norm
    of the per-step velocity change is bounded by ``a_max * dt_s``.
    """
    snap = snapshots[drone]
    position = np.asarray(getattr(snap, "position")[:2], dtype=np.float64)
    velocity_raw = getattr(snap, "velocity", None)
    velocity = np.asarray(
        velocity_raw[:2] if velocity_raw is not None else (0.0, 0.0),
        dtype=np.float64,
    )
    others = [
        np.asarray(getattr(other, "position")[:2], dtype=np.float64)
        for other_id, other in snapshots.items()
        if other_id != drone
    ]
    centroid = np.mean(others, axis=0) if others else np.zeros(2)
    outward = position - centroid
    norm = float(np.linalg.norm(outward))
    if norm <= 1e-9:
        outward = -velocity
        norm = float(np.linalg.norm(outward))
    if norm <= 1e-9:
        outward = np.array([1.0 if drone % 2 == 0 else -1.0, 0.0])
        norm = 1.0
    desired = outward / norm * min(retreat_speed_mps, v_max)
    delta = desired - velocity
    max_delta = max(0.0, a_max * dt_s)
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_delta > 0.0:
        delta *= max_delta / delta_norm
    command = velocity + delta
    speed = float(np.linalg.norm(command))
    if speed > v_max:
        command *= v_max / speed
    return command
