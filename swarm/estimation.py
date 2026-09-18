"""Shared-state estimator for Phase 6 fault injection.

The estimator sits between raw shared telemetry and Runtime Assurance.  It turns
perfect shared ground truth into delayed, dropout-prone, covariance-bearing
estimates so that ``perception noise / latency / stale telemetry`` become
configuration rather than scattered code changes.

Mapping to the existing margins (docs/decisions/phase6_shared_state_estimator.md):

- covariance -> ``perception_margin`` (sigma projected onto the pair LOS)
- delay -> estimator outputs an older state; ``M_comm`` keeps growing with age
- dropout -> hold the last estimate and inflate covariance over time
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from swarm.geometry import Vector3
from swarm.safety import DroneSnapshot


@dataclass(frozen=True)
class SharedStateEstimatorConfig:
    """Frozen estimator parameters."""

    measurement_cov_m2: float = 0.01
    process_noise_m2_per_s: float = 0.02
    dropout_rate: float = 0.0
    delay_ms: int = 0
    max_cov_m2: float = 1.0
    history_len: int = 256

    def __post_init__(self) -> None:
        if self.measurement_cov_m2 < 0:
            raise ValueError("measurement_cov_m2 must be non-negative")
        if self.process_noise_m2_per_s < 0:
            raise ValueError("process_noise_m2_per_s must be non-negative")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if self.max_cov_m2 < self.measurement_cov_m2:
            raise ValueError("max_cov_m2 must be >= measurement_cov_m2")
        if self.history_len < 1:
            raise ValueError("history_len must be positive")


@dataclass(frozen=True)
class EstimatedState:
    """Covariance-bearing per-drone estimate consumed by Runtime Assurance."""

    drone_id: int
    position: Vector3
    velocity: Vector3
    covariance: tuple[float, float, float, float]
    timestamp_ms: int
    dropped: bool


class SharedStateEstimator:
    """Deterministic per-drone shared-state estimator with delay/dropout/noise."""

    def __init__(self, config: SharedStateEstimatorConfig | None = None) -> None:
        self.config = config or SharedStateEstimatorConfig()
        self._history: dict[int, deque[tuple[int, Vector3, Vector3]]] = {}
        self._last: dict[int, tuple[Vector3, Vector3, int, bool]] = {}
        self.dropped_total = 0

    def step(
        self,
        raw: dict[int, DroneSnapshot],
        now_ms: int,
        rng: Any | None = None,
    ) -> dict[int, EstimatedState]:
        """Advance the estimator one control tick and return estimated states."""
        out: dict[int, EstimatedState] = {}
        for drone_id, snap in raw.items():
            position = snap.position
            velocity = snap.velocity or (0.0, 0.0, 0.0)
            history = self._history.setdefault(
                drone_id, deque(maxlen=self.config.history_len)
            )

            dropped = rng is not None and rng.random() < self.config.dropout_rate
            if dropped:
                self.dropped_total += 1
            else:
                history.append((now_ms, position, velocity))

            if drone_id not in self._last:
                self._last[drone_id] = (position, velocity, now_ms, dropped)
            elif not dropped:
                self._last[drone_id] = (position, velocity, now_ms, False)
            else:
                old_pos, old_vel, old_rx, _ = self._last[drone_id]
                self._last[drone_id] = (old_pos, old_vel, old_rx, True)

            target_ts = now_ms - self.config.delay_ms
            sel_pos, sel_vel, sel_ts = self._last[drone_id][:3]
            for ts, pos, vel in reversed(history):
                if ts <= target_ts:
                    sel_pos, sel_vel, sel_ts = pos, vel, ts
                    break

            dt_rx_s = max(0.0, (now_ms - self._last[drone_id][2]) / 1000.0)
            cov = min(
                self.config.measurement_cov_m2
                + self.config.process_noise_m2_per_s * dt_rx_s,
                self.config.max_cov_m2,
            )
            out[drone_id] = EstimatedState(
                drone_id=drone_id,
                position=sel_pos,
                velocity=sel_vel,
                covariance=(cov, 0.0, 0.0, cov),
                timestamp_ms=sel_ts,
                dropped=self._last[drone_id][3],
            )
        return out

    def reset(self) -> None:
        self._history.clear()
        self._last.clear()
        self.dropped_total = 0
