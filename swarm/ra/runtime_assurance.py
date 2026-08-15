"""Risk-Adaptive Runtime Assurance orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.ra.cbf import cbf_constraint, project_safe_action
from swarm.ra.margin import PairMarginTracker, normalized_margin
from swarm.ra.margins import (
    RuntimeAssuranceParams,
    closing_speed,
    dynamic_safety_boundary,
)
from swarm.ra.predictor import PredictiveMonitor


@dataclass
class FilterResult:
    drone: int
    mode: str
    nominal_action: tuple[float, float]
    safe_action: tuple[float, float]
    safety_margin: float
    predicted_margin: float
    time_to_min_margin: float
    time_to_safety_boundary: float | None
    prediction_confidence: float
    recovery_buffer: float | None
    semantic_recovery_feasible: bool
    degradation: float
    worst_pair: int | None
    intervened: bool
    proactive: bool


class RuntimeAssurance:
    """Filter nominal MARL velocity commands through the CBF safety set."""

    def __init__(
        self,
        params: RuntimeAssuranceParams | None = None,
        v_max: float = 2.0,
        perception_sigma: float = 0.1,
        rho_warn: float = 0.2,
    ) -> None:
        self.params = params or RuntimeAssuranceParams()
        self.v_max = v_max
        self.perception_sigma = perception_sigma
        self.rho_warn = rho_warn
        self.trackers: dict[tuple[int, int], PairMarginTracker] = {}
        self.monitor = PredictiveMonitor(q_pred=self.params.q_pred)
        self._last_t: float | None = None

    def _tracker(self, i: int, j: int) -> PairMarginTracker:
        key = (min(i, j), max(i, j))
        if key not in self.trackers:
            self.trackers[key] = PairMarginTracker(self.params)
        return self.trackers[key]

    def filter(
        self,
        snapshots: dict[int, object],
        nominal_actions: dict[int, np.ndarray],
        t: float,
        aoi: dict[tuple[int, int], float] | None = None,
    ) -> dict[int, FilterResult]:
        aoi = aoi or {}
        results: dict[int, FilterResult] = {}

        dt = self.params.degradation_dt
        if self._last_t is not None:
            dt = max(1e-3, t - self._last_t)
        self._last_t = t
        for agent_id, snapshot in snapshots.items():
            if snapshot.velocity is not None:
                self.monitor.update_acceleration(agent_id, snapshot.velocity, dt)

        for drone_id, snapshot in snapshots.items():
            p_i = np.asarray(snapshot.position[:2], dtype=np.float64)
            v_i = np.asarray(snapshot.velocity[:2], dtype=np.float64) if snapshot.velocity else np.zeros(2)
            u_nom = np.asarray(nominal_actions[drone_id], dtype=np.float64)

            constraints: list[tuple[np.ndarray, float]] = []
            worst_rho = float("inf")
            worst_g = 0.0
            worst_pair: int | None = None
            worst_pred_rho = float("inf")
            worst_pred_tau = 0.0
            worst_ttsb: float | None = None
            worst_confidence = 1.0

            for other_id, other in snapshots.items():
                if other_id == drone_id:
                    continue
                p_j = np.asarray(other.position[:2], dtype=np.float64)
                v_j = np.asarray(other.velocity[:2], dtype=np.float64) if other.velocity else np.zeros(2)

                distance = float(np.linalg.norm(p_i - p_j))
                v_cl = closing_speed(snapshot.position, other.position, snapshot.velocity or (0.0, 0.0, 0.0), other.velocity or (0.0, 0.0, 0.0))
                pair_aoi = aoi.get((drone_id, other_id), 0.0)
                d_safe = dynamic_safety_boundary(
                    closing_speed=v_cl,
                    perception_sigma_i=self.perception_sigma,
                    perception_sigma_j=self.perception_sigma,
                    aoi=pair_aoi,
                    params=self.params,
                )
                rho = normalized_margin(distance, d_safe)
                tracker = self._tracker(drone_id, other_id)
                g = tracker.update(rho, t)

                a, b = cbf_constraint(p_i, p_j, v_j, d_safe, self.params.alpha)
                constraints.append((a, b))

                if rho < worst_rho:
                    worst_rho = rho
                    worst_pair = other_id
                    worst_g = g

                pred = self.monitor.predict(
                    agent_i=drone_id,
                    agent_j=other_id,
                    p_i=snapshot.position,
                    p_j=other.position,
                    v_i=snapshot.velocity or (0.0, 0.0, 0.0),
                    v_j=other.velocity or (0.0, 0.0, 0.0),
                    sigma_i=self.perception_sigma,
                    sigma_j=self.perception_sigma,
                    aoi=pair_aoi,
                    params=self.params,
                    horizon=self.params.prediction_horizon,
                )
                if pred.rho_min_pred < worst_pred_rho:
                    worst_pred_rho = pred.rho_min_pred
                    worst_pred_tau = pred.tau_star
                    worst_ttsb = pred.ttsb
                    worst_confidence = pred.confidence

            u_safe = project_safe_action(u_nom, constraints, self.v_max)
            intervened = bool(np.linalg.norm(u_safe - u_nom) > 1e-6)

            if intervened:
                mode = "override"
            elif worst_rho < self.rho_warn:
                mode = "warning"
            else:
                mode = "normal"

            results[drone_id] = FilterResult(
                drone=drone_id,
                mode=mode,
                nominal_action=(float(u_nom[0]), float(u_nom[1])),
                safe_action=(float(u_safe[0]), float(u_safe[1])),
                safety_margin=float(worst_rho),
                predicted_margin=float(worst_pred_rho),
                time_to_min_margin=float(worst_pred_tau),
                time_to_safety_boundary=worst_ttsb,
                prediction_confidence=float(worst_confidence),
                recovery_buffer=(
                    worst_ttsb - self.params.estimated_recovery_latency
                    if worst_ttsb is not None
                    else None
                ),
                semantic_recovery_feasible=(
                    worst_ttsb is not None
                    and worst_ttsb > self.params.estimated_recovery_latency
                ),
                degradation=float(worst_g),
                worst_pair=worst_pair,
                intervened=intervened,
                proactive=worst_pred_rho < self.params.rho_pred_threshold,
            )
        return results
