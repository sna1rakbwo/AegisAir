"""Predictive safety monitor (upgraded first version).

Implements the plan's P0-P2 core:
    - constant-velocity (CV) and filtered constant-acceleration (CA) state
      prediction;
    - future covariance growth with horizon;
    - future dynamic safety boundary and future normalized margin;
    - rho_min_pred, tau*, TTSB, and a heuristic prediction confidence.

Prediction is advisory only; hard safety remains with the CBF filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from swarm.geometry import Vector3, dot, norm, subtract


def constant_velocity_position(p: Vector3, v: Vector3, tau: float) -> Vector3:
    return (p[0] + v[0] * tau, p[1] + v[1] * tau, p[2] + v[2] * tau)


def predicted_distance(
    p_i: Vector3, p_j: Vector3, v_i: Vector3, v_j: Vector3, tau: float
) -> float:
    p_hat_i = constant_velocity_position(p_i, v_i, tau)
    p_hat_j = constant_velocity_position(p_j, v_j, tau)
    return norm(subtract(p_hat_i, p_hat_j))


@dataclass(frozen=True)
class CpaResult:
    t_cpa: float
    d_cpa: float


def closest_point_of_approach(
    p_i: Vector3, p_j: Vector3, v_i: Vector3, v_j: Vector3, horizon: float
) -> CpaResult:
    r = subtract(p_i, p_j)
    v = subtract(v_i, v_j)
    vv = dot(v, v)
    if vv == 0:
        return CpaResult(t_cpa=0.0, d_cpa=norm(r))
    t_cpa = -dot(r, v) / vv
    t_cpa = max(0.0, min(horizon, t_cpa))
    d_cpa = norm((r[0] + v[0] * t_cpa, r[1] + v[1] * t_cpa, r[2] + v[2] * t_cpa))
    return CpaResult(t_cpa=t_cpa, d_cpa=d_cpa)


def predicted_min_margin(
    *,
    p_i: Vector3,
    p_j: Vector3,
    v_i: Vector3,
    v_j: Vector3,
    d_safe: float,
    horizon: float,
    steps: int = 24,
) -> tuple[float, float]:
    """Simple CV predicted minimum margin (kept as the P1 baseline)."""
    best_rho = float("inf")
    best_tau = 0.0
    for step in range(steps + 1):
        tau = horizon * step / steps
        d_hat = predicted_distance(p_i, p_j, v_i, v_j, tau)
        rho_hat = (d_hat - d_safe) / d_safe
        if rho_hat < best_rho:
            best_rho = rho_hat
            best_tau = tau
    return best_rho, best_tau


@dataclass(frozen=True)
class PredictionResult:
    model: str
    rho_min_pred: float
    tau_star: float
    ttsb: float | None
    predicted_min_distance: float
    predicted_safe_distance: float
    confidence: float


class AccelerationFilter:
    """EMA-smoothed acceleration estimate from velocity history."""

    def __init__(self, lambda_a: float = 0.8) -> None:
        self.lambda_a = lambda_a
        self.ema = 0.0
        self.prev_velocity: Vector3 | None = None
        self.recent: list[float] = []

    def update(self, velocity: Vector3, dt: float) -> float:
        """Return the smoothed acceleration magnitude estimate."""
        if self.prev_velocity is None or dt <= 0:
            a = 0.0
        else:
            delta = subtract(velocity, self.prev_velocity)
            a = norm(delta) / dt
        self.ema = self.lambda_a * self.ema + (1.0 - self.lambda_a) * a
        self.prev_velocity = velocity
        self.recent.append(a)
        self.recent = self.recent[-32:]
        return self.ema

    def variability(self) -> float:
        if len(self.recent) < 2:
            return 0.0
        mean = sum(self.recent) / len(self.recent)
        var = sum((x - mean) ** 2 for x in self.recent) / len(self.recent)
        return math.sqrt(var)


def _position(t: float, p: Vector3, v: Vector3, a: float, direction: Vector3) -> Vector3:
    """Filtered-CA position: p + v*t + 0.5*a*t^2 along the acceleration direction."""
    if a == 0.0:
        return (p[0] + v[0] * t, p[1] + v[1] * t, p[2] + v[2] * t)
    scale = 0.5 * a * t * t
    return (p[0] + v[0] * t + direction[0] * scale,
            p[1] + v[1] * t + direction[1] * scale,
            p[2] + v[2] * t + direction[2] * scale)


def _velocity(t: float, v: Vector3, a: float, direction: Vector3) -> Vector3:
    if a == 0.0:
        return v
    scale = a * t
    return (v[0] + direction[0] * scale, v[1] + direction[1] * scale, v[2] + direction[2] * scale)


class PredictiveMonitor:
    """Pairwise future-margin predictor."""

    def __init__(self, lambda_a: float = 0.8, q_pred: float = 0.0) -> None:
        self.lambda_a = lambda_a
        self.q_pred = q_pred
        self.acc_filters: dict[int, AccelerationFilter] = {}

    def _acc_filter(self, agent_id: int) -> AccelerationFilter:
        if agent_id not in self.acc_filters:
            self.acc_filters[agent_id] = AccelerationFilter(self.lambda_a)
        return self.acc_filters[agent_id]

    def update_acceleration(self, agent_id: int, velocity: Vector3, dt: float) -> float:
        return self._acc_filter(agent_id).update(velocity, dt)

    def predict(
        self,
        *,
        agent_i: int,
        agent_j: int,
        p_i: Vector3,
        p_j: Vector3,
        v_i: Vector3,
        v_j: Vector3,
        sigma_i: float,
        sigma_j: float,
        aoi: float,
        params,
        horizon: float,
        steps: int = 30,
    ) -> PredictionResult:
        """Return the worst predicted margin, time of that margin, and TTSB."""
        a_i = self._acc_filter(agent_i).ema
        a_j = self._acc_filter(agent_j).ema
        dir_i = (1.0, 0.0, 0.0)
        dir_j = (1.0, 0.0, 0.0)

        best_rho = float("inf")
        best_tau = 0.0
        best_distance = float("inf")
        best_d_safe = 0.0
        ttsb: float | None = None

        for step in range(steps + 1):
            tau = horizon * step / steps
            p_hat_i = _position(tau, p_i, v_i, a_i, dir_i)
            p_hat_j = _position(tau, p_j, v_j, a_j, dir_j)
            v_hat_i = _velocity(tau, v_i, a_i, dir_i)
            v_hat_j = _velocity(tau, v_j, a_j, dir_j)

            r = subtract(p_hat_i, p_hat_j)
            distance = norm(r)
            if distance == 0:
                rho_hat = -1.0
            else:
                n = (r[0] / distance, r[1] / distance, r[2] / distance)
                rel_v = subtract(v_hat_i, v_hat_j)
                v_cl = max(0.0, -dot(r, rel_v) / distance)

                # Future covariance: sigma^2(tau) = sigma^2 + q*tau.
                sigma_sq_i = sigma_i**2 + self.q_pred * tau
                sigma_sq_j = sigma_j**2 + self.q_pred * tau
                m_perc = params.beta * math.sqrt(sigma_sq_i + sigma_sq_j)
                m_dyn = v_cl * params.tau_r + v_cl**2 / (2.0 * params.a_eff)
                # Communication margin only grows when the link is actually
                # stale.  With healthy telemetry (AoI == 0) the worst-case
                # "no new telemetry" growth is far too conservative.
                if aoi > 0:
                    future_aoi = aoi + tau
                    m_comm = params.v_max * future_aoi + 0.5 * params.a_max * future_aoi**2
                else:
                    m_comm = 0.0
                d_safe_hat = params.d0 + m_dyn + m_perc + m_comm

                rho_hat = (distance - d_safe_hat) / d_safe_hat
                if rho_hat < best_rho:
                    best_rho = rho_hat
                    best_tau = tau
                    best_distance = distance
                    best_d_safe = d_safe_hat

            if ttsb is None and rho_hat <= 0.0:
                ttsb = tau

        sigma_a = max(self._acc_filter(agent_i).variability(), self._acc_filter(agent_j).variability())
        confidence = math.exp(
            -0.2 * best_tau
            - 0.5 * sigma_a
            - 0.5 * aoi
            - 1.0 * max(sigma_i, sigma_j)
        )
        return PredictionResult(
            model="FILTERED_CA",
            rho_min_pred=best_rho,
            tau_star=best_tau,
            ttsb=ttsb,
            predicted_min_distance=best_distance,
            predicted_safe_distance=best_d_safe,
            confidence=confidence,
        )
