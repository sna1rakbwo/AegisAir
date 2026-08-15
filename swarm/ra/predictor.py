"""Short-horizon predictive safety monitor (plan.md sections 17-18).

Constant-velocity extrapolation predicts the future pairwise distance and
normalized margin; the closest-point-of-approach (CPA) formula provides the
analytic minimum separation.  Prediction is only for early warning; it never
replaces the hard CBF safety filter.
"""

from __future__ import annotations

from dataclasses import dataclass

from swarm.geometry import Vector3, dot, norm, scale, subtract


def constant_velocity_position(p: Vector3, v: Vector3, tau: float) -> Vector3:
    return (p[0] + v[0] * tau, p[1] + v[1] * tau, p[2] + v[2] * tau)


def predicted_distance(
    p_i: Vector3,
    p_j: Vector3,
    v_i: Vector3,
    v_j: Vector3,
    tau: float,
) -> float:
    p_hat_i = constant_velocity_position(p_i, v_i, tau)
    p_hat_j = constant_velocity_position(p_j, v_j, tau)
    return norm(subtract(p_hat_i, p_hat_j))


@dataclass(frozen=True)
class CpaResult:
    t_cpa: float
    d_cpa: float


def closest_point_of_approach(
    p_i: Vector3,
    p_j: Vector3,
    v_i: Vector3,
    v_j: Vector3,
    horizon: float,
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
    """Return (min predicted rho over the horizon, time to that minimum)."""
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
