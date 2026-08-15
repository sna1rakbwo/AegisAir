"""Dynamic / perception / communication safety margins (plan.md sections 11-14).

All geometry is 3D FLU.  ``position``/``velocity`` use the ``swarm.geometry``
``Vector3`` convention and the pairwise direction is ``n = (p_i - p_j) / d``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from swarm.geometry import Vector3, dot, norm, subtract


@dataclass(frozen=True)
class RuntimeAssuranceParams:
    """Frozen parameters for the risk-adaptive safety boundary."""

    d0: float = 0.5
    tau_r: float = 0.3
    a_eff: float = 2.0
    beta: float = 3.0
    v_max: float = 2.0
    a_max: float = 3.0
    alpha: float = 1.0
    ema_lambda: float = 0.8
    degradation_dt: float = 0.1
    prediction_horizon: float = 3.0
    rho_pred_threshold: float = 0.0
    q_pred: float = 0.0

    def __post_init__(self) -> None:
        if self.d0 < 0 or self.a_eff <= 0 or self.v_max < 0 or self.a_max < 0:
            raise ValueError("invalid RuntimeAssuranceParams")
        if not 0.0 <= self.ema_lambda <= 1.0:
            raise ValueError("ema_lambda must be in [0, 1]")


def unit_direction(p_i: Vector3, p_j: Vector3) -> Vector3:
    """Return n = (p_i - p_j) / ||p_i - p_j||, or a zero vector if coincident."""
    delta = subtract(p_i, p_j)
    length = norm(delta)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (delta[0] / length, delta[1] / length, delta[2] / length)


def closing_speed(p_i: Vector3, p_j: Vector3, v_i: Vector3, v_j: Vector3) -> float:
    """Non-negative rate at which the pair distance is shrinking."""
    delta = subtract(p_i, p_j)
    distance = norm(delta)
    if distance == 0:
        return 0.0
    relative_velocity = subtract(v_i, v_j)
    return max(0.0, -dot(delta, relative_velocity) / distance)


def dynamics_margin(closing_speed: float, params: RuntimeAssuranceParams) -> float:
    """M_dyn = v_cl * tau_r + v_cl^2 / (2 a_eff)."""
    return closing_speed * params.tau_r + (closing_speed**2) / (2.0 * params.a_eff)


def perception_margin(
    sigma_i: float,
    sigma_j: float,
    params: RuntimeAssuranceParams,
) -> float:
    """M_perc = beta * sqrt(n^T (sigma_i^2 I + sigma_j^2 I) n).

    v1 uses isotropic scalar position uncertainty; full covariance can be
    added later without changing the public interface.
    """

    sigma_total_sq = sigma_i**2 + sigma_j**2
    return params.beta * math.sqrt(sigma_total_sq)


def communication_margin(aoi: float, params: RuntimeAssuranceParams) -> float:
    """M_comm = v_max * AoI + 0.5 * a_max * AoI^2."""
    aoi = max(0.0, aoi)
    return params.v_max * aoi + 0.5 * params.a_max * aoi**2


def dynamic_safety_boundary(
    *,
    closing_speed: float,
    perception_sigma_i: float,
    perception_sigma_j: float,
    aoi: float,
    params: RuntimeAssuranceParams,
) -> float:
    """Combined d_safe = d0 + M_dyn + M_perc + M_comm."""

    return (
        params.d0
        + dynamics_margin(closing_speed, params)
        + perception_margin(perception_sigma_i, perception_sigma_j, params)
        + communication_margin(aoi, params)
    )
