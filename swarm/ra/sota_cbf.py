"""用于公平对比的近期 CBF 基线适配。

实现保持 AegisAir 的集中式多机 QP、加速度 box 和确定性 infeasible fallback，
只替换 barrier 约束：

* ZOCBF：下一采样时刻的零阶 barrier 差分约束；
* PB-CBF：有限制动力下的径向 prediction-based barrier。
"""

from __future__ import annotations

import numpy as np

from swarm.ra.projection import (
    project_box as _project_box,
    project_box_and_single_halfspace as _project_box_and_single_halfspace,
    project_halfspace as _project_halfspace,
)


def _projection_residual(
    x: np.ndarray,
    halfspaces: list[tuple[np.ndarray, float]],
    a_max: float,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
) -> float:
    """Maximum box, fixed-input, or half-space violation."""
    box_violation = float(np.max(np.abs(x[~fixed_mask]) - a_max, initial=-np.inf))
    fixed_violation = float(
        np.max(np.abs(x[fixed_mask] - fixed_values[fixed_mask]), initial=0.0)
    )
    halfspace_violation = max(
        (b - float(np.dot(c, x)) for c, b in halfspaces), default=-np.inf
    )
    return max(0.0, box_violation, fixed_violation, halfspace_violation)


def _solve_projection_qp(
    *,
    x_nom: np.ndarray,
    halfspaces: list[tuple[np.ndarray, float]],
    a_max: float,
    max_iters: int,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, int]:
    """Project onto control bounds, fixed inputs, and barrier half-spaces."""
    if fixed_mask is None:
        fixed_mask = np.zeros_like(x_nom, dtype=bool)
        fixed_values = np.zeros_like(x_nom)
    elif fixed_values is None or fixed_mask.shape != x_nom.shape or fixed_values.shape != x_nom.shape:
        raise ValueError("fixed coordinate arrays must match x_nom")
    x_nom = x_nom.copy()
    x_nom[fixed_mask] = fixed_values[fixed_mask]
    if len(halfspaces) == 1:
        c, b = halfspaces[0]
        return _project_box_and_single_halfspace(
            x_nom=x_nom,
            c=c,
            b=b,
            a_max=a_max,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
    corrections = [np.zeros_like(x_nom) for _ in range(len(halfspaces) + 1)]
    x = x_nom.copy()
    used = 0
    for used in range(1, max_iters + 1):
        previous = x.copy()
        current = x.copy()
        correction_change = 0.0
        for index in range(len(halfspaces) + 1):
            shifted = current + corrections[index]
            if index == 0:
                projected = _project_box(
                    shifted,
                    a_max,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                )
            else:
                c, b = halfspaces[index - 1]
                projected = _project_halfspace(shifted, c, b)
            correction = shifted - projected
            correction_change = max(
                correction_change,
                float(np.linalg.norm(correction - corrections[index])),
            )
            corrections[index] = correction
            current = projected
        x = current
        # A stationary iterate can be transient while Dykstra duals change.
        if (
            float(np.linalg.norm(x - previous)) <= 1e-9
            and correction_change <= 1e-9
            and _projection_residual(
                x, halfspaces, a_max, fixed_mask, fixed_values
            ) <= 1e-6
        ):
            break
    feasible = _projection_residual(x, halfspaces, a_max, fixed_mask, fixed_values) <= 1e-6
    return x, feasible, used


def _pack_nominal(
    a_nom: dict[int, np.ndarray], drone_ids: list[int]
) -> np.ndarray:
    return np.concatenate([np.asarray(a_nom[drone], dtype=np.float64) for drone in drone_ids])


def _fixed_coordinates(
    *,
    drone_ids: list[int],
    fixed_accelerations: dict[int, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    fixed = {
        drone: np.asarray(value, dtype=np.float64)
        for drone, value in (fixed_accelerations or {}).items()
    }
    unknown = set(fixed) - set(drone_ids)
    if unknown:
        raise ValueError(f"fixed acceleration contains unknown drones: {sorted(unknown)}")
    mask = np.zeros(2 * len(drone_ids), dtype=bool)
    values = np.zeros(2 * len(drone_ids), dtype=np.float64)
    for index, drone in enumerate(drone_ids):
        if drone not in fixed:
            continue
        if fixed[drone].shape != (2,) or not np.all(np.isfinite(fixed[drone])):
            raise ValueError("fixed accelerations must be finite planar vectors")
        mask[2 * index : 2 * index + 2] = True
        values[2 * index : 2 * index + 2] = fixed[drone]
    return mask, values, fixed


def _unpack_or_brake(
    *,
    x: np.ndarray,
    feasible: bool,
    drone_ids: list[int],
    velocities: dict[int, np.ndarray],
    a_max: float,
    infeasible_fallback: str,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    if infeasible_fallback not in {"velocity_cancel", "max_brake"}:
        raise ValueError("unknown SOTA-CBF infeasible fallback")
    output = {}
    fixed_accelerations = fixed_accelerations or {}
    for index, drone in enumerate(drone_ids):
        if drone in fixed_accelerations:
            output[drone] = np.asarray(
                fixed_accelerations[drone], dtype=np.float64
            ).copy()
            continue
        if feasible:
            output[drone] = x[2 * index : 2 * index + 2].copy()
            continue
        velocity = np.asarray(velocities[drone], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if infeasible_fallback == "max_brake" and speed > 1e-9:
            output[drone] = -a_max * velocity / speed
        else:
            output[drone] = np.clip(-velocity, -a_max, a_max)
    return output


def solve_zocbf_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    s_now: dict[tuple[int, int], float],
    s_next: dict[tuple[int, int], float],
    dt: float,
    gamma: float,
    delta: float,
    a_max: float,
    beta: dict[int, float],
    command_scale: float | None = None,
    max_iters: int = 3000,
    infeasible_fallback: str = "velocity_cancel",
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """集中式 ZOCBF 一步 QP。

    使用当前视线方向的局部安全半空间 ``h=n^T r-s``。在 PX4 一阶速度
    跟踪的精确 ZOH 位置模型下施加
    ``h(k+1) >= (1-gamma) h(k) + delta``。
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("ZOCBF gamma must be in (0, 1]")
    if delta < 0.0:
        raise ValueError("ZOCBF delta must be non-negative")
    drone_ids = sorted(positions)
    scale = dt if command_scale is None else command_scale
    if scale <= 0.0:
        raise ValueError("command_scale must be positive")
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    x_nom = _pack_nominal(a_nom, drone_ids)
    fixed_mask, fixed_values, fixed = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )
    halfspaces = []
    for (i, j), distance_now in sorted(s_now.items()):
        r = np.asarray(positions[i]) - np.asarray(positions[j])
        v = np.asarray(velocities[i]) - np.asarray(velocities[j])
        distance = float(np.linalg.norm(r))
        if distance <= 1e-9:
            # No instantaneous acceleration can restore a separation direction.
            # Make the numerical infeasibility explicit so the caller fails closed.
            halfspaces.append((np.zeros_like(x_nom), 1.0))
            continue
        n = r / distance if distance > 1e-9 else np.array([1.0, 0.0])
        h_now = float(np.dot(n, r)) - distance_now
        c = np.zeros_like(x_nom)
        c[2 * index[i] : 2 * index[i] + 2] = scale * beta[i] * n
        c[2 * index[j] : 2 * index[j] + 2] = -scale * beta[j] * n
        b = (
            (1.0 - gamma) * h_now
            + delta
            - float(np.dot(n, r + dt * v))
            + s_next[(i, j)]
        )
        halfspaces.append((c, b))
    x, feasible, iterations = _solve_projection_qp(
        x_nom=x_nom,
        halfspaces=halfspaces,
        a_max=a_max,
        max_iters=max_iters,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return (
        _unpack_or_brake(
            x=x,
            feasible=feasible,
            drone_ids=drone_ids,
            velocities=velocities,
            a_max=a_max,
            infeasible_fallback=infeasible_fallback,
            fixed_accelerations=fixed,
        ),
        feasible,
        iterations,
    )


def solve_prediction_based_cbf_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    static_distance: dict[tuple[int, int], float],
    alpha: float,
    braking_accel: float,
    a_max: float,
    max_iters: int = 3000,
    infeasible_fallback: str = "velocity_cancel",
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """有限制动力的径向 PB-CBF QP。

    对每一对飞行器使用论文双积分示例的径向构造
    ``h_p = d-D-q^2/(2*mu)``（仅在 ``q<0`` 的接近阶段计制动项），其中
    ``q=n^T(v_i-v_j)``。约束 ``h_dot+alpha*h>=0`` 对相对加速度为线性。
    """
    if braking_accel <= 0.0:
        raise ValueError("PB-CBF braking_accel must be positive")
    if alpha <= 0.0:
        raise ValueError("PB-CBF alpha must be positive")
    drone_ids = sorted(positions)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    x_nom = _pack_nominal(a_nom, drone_ids)
    fixed_mask, fixed_values, fixed = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )
    halfspaces = []
    for (i, j), safe_distance in sorted(static_distance.items()):
        r = np.asarray(positions[i]) - np.asarray(positions[j])
        v = np.asarray(velocities[i]) - np.asarray(velocities[j])
        distance = float(np.linalg.norm(r))
        n = r / distance if distance > 1e-9 else np.array([1.0, 0.0])
        radial_velocity = float(np.dot(n, v))
        tangential_rate = (
            (float(np.dot(v, v)) - radial_velocity**2) / distance
            if distance > 1e-9
            else 0.0
        )
        closing = min(radial_velocity, 0.0)
        h_pb = distance - safe_distance - closing**2 / (2.0 * braking_accel)
        c = np.zeros_like(x_nom)
        if closing < 0.0:
            coefficient = -closing / braking_accel
            c[2 * index[i] : 2 * index[i] + 2] = coefficient * n
            c[2 * index[j] : 2 * index[j] + 2] = -coefficient * n
            drift = radial_velocity + coefficient * tangential_rate
            b = -alpha * h_pb - drift
            halfspaces.append((c, b))
        elif radial_velocity + alpha * h_pb < 0.0:
            # This branch is only reachable at a degenerate coincident state;
            # keep the fail-closed infeasibility explicit.
            halfspaces.append((c, 1.0))
    x, feasible, iterations = _solve_projection_qp(
        x_nom=x_nom,
        halfspaces=halfspaces,
        a_max=a_max,
        max_iters=max_iters,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return (
        _unpack_or_brake(
            x=x,
            feasible=feasible,
            drone_ids=drone_ids,
            velocities=velocities,
            a_max=a_max,
            infeasible_fallback=infeasible_fallback,
            fixed_accelerations=fixed,
        ),
        feasible,
        iterations,
    )


def minimum_prediction_based_constraint_slack(
    *,
    accelerations: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    static_distance: dict[tuple[int, int], float],
    alpha: float,
    braking_accel: float,
    a_max: float,
    box_constrained_drones: set[int] | None = None,
) -> float:
    """Evaluate actual joint acceleration against radial PB-CBF and box."""
    if braking_accel <= 0.0 or alpha <= 0.0:
        raise ValueError("PB-CBF parameters must be positive")
    drone_ids = sorted(positions)
    if set(accelerations) != set(drone_ids):
        raise ValueError("accelerations must cover every constrained drone")
    packed = np.concatenate(
        [np.asarray(accelerations[drone], dtype=np.float64) for drone in drone_ids]
    )
    if packed.shape != (2 * len(drone_ids),) or not np.all(np.isfinite(packed)):
        raise ValueError("accelerations must be finite planar vectors")
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    constrained = (
        set(drone_ids)
        if box_constrained_drones is None
        else set(box_constrained_drones)
    )
    if not constrained <= set(drone_ids):
        raise ValueError("box constraints contain an unknown drone")
    slacks = [
        float(
            a_max
            - max(
                np.max(np.abs(accelerations[drone]))
                for drone in constrained
            )
        )
    ] if constrained else []
    for (i, j), safe_distance in sorted(static_distance.items()):
        r = np.asarray(positions[i]) - np.asarray(positions[j])
        relative_velocity = np.asarray(velocities[i]) - np.asarray(velocities[j])
        distance = float(np.linalg.norm(r))
        direction = r / distance if distance > 1e-9 else np.array([1.0, 0.0])
        radial_velocity = float(np.dot(direction, relative_velocity))
        tangential_rate = (
            (
                float(np.dot(relative_velocity, relative_velocity))
                - radial_velocity**2
            )
            / distance
            if distance > 1e-9
            else 0.0
        )
        closing = min(radial_velocity, 0.0)
        barrier = distance - safe_distance - closing**2 / (2.0 * braking_accel)
        if closing < 0.0:
            coefficient = -closing / braking_accel
            relative_acceleration = (
                packed[2 * index[i] : 2 * index[i] + 2]
                - packed[2 * index[j] : 2 * index[j] + 2]
            )
            lhs = coefficient * float(np.dot(direction, relative_acceleration))
            drift = radial_velocity + coefficient * tangential_rate
            bound = -alpha * barrier - drift
            slacks.append(lhs - bound)
        elif radial_velocity + alpha * barrier < 0.0:
            slacks.append(-1.0)
    return min(slacks, default=float("inf"))
