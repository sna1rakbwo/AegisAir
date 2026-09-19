"""用于公平对比的近期 CBF 基线适配。

实现保持 AegisAir 的集中式多机 QP、加速度 box 和确定性 infeasible fallback，
只替换 barrier 约束：

* ZOCBF：下一采样时刻的零阶 barrier 差分约束；
* PB-CBF：有限制动力下的径向 prediction-based barrier。
"""

from __future__ import annotations

import numpy as np


def _project_box(x: np.ndarray, a_max: float) -> np.ndarray:
    return np.clip(x, -a_max, a_max)


def _project_halfspace(x: np.ndarray, c: np.ndarray, b: float) -> np.ndarray:
    residual = b - float(np.dot(c, x))
    if residual > 1e-9:
        norm_sq = float(np.dot(c, c))
        if norm_sq > 1e-12:
            return x + residual * c / norm_sq
    return x


def _solve_projection_qp(
    *,
    x_nom: np.ndarray,
    halfspaces: list[tuple[np.ndarray, float]],
    a_max: float,
    max_iters: int,
) -> tuple[np.ndarray, bool, int]:
    """Project onto a box intersected with linear barrier half-spaces."""
    corrections = [np.zeros_like(x_nom) for _ in range(len(halfspaces) + 1)]
    x = x_nom.copy()
    used = 0
    for used in range(1, max_iters + 1):
        previous = x.copy()
        previous_corrections = [correction.copy() for correction in corrections]
        # Dykstra corrections are defined relative to the point produced by
        # the preceding complete sweep.  Restarting from ``x_nom`` here
        # invalidates those corrections and can report a nonempty
        # box/half-space intersection as infeasible.
        current = x
        for index in range(len(halfspaces) + 1):
            shifted = current + corrections[index]
            if index == 0:
                projected = _project_box(shifted, a_max)
            else:
                c, b = halfspaces[index - 1]
                projected = _project_halfspace(shifted, c, b)
            corrections[index] = shifted - projected
            current = projected
        x = current
        # The primal point can be temporarily unchanged while Dykstra's
        # correction terms still evolve.  Stopping on the primal difference
        # alone can therefore return a feasible but non-optimal projection.
        primal_change = float(np.linalg.norm(x - previous))
        correction_change = max(
            float(np.linalg.norm(correction - old))
            for correction, old in zip(corrections, previous_corrections)
        )
        if primal_change <= 1e-9 and correction_change <= 1e-9:
            break
    feasible = bool(np.all(np.abs(x) <= a_max + 1e-6))
    feasible = feasible and all(float(np.dot(c, x)) >= b - 1e-6 for c, b in halfspaces)
    return x, feasible, used


def _pack_nominal(
    a_nom: dict[int, np.ndarray], drone_ids: list[int]
) -> np.ndarray:
    return np.concatenate([np.asarray(a_nom[drone], dtype=np.float64) for drone in drone_ids])


def _unpack_or_brake(
    *,
    x: np.ndarray,
    feasible: bool,
    drone_ids: list[int],
    velocities: dict[int, np.ndarray],
    a_max: float,
    infeasible_fallback: str,
) -> dict[int, np.ndarray]:
    if infeasible_fallback not in {"velocity_cancel", "max_brake"}:
        raise ValueError("unknown SOTA-CBF infeasible fallback")
    output = {}
    for index, drone in enumerate(drone_ids):
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
    halfspaces = []
    for (i, j), distance_now in sorted(s_now.items()):
        r = np.asarray(positions[i]) - np.asarray(positions[j])
        v = np.asarray(velocities[i]) - np.asarray(velocities[j])
        distance = float(np.linalg.norm(r))
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
        x_nom=x_nom, halfspaces=halfspaces, a_max=a_max, max_iters=max_iters
    )
    return (
        _unpack_or_brake(
            x=x,
            feasible=feasible,
            drone_ids=drone_ids,
            velocities=velocities,
            a_max=a_max,
            infeasible_fallback=infeasible_fallback,
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
        x_nom=x_nom, halfspaces=halfspaces, a_max=a_max, max_iters=max_iters
    )
    return (
        _unpack_or_brake(
            x=x,
            feasible=feasible,
            drone_ids=drone_ids,
            velocities=velocities,
            a_max=a_max,
            infeasible_fallback=infeasible_fallback,
        ),
        feasible,
        iterations,
    )
