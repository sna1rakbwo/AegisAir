"""Huang et al. (ECC 2025) PCBF 的最小、确定性双机适配。

该模块实现论文中安全 MPC 的核心结构：预测域中的软状态约束、硬输入
约束和硬终端不变集合；每周期仅执行最优控制序列第一项。它刻意独立于
HOCBF/PB-CBF，供 Gate 0 离线复现使用，尚未接入 PX4 runner。

为保持无额外求解器依赖，优化器采用 projected subgradient：对当前 pair
line-of-sight 的线性内近似 ``n^T(p_i-p_j) >= d`` 建模。该近似是欧氏距离
安全约束的保守充分条件；每次重规划都会更新 ``n``。终端集合为
``n^T(p_i-p_j) >= d_terminal`` 且相对投影速度不再接近，零加速度局部控制
器保持该终端集不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCBFConfig:
    """冻结前仅允许在 calibration 中选择的 PCBF 数值参数。"""

    horizon: int = 20
    dt: float = 0.05
    # ``a`` is converted into a realized one-sample velocity increment by this
    # frozen interface gain. Gate 0 uses physical double-integrator scaling
    # (dt); live adaptation supplies the PX4 first-order realized gain.
    control_scale_s: float | None = None
    a_max: float = 2.0
    terminal_buffer_m: float = 0.10
    terminal_velocity_tolerance_mps: float = 0.0
    slack_weight: float = 20.0
    tracking_weight: float = 0.05
    velocity_tracking_weight: float = 0.50
    iterations: int = 1200
    step_size: float = 0.03
    projection_iterations: int = 400
    tolerance: float = 1e-6
    lateral_candidates: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("PCBF horizon must be at least 2")
        if self.dt <= 0.0 or self.a_max <= 0.0:
            raise ValueError("PCBF dt and a_max must be positive")
        if self.control_scale_s is not None and self.control_scale_s <= 0.0:
            raise ValueError("PCBF control_scale_s must be positive")
        if self.terminal_buffer_m < 0.0:
            raise ValueError("PCBF terminal buffer must be non-negative")
        if self.terminal_velocity_tolerance_mps < 0.0:
            raise ValueError("PCBF terminal velocity tolerance must be non-negative")
        if (
            self.slack_weight <= 0.0
            or self.tracking_weight <= 0.0
            or self.velocity_tracking_weight <= 0.0
        ):
            raise ValueError("PCBF objective weights must be positive")
        if self.iterations < 1 or self.projection_iterations < 1:
            raise ValueError("PCBF iteration counts must be positive")
        if self.step_size <= 0.0 or self.tolerance <= 0.0:
            raise ValueError("PCBF step size and tolerance must be positive")


@dataclass(frozen=True)
class PCBFResult:
    accelerations: dict[int, np.ndarray]
    feasible: bool
    terminal_feasible: bool
    value: float
    slack_sum: float
    iterations: int
    status: str
    fail_closed_reason: str | None
    plan: np.ndarray | None


def _solve_candidate_pcbf(
    *,
    nominal: np.ndarray,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    drone_ids: list[int],
    config: PCBFConfig,
    fixed_accelerations: dict[int, np.ndarray],
) -> PCBFResult:
    """Fast nonlinear PCBF adaptation for the two-UAV comparison.

    The published PCBF formulation permits nonlinear dynamics/constraints.
    Here the finite-horizon optimization is solved over a deterministic bank
    of nominal and left/right avoidance sequences.  Every candidate is scored
    with the actual Euclidean distance constraint and a terminal bounded-
    braking recoverability condition.  This removes the invalid fixed-ray
    approximation while keeping runtime bounded and auditable.
    """
    if len(drone_ids) != 2 or len(safe_distances) != 1:
        raise ValueError("candidate PCBF currently supports the frozen two-UAV C1 comparison")
    i, j = next(iter(safe_distances))
    distance_safe = float(safe_distances[(i, j)])
    relative = positions[i] - positions[j]
    distance = float(np.linalg.norm(relative))
    if distance <= 1e-9:
        return PCBFResult(
            accelerations=_brake(
                velocities, config.a_max, fixed_accelerations
            ), feasible=False,
            terminal_feasible=False, value=float("inf"), slack_sum=float("inf"),
            iterations=0, status="fail_closed", fail_closed_reason="coincident_pair", plan=None,
        )
    radial = relative / distance
    tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    controlled = [drone for drone in drone_ids if drone not in fixed_accelerations]
    candidates: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        for magnitude in config.lateral_candidates:
            plan = np.repeat(nominal[None, :, :], config.horizon, axis=0)
            if magnitude > 0.0:
                if i in controlled:
                    plan[:, index[i]] += sign * magnitude * tangent
                if j in controlled:
                    plan[:, index[j]] -= sign * magnitude * tangent
            plan = np.clip(plan, -config.a_max, config.a_max)
            for drone, acceleration in fixed_accelerations.items():
                plan[:, index[drone]] = acceleration
            candidates.append(plan)

    best: tuple[float, float, float, np.ndarray] | None = None
    for plan in candidates:
        predicted_p, predicted_v = _predict(
            plan, positions=positions, velocities=velocities,
            drone_ids=drone_ids, config=config,
        )
        violations = []
        for step in range(1, config.horizon + 1):
            predicted_distance = float(np.linalg.norm(predicted_p[i][step] - predicted_p[j][step]))
            violations.append(max(0.0, distance_safe - predicted_distance))
        terminal_r = predicted_p[i][-1] - predicted_p[j][-1]
        terminal_v = predicted_v[i][-1] - predicted_v[j][-1]
        terminal_distance = float(np.linalg.norm(terminal_r))
        terminal_n = terminal_r / terminal_distance if terminal_distance > 1e-9 else radial
        terminal_closing = max(0.0, -float(terminal_n @ terminal_v))
        relative_braking = config.a_max * sum(
            drone in controlled for drone in (i, j)
        )
        recovery_margin = (
            terminal_distance
            - distance_safe
            - (
                terminal_closing**2 / (2.0 * relative_braking)
                if relative_braking > 0.0
                else (float("inf") if terminal_closing > 0.0 else 0.0)
            )
        )
        terminal_deficit = max(0.0, config.terminal_buffer_m - recovery_margin)
        slack_sum = float(sum(violations))
        tracking_cost = float(np.sum((plan - nominal[None, :, :]) ** 2))
        score = (
            config.slack_weight * 1000.0 * slack_sum
            + config.slack_weight * 1000.0 * terminal_deficit
            + config.tracking_weight * tracking_cost
        )
        candidate = (score, slack_sum, terminal_deficit, plan)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    score, slack_sum, terminal_deficit, plan = best
    terminal_feasible = terminal_deficit <= config.tolerance
    if not terminal_feasible:
        return PCBFResult(
            accelerations=_brake(
                velocities, config.a_max, fixed_accelerations
            ), feasible=False,
            terminal_feasible=False, value=score, slack_sum=slack_sum,
            iterations=len(candidates), status="fail_closed",
            fail_closed_reason="terminal_recovery_infeasible", plan=plan,
        )
    return PCBFResult(
        accelerations={drone: plan[0, index[drone]].copy() for drone in drone_ids},
        feasible=True, terminal_feasible=True, value=score,
        slack_sum=slack_sum, iterations=len(candidates),
        status="optimal_approx", fail_closed_reason=None, plan=plan,
    )


def _brake(
    velocities: dict[int, np.ndarray],
    a_max: float,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    output = {}
    fixed_accelerations = fixed_accelerations or {}
    for drone, velocity in velocities.items():
        if drone in fixed_accelerations:
            output[drone] = fixed_accelerations[drone].copy()
            continue
        v = np.asarray(velocity, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        output[drone] = (-a_max * v / speed) if speed > 1e-9 else np.zeros(2)
    return output


def _control_coefficient(
    step: int, control_step: int, dt: float, control_scale_s: float
) -> float:
    """Coefficient from ``a_control_step`` to position at ``step``."""
    return (
        dt * control_scale_s * (step - control_step - 0.5)
        if control_step < step
        else 0.0
    )


def _predict(
    plan: np.ndarray,
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    drone_ids: list[int],
    config: PCBFConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return predicted positions/velocities with shape ``(H+1, 2)``."""
    predicted_p: dict[int, np.ndarray] = {}
    predicted_v: dict[int, np.ndarray] = {}
    for index, drone in enumerate(drone_ids):
        p = np.zeros((config.horizon + 1, 2), dtype=np.float64)
        v = np.zeros((config.horizon + 1, 2), dtype=np.float64)
        p[0] = positions[drone]
        v[0] = velocities[drone]
        for step in range(config.horizon):
            a = plan[step, index]
            control_scale = config.control_scale_s or config.dt
            p[step + 1] = (
                p[step]
                + config.dt * v[step]
                + 0.5 * config.dt * control_scale * a
            )
            v[step + 1] = v[step] + control_scale * a
        predicted_p[drone] = p
        predicted_v[drone] = v
    return predicted_p, predicted_v


def _terminal_halfspaces(
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    directions: dict[tuple[int, int], np.ndarray],
    drone_ids: list[int],
    config: PCBFConfig,
) -> list[tuple[np.ndarray, float]]:
    """Build terminal constraints ``c^T plan >= b``."""
    n_controls = config.horizon * len(drone_ids) * 2
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    constraints: list[tuple[np.ndarray, float]] = []
    for (i, j), distance in sorted(safe_distances.items()):
        direction = directions[(i, j)]
        c = np.zeros(n_controls, dtype=np.float64)
        for control_step in range(config.horizon):
            coefficient = _control_coefficient(
                config.horizon, control_step, config.dt,
                config.control_scale_s or config.dt,
            )
            offset_i = (control_step * len(drone_ids) + index[i]) * 2
            offset_j = (control_step * len(drone_ids) + index[j]) * 2
            c[offset_i : offset_i + 2] = coefficient * direction
            c[offset_j : offset_j + 2] = -coefficient * direction
        initial_projection = float(direction @ (positions[i] - positions[j]))
        drift_projection = config.horizon * config.dt * float(direction @ (velocities[i] - velocities[j]))
        constraints.append((c, distance + config.terminal_buffer_m - initial_projection - drift_projection))
        # A non-closing terminal projected relative velocity is invariant under
        # the zero-acceleration local controller: n^T r cannot subsequently
        # decrease.  Unlike forcing every vehicle to hover, this invariant
        # set permits mission progress before the terminal backup is needed.
        c = np.zeros(n_controls, dtype=np.float64)
        for control_step in range(config.horizon):
            offset_i = (control_step * len(drone_ids) + index[i]) * 2
            offset_j = (control_step * len(drone_ids) + index[j]) * 2
            c[offset_i : offset_i + 2] = config.control_scale_s or config.dt
            c[offset_j : offset_j + 2] = -(config.control_scale_s or config.dt)
            c[offset_i : offset_i + 2] *= direction
            c[offset_j : offset_j + 2] *= direction
        terminal_relative_velocity = float(direction @ (velocities[i] - velocities[j]))
        constraints.append((c, -terminal_relative_velocity - config.terminal_velocity_tolerance_mps))
    return constraints


def _project_halfspace(x: np.ndarray, c: np.ndarray, b: float) -> np.ndarray:
    deficit = b - float(c @ x)
    norm_sq = float(c @ c)
    if deficit > 0.0 and norm_sq > 1e-14:
        return x + (deficit / norm_sq) * c
    return x


def _project_terminal_and_box(
    plan: np.ndarray,
    *,
    terminal_constraints: list[tuple[np.ndarray, float]],
    config: PCBFConfig,
) -> tuple[np.ndarray, bool]:
    """Dykstra projection onto terminal halfspaces and the acceleration box."""
    flat = plan.reshape(-1).copy()
    corrections = [np.zeros_like(flat) for _ in range(len(terminal_constraints) + 1)]
    for _ in range(config.projection_iterations):
        previous = flat.copy()
        current = flat
        shifted = current + corrections[0]
        projected = np.clip(shifted, -config.a_max, config.a_max)
        corrections[0] = shifted - projected
        current = projected
        for index, (c, b) in enumerate(terminal_constraints, start=1):
            shifted = current + corrections[index]
            projected = _project_halfspace(shifted, c, b)
            corrections[index] = shifted - projected
            current = projected
        flat = current
        if float(np.linalg.norm(flat - previous)) <= config.tolerance:
            break
    feasible = bool(np.all(np.abs(flat) <= config.a_max + config.tolerance))
    feasible = feasible and all(float(c @ flat) >= b - config.tolerance for c, b in terminal_constraints)
    return flat.reshape(plan.shape), feasible


def solve_pcbf(
    *,
    nominal_accelerations: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    config: PCBFConfig,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> PCBFResult:
    """Solve the frozen-form PCBF safe MPC problem for a centralized fleet.

    ``safe_distances`` must be evaluated before the solve and held constant
    across this horizon. This is the precise Gate 0 mapping of the paper's
    time-invariant ``X``; live dynamic-boundary adaptation is intentionally a
    later, explicitly labeled integration step.
    """
    drone_ids = sorted(positions)
    if set(drone_ids) != set(velocities) or set(drone_ids) != set(nominal_accelerations):
        raise ValueError("PCBF positions, velocities and nominal accelerations must share drone ids")
    if len(drone_ids) < 2:
        raise ValueError("PCBF requires at least two drones")
    fixed = {
        drone: np.asarray(value, dtype=np.float64)
        for drone, value in (fixed_accelerations or {}).items()
    }
    unknown = set(fixed) - set(drone_ids)
    if unknown:
        raise ValueError(f"fixed acceleration contains unknown drones: {sorted(unknown)}")
    if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in fixed.values()):
        raise ValueError("fixed accelerations must be finite planar vectors")
    directions: dict[tuple[int, int], np.ndarray] = {}
    for i, j in safe_distances:
        if i not in positions or j not in positions:
            raise ValueError("PCBF safe-distance pair contains an unknown drone")
        relative = np.asarray(positions[i], dtype=np.float64) - np.asarray(positions[j], dtype=np.float64)
        norm = float(np.linalg.norm(relative))
        if norm <= 1e-9:
            return PCBFResult(
                accelerations=_brake(velocities, config.a_max, fixed), feasible=False,
                terminal_feasible=False, value=float("inf"), slack_sum=float("inf"),
                iterations=0, status="fail_closed", fail_closed_reason="coincident_pair", plan=None,
            )
        directions[(i, j)] = relative / norm
    normalized_positions = {i: np.asarray(positions[i], dtype=np.float64) for i in drone_ids}
    normalized_velocities = {i: np.asarray(velocities[i], dtype=np.float64) for i in drone_ids}
    nominal = np.stack([np.asarray(nominal_accelerations[i], dtype=np.float64) for i in drone_ids])
    if nominal.shape != (len(drone_ids), 2):
        raise ValueError("PCBF nominal accelerations must be planar vectors")
    nominal = np.clip(nominal, -config.a_max, config.a_max)
    for drone, acceleration in fixed.items():
        nominal[drone_ids.index(drone)] = acceleration
    return _solve_candidate_pcbf(
        nominal=nominal,
        positions=normalized_positions,
        velocities=normalized_velocities,
        safe_distances=safe_distances,
        drone_ids=drone_ids,
        config=config,
        fixed_accelerations=fixed,
    )

    # Retained below as the Gate-0 projected-subgradient reference
    # implementation. The C1 PX4 comparison uses the bounded candidate solver.
    plan = np.repeat(np.clip(nominal, -config.a_max, config.a_max)[None, :, :], config.horizon, axis=0)
    terminal_constraints = _terminal_halfspaces(
        positions=normalized_positions, velocities=normalized_velocities,
        safe_distances=safe_distances, directions=directions, drone_ids=drone_ids, config=config,
    )
    plan, terminal_feasible = _project_terminal_and_box(plan, terminal_constraints=terminal_constraints, config=config)
    if not terminal_feasible:
        return PCBFResult(
            accelerations=_brake(normalized_velocities, config.a_max), feasible=False,
            terminal_feasible=False, value=float("inf"), slack_sum=float("inf"),
            iterations=0, status="fail_closed", fail_closed_reason="terminal_infeasible", plan=None,
        )

    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    # The common interface supplies acceleration intent rather than an MPC
    # task cost.  Its one-command velocity image is the deterministic reference
    # used by this mission-tracking adaptation; it contains no AegisAir state.
    velocity_references = {
        drone: normalized_velocities[drone]
        + (config.control_scale_s or config.dt) * nominal[index[drone]]
        for drone in drone_ids
    }
    slack_sum = float("inf")
    for iteration in range(1, config.iterations + 1):
        predicted_p, predicted_v = _predict(plan, positions=normalized_positions, velocities=normalized_velocities, drone_ids=drone_ids, config=config)
        gradient = 2.0 * config.tracking_weight * (plan - nominal[None, :, :])
        for drone in drone_ids:
            agent_index = index[drone]
            for step in range(1, config.horizon + 1):
                velocity_error = predicted_v[drone][step] - velocity_references[drone]
                for control_step in range(step):
                    gradient[control_step, agent_index] += (
                        2.0 * config.velocity_tracking_weight
                        * (config.control_scale_s or config.dt)
                        * velocity_error
                    )
        slack_sum = 0.0
        for (i, j), distance in safe_distances.items():
            direction = directions[(i, j)]
            for step in range(1, config.horizon):
                deficit = distance - float(direction @ (predicted_p[i][step] - predicted_p[j][step]))
                if deficit <= 0.0:
                    continue
                slack_sum += deficit
                for control_step in range(step):
                    coefficient = _control_coefficient(
                        step, control_step, config.dt,
                        config.control_scale_s or config.dt,
                    )
                    gradient[control_step, index[i]] -= config.slack_weight * coefficient * direction
                    gradient[control_step, index[j]] += config.slack_weight * coefficient * direction
        next_plan = plan - config.step_size * gradient
        next_plan, terminal_feasible = _project_terminal_and_box(next_plan, terminal_constraints=terminal_constraints, config=config)
        if not terminal_feasible:
            return PCBFResult(
                accelerations=_brake(normalized_velocities, config.a_max), feasible=False,
                terminal_feasible=False, value=float("inf"), slack_sum=float("inf"),
                iterations=iteration, status="fail_closed", fail_closed_reason="terminal_projection_failed", plan=None,
            )
        if float(np.linalg.norm(next_plan - plan)) <= config.tolerance:
            plan = next_plan
            break
        plan = next_plan
    predicted_p, _ = _predict(plan, positions=normalized_positions, velocities=normalized_velocities, drone_ids=drone_ids, config=config)
    slack_sum = sum(
        max(0.0, distance - float(directions[(i, j)] @ (predicted_p[i][step] - predicted_p[j][step])))
        for (i, j), distance in safe_distances.items()
        for step in range(1, config.horizon)
    )
    velocity_error_cost = sum(
        float(np.sum((predicted_v[drone][1:] - velocity_references[drone]) ** 2))
        for drone in drone_ids
    )
    value = (
        config.slack_weight * slack_sum
        + config.tracking_weight * float(np.sum((plan - nominal[None, :, :]) ** 2))
        + config.velocity_tracking_weight * velocity_error_cost
    )
    return PCBFResult(
        accelerations={drone: plan[0, index[drone]].copy() for drone in drone_ids},
        feasible=True, terminal_feasible=True, value=float(value), slack_sum=float(slack_sum),
        iterations=iteration, status="optimal_approx", fail_closed_reason=None, plan=plan,
    )
