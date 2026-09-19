"""Huang et al. (ECC 2025) predictive control barrier function baseline.

The primary nonlinear program follows Eq. (8) of the paper and minimizes only
the sum of state-constraint slacks. A second, lexicographic program preserves
that PCBF value while minimizing the deviation of the first input from the
nominal command. This tie-break selects a mission-compatible input from the
primary problem's numerical optimum set without changing its objective.

Squared Euclidean UAV separation is a declared nonlinear common-testbed
adaptation of the paper's state set. The hard terminal set contains separated,
stopped agents; zero acceleration keeps this set invariant under the prediction
model. IPOPT returns local solutions for this non-convex adaptation, so solver
status and residuals are exposed rather than presented as a global optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from typing import Any

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class PCBFConfig:
    """Numerical and model parameters for the two-stage PCBF solve."""

    horizon: int = 20
    dt: float = 0.05
    control_scale_s: float | None = None
    a_max: float = 2.0
    terminal_buffer_m: float = 0.10
    terminal_velocity_tolerance_mps: float = 0.0
    position_bound_m: float = 20.0
    velocity_bound_mps: float = 5.0
    max_iterations: int = 300
    multistart_count: int = 3
    tolerance: float = 1e-7
    acceptable_tolerance: float = 1e-5
    lexicographic_tolerance: float = 1e-7

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
        if self.position_bound_m <= 0.0 or self.velocity_bound_mps <= 0.0:
            raise ValueError("PCBF state bounds must be positive")
        if self.max_iterations < 1:
            raise ValueError("PCBF max_iterations must be positive")
        if not 1 <= self.multistart_count <= 4:
            raise ValueError("PCBF multistart_count must be in [1, 4]")
        if (
            self.tolerance <= 0.0
            or self.acceptable_tolerance <= 0.0
            or self.lexicographic_tolerance < 0.0
        ):
            raise ValueError("PCBF numerical tolerances are invalid")


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
    stage1_status: str | None = None
    stage2_status: str | None = None
    max_constraint_violation: float = float("inf")
    tracking_cost: float = float("inf")
    tie_break_applied: bool = False


@dataclass(frozen=True)
class _SolverBundle:
    stage1: Any
    stage2: Any
    lbg_stage1: np.ndarray
    ubg_stage1: np.ndarray
    lbg_stage2: np.ndarray
    ubg_stage2: np.ndarray
    n_controls: int
    n_variables: int
    n_parameters: int


@dataclass(frozen=True)
class _SolveAttempt:
    decision: np.ndarray
    objective: float
    iterations: int
    status: str
    success: bool
    max_violation: float


_SOLVER_IDS = count()


def _brake(
    velocities: dict[int, np.ndarray],
    a_max: float,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    fixed_accelerations = fixed_accelerations or {}
    for drone, velocity in velocities.items():
        if drone in fixed_accelerations:
            output[drone] = fixed_accelerations[drone].copy()
            continue
        vector = np.asarray(velocity, dtype=np.float64)
        speed = float(np.linalg.norm(vector))
        output[drone] = (
            -a_max * vector / speed if speed > 1e-9 else np.zeros(2, dtype=np.float64)
        )
    return output


def _predict(
    plan: np.ndarray,
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    drone_ids: list[int],
    config: PCBFConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    predicted_p: dict[int, np.ndarray] = {}
    predicted_v: dict[int, np.ndarray] = {}
    control_scale = config.control_scale_s or config.dt
    for agent_index, drone in enumerate(drone_ids):
        p = np.zeros((config.horizon + 1, 2), dtype=np.float64)
        v = np.zeros((config.horizon + 1, 2), dtype=np.float64)
        p[0] = positions[drone]
        v[0] = velocities[drone]
        for step in range(config.horizon):
            acceleration = plan[step, agent_index]
            p[step + 1] = (
                p[step]
                + config.dt * v[step]
                + 0.5 * config.dt * control_scale * acceleration
            )
            v[step + 1] = v[step] + control_scale * acceleration
        predicted_p[drone] = p
        predicted_v[drone] = v
    return predicted_p, predicted_v


def _append_constraint(
    constraints: list[Any],
    lower: list[float],
    upper: list[float],
    expression: Any,
    lb: float,
    ub: float,
) -> None:
    constraints.append(expression)
    lower.append(lb)
    upper.append(ub)


@lru_cache(maxsize=16)
def _build_solver_bundle(
    *,
    horizon: int,
    drone_count: int,
    pair_indices: tuple[tuple[int, int], ...],
    dt: float,
    control_scale: float,
    terminal_buffer: float,
    terminal_velocity_tolerance: float,
    position_bound: float,
    velocity_bound: float,
    max_iterations: int,
    tolerance: float,
    acceptable_tolerance: float,
) -> _SolverBundle:
    controls_per_step = 2 * drone_count
    pair_count = len(pair_indices)
    n_parameters = 6 * drone_count + pair_count

    parameters = ca.MX.sym("parameters", n_parameters)
    controls = ca.MX.sym("controls", controls_per_step, horizon)
    slacks = ca.MX.sym("slacks", horizon)
    decision = ca.vertcat(ca.reshape(controls, -1, 1), slacks)

    velocity_offset = 2 * drone_count
    distance_offset = 4 * drone_count
    nominal_offset = distance_offset + pair_count
    positions = [parameters[2 * i : 2 * i + 2] for i in range(drone_count)]
    velocities = [
        parameters[velocity_offset + 2 * i : velocity_offset + 2 * i + 2]
        for i in range(drone_count)
    ]

    position_rollout: list[list[Any]] = [positions]
    velocity_rollout: list[list[Any]] = [velocities]
    for step in range(horizon):
        next_positions: list[Any] = []
        next_velocities: list[Any] = []
        for agent_index in range(drone_count):
            acceleration = controls[2 * agent_index : 2 * agent_index + 2, step]
            next_positions.append(
                position_rollout[-1][agent_index]
                + dt * velocity_rollout[-1][agent_index]
                + 0.5 * dt * control_scale * acceleration
            )
            next_velocities.append(
                velocity_rollout[-1][agent_index] + control_scale * acceleration
            )
        position_rollout.append(next_positions)
        velocity_rollout.append(next_velocities)

    constraints: list[Any] = []
    lower: list[float] = []
    upper: list[float] = []
    for step in range(horizon):
        for agent_index in range(drone_count):
            for component in range(2):
                _append_constraint(
                    constraints,
                    lower,
                    upper,
                    position_rollout[step][agent_index][component],
                    -position_bound,
                    position_bound,
                )
                _append_constraint(
                    constraints,
                    lower,
                    upper,
                    velocity_rollout[step][agent_index][component],
                    -velocity_bound,
                    velocity_bound,
                )
        for pair_index, (first, second) in enumerate(pair_indices):
            safe_distance = parameters[distance_offset + pair_index]
            relative = position_rollout[step][first] - position_rollout[step][second]
            _append_constraint(
                constraints,
                lower,
                upper,
                safe_distance * safe_distance
                - ca.dot(relative, relative)
                - slacks[step],
                -ca.inf,
                0.0,
            )

    for agent_index in range(drone_count):
        for component in range(2):
            _append_constraint(
                constraints,
                lower,
                upper,
                position_rollout[horizon][agent_index][component],
                -position_bound,
                position_bound,
            )
            _append_constraint(
                constraints,
                lower,
                upper,
                velocity_rollout[horizon][agent_index][component],
                -velocity_bound,
                velocity_bound,
            )
            _append_constraint(
                constraints,
                lower,
                upper,
                velocity_rollout[horizon][agent_index][component],
                -terminal_velocity_tolerance,
                terminal_velocity_tolerance,
            )
    for pair_index, (first, second) in enumerate(pair_indices):
        terminal_distance = parameters[distance_offset + pair_index] + terminal_buffer
        terminal_relative = (
            position_rollout[horizon][first] - position_rollout[horizon][second]
        )
        _append_constraint(
            constraints,
            lower,
            upper,
            terminal_distance * terminal_distance
            - ca.dot(terminal_relative, terminal_relative),
            -ca.inf,
            0.0,
        )

    common_constraints = ca.vertcat(*constraints)
    primary_objective = ca.sum1(slacks)
    nominal_first = parameters[nominal_offset : nominal_offset + controls_per_step]
    tracking_objective = ca.sumsqr(controls[:, 0] - nominal_first)

    solver_options = {
        "print_time": False,
        "error_on_fail": False,
        "expand": True,
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": max_iterations,
            "tol": tolerance,
            "acceptable_tol": acceptable_tolerance,
            "acceptable_constr_viol_tol": acceptable_tolerance,
            "mu_strategy": "adaptive",
        },
    }
    solver_id = next(_SOLVER_IDS)
    stage1 = ca.nlpsol(
        f"pcbf_stage1_{solver_id}",
        "ipopt",
        {"x": decision, "p": parameters, "f": primary_objective, "g": common_constraints},
        solver_options,
    )

    value_limit = ca.MX.sym("value_limit")
    stage2_constraints = ca.vertcat(common_constraints, primary_objective - value_limit)
    stage2 = ca.nlpsol(
        f"pcbf_stage2_{solver_id}",
        "ipopt",
        {
            "x": decision,
            "p": ca.vertcat(parameters, value_limit),
            "f": tracking_objective,
            "g": stage2_constraints,
        },
        solver_options,
    )
    return _SolverBundle(
        stage1=stage1,
        stage2=stage2,
        lbg_stage1=np.asarray(lower, dtype=np.float64),
        ubg_stage1=np.asarray(upper, dtype=np.float64),
        lbg_stage2=np.asarray([*lower, -np.inf], dtype=np.float64),
        ubg_stage2=np.asarray([*upper, 0.0], dtype=np.float64),
        n_controls=controls_per_step * horizon,
        n_variables=controls_per_step * horizon + horizon,
        n_parameters=n_parameters,
    )


def _project_sum_to_box(
    values: np.ndarray, target_sum: float, lower: float, upper: float
) -> np.ndarray:
    projected = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
    target = float(np.clip(target_sum, len(projected) * lower, len(projected) * upper))
    for _ in range(2 * len(projected) + 4):
        deficit = target - float(np.sum(projected))
        if abs(deficit) <= 1e-10:
            break
        free = np.flatnonzero(
            projected < upper - 1e-12 if deficit > 0.0 else projected > lower + 1e-12
        )
        if free.size == 0:
            break
        projected[free] += deficit / free.size
        projected = np.clip(projected, lower, upper)
    return projected


def _enforce_terminal_stop(
    plan: np.ndarray,
    *,
    velocities: dict[int, np.ndarray],
    drone_ids: list[int],
    fixed_accelerations: dict[int, np.ndarray],
    config: PCBFConfig,
) -> np.ndarray:
    output = np.asarray(plan, dtype=np.float64).copy()
    control_scale = config.control_scale_s or config.dt
    for agent_index, drone in enumerate(drone_ids):
        if drone in fixed_accelerations:
            output[:, agent_index] = fixed_accelerations[drone]
            continue
        for component in range(2):
            output[:, agent_index, component] = _project_sum_to_box(
                output[:, agent_index, component],
                -float(velocities[drone][component]) / control_scale,
                -config.a_max,
                config.a_max,
            )
    return output


def _initial_plans(
    *,
    nominal: np.ndarray,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    drone_ids: list[int],
    pairs: tuple[tuple[int, int], ...],
    fixed_accelerations: dict[int, np.ndarray],
    config: PCBFConfig,
) -> list[np.ndarray]:
    repeated_nominal = np.repeat(nominal[None, :, :], config.horizon, axis=0)
    plans = [
        _enforce_terminal_stop(
            repeated_nominal,
            velocities=velocities,
            drone_ids=drone_ids,
            fixed_accelerations=fixed_accelerations,
            config=config,
        )
    ]
    if config.multistart_count > 1 and pairs:
        first, second = pairs[0]
        relative = positions[first] - positions[second]
        norm = float(np.linalg.norm(relative))
        radial = relative / norm if norm > 1e-9 else np.array([1.0, 0.0])
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        index = {drone: agent_index for agent_index, drone in enumerate(drone_ids)}
        split = max(1, config.horizon // 2)
        shape = np.ones(config.horizon, dtype=np.float64)
        shape[split:] = -split / max(1, config.horizon - split)
        for sign in (1.0, -1.0):
            plan = repeated_nominal.copy()
            pulse = sign * 0.75 * config.a_max * shape[:, None] * tangent[None, :]
            if first not in fixed_accelerations:
                plan[:, index[first]] += pulse
            if second not in fixed_accelerations:
                plan[:, index[second]] -= pulse
            plans.append(
                _enforce_terminal_stop(
                    plan,
                    velocities=velocities,
                    drone_ids=drone_ids,
                    fixed_accelerations=fixed_accelerations,
                    config=config,
                )
            )
    if config.multistart_count > 3:
        plans.append(
            _enforce_terminal_stop(
                np.zeros_like(repeated_nominal),
                velocities=velocities,
                drone_ids=drone_ids,
                fixed_accelerations=fixed_accelerations,
                config=config,
            )
        )
    unique: list[np.ndarray] = []
    for plan in plans:
        if not any(np.allclose(plan, existing, atol=1e-12, rtol=0.0) for existing in unique):
            unique.append(plan)
        if len(unique) >= config.multistart_count:
            break
    return unique


def _pack_decision(plan: np.ndarray, slacks: np.ndarray) -> np.ndarray:
    controls = np.asarray(plan, dtype=np.float64).reshape(plan.shape[0], -1).T
    return np.concatenate([
        controls.reshape(-1, order="F"), np.asarray(slacks, dtype=np.float64)
    ])


def _unpack_decision(
    decision: np.ndarray, *, horizon: int, drone_count: int
) -> tuple[np.ndarray, np.ndarray]:
    n_controls = horizon * drone_count * 2
    controls = decision[:n_controls].reshape((2 * drone_count, horizon), order="F")
    return (
        controls.T.reshape((horizon, drone_count, 2)),
        decision[n_controls:],
    )


def _initial_slacks(
    plan: np.ndarray,
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    drone_ids: list[int],
    config: PCBFConfig,
) -> np.ndarray:
    predicted_p, _ = _predict(
        plan,
        positions=positions,
        velocities=velocities,
        drone_ids=drone_ids,
        config=config,
    )
    slacks = np.zeros(config.horizon, dtype=np.float64)
    for step in range(config.horizon):
        violations = [
            distance * distance
            - float(np.dot(
                predicted_p[first][step] - predicted_p[second][step],
                predicted_p[first][step] - predicted_p[second][step],
            ))
            for (first, second), distance in safe_distances.items()
        ]
        slacks[step] = max(0.0, *violations)
    return slacks


def _variable_bounds(
    *,
    bundle: _SolverBundle,
    drone_ids: list[int],
    fixed_accelerations: dict[int, np.ndarray],
    config: PCBFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(bundle.n_variables, -np.inf, dtype=np.float64)
    upper = np.full(bundle.n_variables, np.inf, dtype=np.float64)
    controls_per_step = 2 * len(drone_ids)
    for step in range(config.horizon):
        for agent_index, drone in enumerate(drone_ids):
            for component in range(2):
                flat_index = step * controls_per_step + 2 * agent_index + component
                if drone in fixed_accelerations:
                    value = float(fixed_accelerations[drone][component])
                    lower[flat_index] = value
                    upper[flat_index] = value
                else:
                    lower[flat_index] = -config.a_max
                    upper[flat_index] = config.a_max
    lower[bundle.n_controls :] = 0.0
    return lower, upper


def _max_bound_violation(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    lower_violation = np.where(np.isfinite(lower), lower - values, 0.0)
    upper_violation = np.where(np.isfinite(upper), values - upper, 0.0)
    return float(max(0.0, np.max(lower_violation), np.max(upper_violation)))


def _run_solver(
    solver: Any,
    *,
    initial: np.ndarray,
    parameters: np.ndarray,
    lower_variables: np.ndarray,
    upper_variables: np.ndarray,
    lower_constraints: np.ndarray,
    upper_constraints: np.ndarray,
    acceptable_tolerance: float,
) -> _SolveAttempt:
    try:
        result = solver(
            x0=initial,
            p=parameters,
            lbx=lower_variables,
            ubx=upper_variables,
            lbg=lower_constraints,
            ubg=upper_constraints,
        )
        decision = np.asarray(result["x"], dtype=np.float64).reshape(-1)
        constraint_values = np.asarray(result["g"], dtype=np.float64).reshape(-1)
        stats = solver.stats()
        max_violation = max(
            _max_bound_violation(decision, lower_variables, upper_variables),
            _max_bound_violation(
                constraint_values, lower_constraints, upper_constraints
            ),
        )
        success = bool(stats.get("success", False))
        success = success and bool(np.all(np.isfinite(decision)))
        success = success and max_violation <= 10.0 * acceptable_tolerance
        return _SolveAttempt(
            decision=decision,
            objective=float(result["f"]),
            iterations=int(stats.get("iter_count", 0)),
            status=str(stats.get("return_status", "unknown")),
            success=success,
            max_violation=max_violation,
        )
    except Exception as exc:  # CasADi plugin failures must fail closed.
        return _SolveAttempt(
            decision=np.asarray(initial, dtype=np.float64),
            objective=float("inf"),
            iterations=0,
            status=f"exception:{type(exc).__name__}",
            success=False,
            max_violation=float("inf"),
        )


def _terminal_is_feasible(
    plan: np.ndarray,
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    drone_ids: list[int],
    config: PCBFConfig,
) -> bool:
    predicted_p, predicted_v = _predict(
        plan,
        positions=positions,
        velocities=velocities,
        drone_ids=drone_ids,
        config=config,
    )
    tolerance = 10.0 * config.acceptable_tolerance
    for drone in drone_ids:
        if np.max(np.abs(predicted_p[drone][-1])) > config.position_bound_m + tolerance:
            return False
        if np.max(np.abs(predicted_v[drone][-1])) > config.velocity_bound_mps + tolerance:
            return False
        if (
            np.max(np.abs(predicted_v[drone][-1]))
            > config.terminal_velocity_tolerance_mps + tolerance
        ):
            return False
    for (first, second), distance in safe_distances.items():
        terminal_distance = float(
            np.linalg.norm(predicted_p[first][-1] - predicted_p[second][-1])
        )
        if terminal_distance + tolerance < distance + config.terminal_buffer_m:
            return False
    return True


def solve_pcbf(
    *,
    nominal_accelerations: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    safe_distances: dict[tuple[int, int], float],
    config: PCBFConfig,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> PCBFResult:
    """Solve Huang's PCBF and then select the closest first nominal input."""
    drone_ids = sorted(positions)
    if set(drone_ids) != set(velocities) or set(drone_ids) != set(nominal_accelerations):
        raise ValueError("PCBF positions, velocities and nominal accelerations must share drone ids")
    if len(drone_ids) < 2:
        raise ValueError("PCBF requires at least two drones")

    normalized_positions = {
        drone: np.asarray(positions[drone], dtype=np.float64) for drone in drone_ids
    }
    normalized_velocities = {
        drone: np.asarray(velocities[drone], dtype=np.float64) for drone in drone_ids
    }
    nominal = np.stack([
        np.asarray(nominal_accelerations[drone], dtype=np.float64)
        for drone in drone_ids
    ])
    if nominal.shape != (len(drone_ids), 2):
        raise ValueError("PCBF nominal accelerations must be planar vectors")
    state_values = [*normalized_positions.values(), *normalized_velocities.values()]
    if any(
        value.shape != (2,) or not np.all(np.isfinite(value)) for value in state_values
    ) or not np.all(np.isfinite(nominal)):
        raise ValueError("PCBF state and nominal inputs must be finite planar vectors")
    nominal = np.clip(nominal, -config.a_max, config.a_max)

    fixed = {
        drone: np.asarray(value, dtype=np.float64)
        for drone, value in (fixed_accelerations or {}).items()
    }
    unknown = set(fixed) - set(drone_ids)
    if unknown:
        raise ValueError(f"fixed acceleration contains unknown drones: {sorted(unknown)}")
    if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in fixed.values()):
        raise ValueError("fixed accelerations must be finite planar vectors")
    if any(np.linalg.norm(acceleration) > 1e-12 for acceleration in fixed.values()):
        # A permanently fixed nonzero acceleration has no compact invariant
        # terminal set under the bounded position/velocity model. Preserve the
        # revoked input and fail closed instead of claiming PCBF feasibility.
        return PCBFResult(
            accelerations=_brake(normalized_velocities, config.a_max, fixed),
            feasible=False,
            terminal_feasible=False,
            value=float("inf"),
            slack_sum=float("inf"),
            iterations=0,
            status="fail_closed",
            fail_closed_reason="fixed_input_has_no_invariant_terminal_policy",
            plan=None,
            stage1_status="not_started",
        )
    for drone, acceleration in fixed.items():
        nominal[drone_ids.index(drone)] = acceleration

    normalized_distances: dict[tuple[int, int], float] = {}
    for (first, second), raw_distance in safe_distances.items():
        if first == second or first not in positions or second not in positions:
            raise ValueError("PCBF safe-distance pair is invalid")
        pair = tuple(sorted((first, second)))
        if pair in normalized_distances:
            raise ValueError("PCBF safe-distance pairs must be unique")
        distance = float(raw_distance)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("PCBF safe distances must be positive and finite")
        normalized_distances[pair] = distance
    if not normalized_distances:
        raise ValueError("PCBF requires at least one safety constraint")

    pairs = tuple(sorted(normalized_distances))
    index = {drone: agent_index for agent_index, drone in enumerate(drone_ids)}
    pair_indices = tuple((index[first], index[second]) for first, second in pairs)
    bundle = _build_solver_bundle(
        horizon=config.horizon,
        drone_count=len(drone_ids),
        pair_indices=pair_indices,
        dt=config.dt,
        control_scale=config.control_scale_s or config.dt,
        terminal_buffer=config.terminal_buffer_m,
        terminal_velocity_tolerance=config.terminal_velocity_tolerance_mps,
        position_bound=config.position_bound_m,
        velocity_bound=config.velocity_bound_mps,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        acceptable_tolerance=config.acceptable_tolerance,
    )

    parameter_values = np.concatenate([
        *(normalized_positions[drone] for drone in drone_ids),
        *(normalized_velocities[drone] for drone in drone_ids),
        np.asarray([normalized_distances[pair] for pair in pairs]),
        nominal.reshape(-1),
    ]).astype(np.float64)
    if parameter_values.size != bundle.n_parameters:
        raise RuntimeError("internal PCBF parameter layout mismatch")
    lower_variables, upper_variables = _variable_bounds(
        bundle=bundle,
        drone_ids=drone_ids,
        fixed_accelerations=fixed,
        config=config,
    )

    stage1_attempts: list[_SolveAttempt] = []
    for initial_plan in _initial_plans(
        nominal=nominal,
        positions=normalized_positions,
        velocities=normalized_velocities,
        drone_ids=drone_ids,
        pairs=pairs,
        fixed_accelerations=fixed,
        config=config,
    ):
        initial_slacks = _initial_slacks(
            initial_plan,
            positions=normalized_positions,
            velocities=normalized_velocities,
            safe_distances=normalized_distances,
            drone_ids=drone_ids,
            config=config,
        )
        attempt = _run_solver(
            bundle.stage1,
            initial=_pack_decision(initial_plan, initial_slacks),
            parameters=parameter_values,
            lower_variables=lower_variables,
            upper_variables=upper_variables,
            lower_constraints=bundle.lbg_stage1,
            upper_constraints=bundle.ubg_stage1,
            acceptable_tolerance=config.acceptable_tolerance,
        )
        stage1_attempts.append(attempt)
        if attempt.success and attempt.objective <= config.acceptable_tolerance:
            break

    successful_stage1 = [attempt for attempt in stage1_attempts if attempt.success]
    total_iterations = sum(attempt.iterations for attempt in stage1_attempts)
    if not successful_stage1:
        return PCBFResult(
            accelerations=_brake(normalized_velocities, config.a_max, fixed),
            feasible=False,
            terminal_feasible=False,
            value=float("inf"),
            slack_sum=float("inf"),
            iterations=total_iterations,
            status="fail_closed",
            fail_closed_reason="stage1_infeasible_or_failed",
            plan=None,
            stage1_status=(stage1_attempts[-1].status if stage1_attempts else "not_started"),
            max_constraint_violation=min(
                (attempt.max_violation for attempt in stage1_attempts),
                default=float("inf"),
            ),
        )

    primary = min(
        successful_stage1,
        key=lambda attempt: (attempt.objective, attempt.max_violation),
    )
    value_limit = max(0.0, primary.objective) + config.lexicographic_tolerance
    stage2 = _run_solver(
        bundle.stage2,
        initial=primary.decision,
        parameters=np.concatenate([parameter_values, [value_limit]]),
        lower_variables=lower_variables,
        upper_variables=upper_variables,
        lower_constraints=bundle.lbg_stage2,
        upper_constraints=bundle.ubg_stage2,
        acceptable_tolerance=config.acceptable_tolerance,
    )
    total_iterations += stage2.iterations
    stage2_slack_sum = float("inf")
    if stage2.success:
        _, stage2_slacks = _unpack_decision(
            stage2.decision,
            horizon=config.horizon,
            drone_count=len(drone_ids),
        )
        stage2_slack_sum = float(np.sum(stage2_slacks))
    value_preservation_tolerance = max(
        10.0 * config.tolerance,
        config.lexicographic_tolerance,
    )
    tie_break_applied = bool(
        stage2.success
        and stage2_slack_sum <= value_limit + value_preservation_tolerance
    )
    selected = stage2 if tie_break_applied else primary

    plan, slacks = _unpack_decision(
        selected.decision,
        horizon=config.horizon,
        drone_count=len(drone_ids),
    )
    terminal_feasible = _terminal_is_feasible(
        plan,
        positions=normalized_positions,
        velocities=normalized_velocities,
        safe_distances=normalized_distances,
        drone_ids=drone_ids,
        config=config,
    )
    if not terminal_feasible:
        return PCBFResult(
            accelerations=_brake(normalized_velocities, config.a_max, fixed),
            feasible=False,
            terminal_feasible=False,
            value=float(primary.objective),
            slack_sum=float(np.sum(slacks)),
            iterations=total_iterations,
            status="fail_closed",
            fail_closed_reason="terminal_verification_failed",
            plan=plan,
            stage1_status=primary.status,
            stage2_status=stage2.status,
            max_constraint_violation=selected.max_violation,
            tie_break_applied=tie_break_applied,
        )

    accelerations = {
        drone: plan[0, agent_index].copy()
        for agent_index, drone in enumerate(drone_ids)
    }
    return PCBFResult(
        accelerations=accelerations,
        feasible=True,
        terminal_feasible=True,
        value=float(max(0.0, primary.objective)),
        slack_sum=float(max(0.0, np.sum(slacks))),
        iterations=total_iterations,
        status="solved_local",
        fail_closed_reason=None,
        plan=plan,
        stage1_status=primary.status,
        stage2_status=stage2.status,
        max_constraint_violation=selected.max_violation,
        tracking_cost=float(np.sum((plan[0] - nominal) ** 2)),
        tie_break_applied=tie_break_applied,
    )
