"""Centralized acceleration-aware HOCBF for multi-UAV Runtime Assurance.

The barrier is second-order in the PX4-compatible double-integrator model
``p_dot = v``, ``v_dot = a``.  For a pair ``(i, j)``:

    r = p_i - p_j,  v = v_i - v_j
    h = ||r||^2 - s^2            (s = d_safe, sampled-and-held per step)
    psi2 = h_ddot + (k1+k2) h_dot + k1*k2 h >= 0

With ``s_dot = s_ddot = 0`` inside one QP solve this reduces to the linear
constraint on the relative acceleration:

    2 r^T (a_i - a_j) >=
        -2 ||v||^2 - 2(k1+k2) r^T v - k1*k2 (||r||^2 - s^2)

All pairwise constraints enter one centralized QP so a correction for UAV 2
cannot be overwritten by a later correction for UAV 3 (no sequential
projection).
"""

from __future__ import annotations

import numpy as np

from swarm.ra.projection import (
    project_box as _project_box,
    project_box_and_single_halfspace,
    project_halfspace as _project_halfspace,
)


def _pair_constraint(
    *,
    i: int,
    j: int,
    n_agents: int,
    p_i: np.ndarray,
    p_j: np.ndarray,
    v_i: np.ndarray,
    v_j: np.ndarray,
    s: float,
    k1: float,
    k2: float,
) -> tuple[np.ndarray, float]:
    """Return ``c`` and ``b`` such that ``c^T A >= b`` constrains the pair."""
    r = np.asarray(p_i, dtype=np.float64) - np.asarray(p_j, dtype=np.float64)
    v = np.asarray(v_i, dtype=np.float64) - np.asarray(v_j, dtype=np.float64)
    r2 = float(np.dot(r, r))
    v2 = float(np.dot(v, v))
    b = -2.0 * v2 - 2.0 * (k1 + k2) * float(np.dot(r, v)) - k1 * k2 * (r2 - s * s)

    c = np.zeros(2 * n_agents, dtype=np.float64)
    c[2 * i : 2 * i + 2] = 2.0 * r
    c[2 * j : 2 * j + 2] = -2.0 * r
    return c, b


def minimum_hocbf_constraint_slack(
    *,
    accelerations: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    d_safe: dict[tuple[int, int], float],
    k1: float,
    k2: float,
    a_max: float,
    box_constrained_drones: set[int] | None = None,
) -> float:
    """Evaluate the actual joint acceleration against the HOCBF and box."""
    drone_ids = sorted(positions)
    if set(accelerations) != set(drone_ids):
        raise ValueError("accelerations must cover every constrained drone")
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    packed = np.concatenate(
        [np.asarray(accelerations[drone], dtype=np.float64) for drone in drone_ids]
    )
    if packed.shape != (2 * len(drone_ids),) or not np.all(np.isfinite(packed)):
        raise ValueError("accelerations must be finite planar vectors")
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
    for (i, j), safe_distance in d_safe.items():
        c, b = _pair_constraint(
            i=index[i],
            j=index[j],
            n_agents=len(drone_ids),
            p_i=positions[i],
            p_j=positions[j],
            v_i=velocities[i],
            v_j=velocities[j],
            s=safe_distance,
            k1=k1,
            k2=k2,
        )
        slacks.append(float(np.dot(c, packed) - b))
    return min(slacks, default=float("inf"))


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


def _fallback_accelerations(
    *,
    x: np.ndarray,
    feasible: bool,
    drone_ids: list[int],
    velocities: dict[int, np.ndarray],
    a_max: float,
    infeasible_fallback: str,
    fixed_accelerations: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for index, drone in enumerate(drone_ids):
        if drone in fixed_accelerations:
            output[drone] = fixed_accelerations[drone].copy()
        elif feasible:
            output[drone] = x[2 * index : 2 * index + 2].copy()
        else:
            velocity = np.asarray(velocities[drone][:2], dtype=np.float64)
            speed = float(np.linalg.norm(velocity))
            if infeasible_fallback == "max_brake" and speed > 1e-9:
                output[drone] = -a_max * velocity / speed
            else:
                output[drone] = np.clip(-velocity, -a_max, a_max)
    return output


def solve_acceleration_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    d_safe: dict[tuple[int, int], float],
    k1: float,
    k2: float,
    a_max: float,
    max_iters: int = 3000,
    tol: float = 1e-7,
    infeasible_fallback: str = "velocity_cancel",
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """Solve one centralized QP for every drone's safe acceleration.

    Returns ``(a_safe, feasible, iterations)``.  If the QP is infeasible the
    returned accelerations are a deterministic hard-brake fallback and
    ``feasible`` is ``False``.
    """
    if infeasible_fallback not in {"velocity_cancel", "max_brake"}:
        raise ValueError("unknown HOCBF infeasible fallback")
    drone_ids = sorted(positions)
    n_agents = len(drone_ids)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    fixed_mask, fixed_values, fixed = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )

    x = np.zeros(2 * n_agents, dtype=np.float64)
    for idx, drone in enumerate(drone_ids):
        x[2 * idx : 2 * idx + 2] = np.asarray(a_nom[drone], dtype=np.float64)
    x[fixed_mask] = fixed_values[fixed_mask]

    halfspaces: list[tuple[np.ndarray, float]] = []
    for (i, j), s in d_safe.items():
        c, b = _pair_constraint(
            i=index[i],
            j=index[j],
            n_agents=n_agents,
            p_i=positions[i],
            p_j=positions[j],
            v_i=velocities[i],
            v_j=velocities[j],
            s=s,
            k1=k1,
            k2=k2,
        )
        halfspaces.append((c, b))

    if len(halfspaces) == 1:
        x0, feasible, iterations = project_box_and_single_halfspace(
            x_nom=x,
            c=halfspaces[0][0],
            b=halfspaces[0][1],
            a_max=a_max,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
        return (
            _fallback_accelerations(
                x=x0,
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

    # Dykstra's alternating projection onto the intersection of the box and
    # every pairwise half-space converges to the closest point to a_nom.
    corrections = [np.zeros_like(x) for _ in range(len(halfspaces) + 1)]
    x0 = x.copy()
    iterations = 0
    for iterations in range(1, max_iters + 1):
        previous = x0.copy()
        previous_corrections = [correction.copy() for correction in corrections]
        cur = x0
        correction_change = 0.0
        for idx in range(len(halfspaces) + 1):
            y = cur + corrections[idx]
            if idx == 0:
                proj = _project_box(
                    y,
                    a_max,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                )
            else:
                c, b = halfspaces[idx - 1]
                proj = _project_halfspace(y, c, b)
            correction = y - proj
            correction_change = max(
                correction_change,
                float(np.linalg.norm(correction - corrections[idx])),
            )
            corrections[idx] = correction
            cur = proj
        x0 = cur
        if (
            float(np.linalg.norm(x0 - previous)) <= tol
            and correction_change <= tol
        ):
            break

    feasible = True
    box_ok = np.all(np.abs(x0[~fixed_mask]) <= a_max + 1e-6)
    fixed_ok = np.allclose(
        x0[fixed_mask], fixed_values[fixed_mask], atol=1e-6, rtol=0.0
    )
    for c, b in halfspaces:
        if float(np.dot(c, x0)) < b - 1e-6:
            feasible = False
            break
    if not box_ok or not fixed_ok:
        feasible = False

    a_safe = _fallback_accelerations(
        x=x0,
        feasible=feasible,
        drone_ids=drone_ids,
        velocities=velocities,
        a_max=a_max,
        infeasible_fallback=infeasible_fallback,
        fixed_accelerations=fixed,
    )
    return a_safe, feasible, iterations


def minimum_acceleration_box_reserve(
    *,
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    d_safe: dict[tuple[int, int], float],
    k1: float,
    k2: float,
    a_max: float,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> float:
    """Return the minimum normalized HOCBF feasibility headroom.

    For each pair, ``a_max * ||c||_1`` is the largest left-hand side the
    shared component-wise acceleration box can supply.  The returned reserve
    is ``(max_lhs-b)/max_lhs``: negative means that pair is impossible even at
    the box extremum, while values below one mean that positive separating
    acceleration is already required.  This is an auditable state/constraint
    diagnostic, independent of the nominal controller.
    """
    drone_ids = sorted(positions)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    fixed_mask, fixed_values, _ = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )
    reserves = []
    for (i, j), safe_distance in d_safe.items():
        c, b = _pair_constraint(
            i=index[i],
            j=index[j],
            n_agents=len(drone_ids),
            p_i=positions[i],
            p_j=positions[j],
            v_i=velocities[i],
            v_j=velocities[j],
            s=safe_distance,
            k1=k1,
            k2=k2,
        )
        fixed_lhs = float(np.dot(c[fixed_mask], fixed_values[fixed_mask]))
        available_lhs = a_max * float(np.sum(np.abs(c[~fixed_mask])))
        maximum_lhs = fixed_lhs + available_lhs
        if available_lhs <= 1e-12:
            reserves.append(float("-inf") if maximum_lhs < b else float("inf"))
        else:
            reserves.append((maximum_lhs - b) / available_lhs)
    return min(reserves, default=float("inf"))


def beta_of_tau(dt: float, tau: float) -> float:
    """Position-update coefficient of the exact first-order velocity dynamics.

    For ``dv/dt = (u - v)/tau`` under ZOH over one sample:
        v_{k+1} = (1-alpha) v_k + alpha u_k
        p_{k+1} = p_k + v_k dt + beta (u_k - v_k)
    where ``alpha = 1 - exp(-dt/tau)`` and ``beta = dt - tau*alpha``.  With the
    acceleration intent ``u_k = v_k + a_k dt`` this becomes
        p_{k+1} = p_k + v_k dt + beta a_k dt.
    """
    if tau <= 0.0:
        return dt
    alpha = 1.0 - float(np.exp(-dt / tau))
    return dt - tau * alpha


def solve_robust_sampled_data_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    s_now: dict[tuple[int, int], float],
    s_next: dict[tuple[int, int], float],
    dt: float,
    gamma: float,
    a_max: float,
    beta: dict[int, tuple[float, float]],
    command_scale: float | None = None,
    max_iters: int = 3000,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """Robust projected-barrier sampled-data QP with tau-interval uncertainty.

    Uses the conservative projected barrier ``n^T r - D`` (a linear sufficient
    condition for ``||r|| >= D``) so the one-step constraint is exact-linear in
    the accelerations.  For each pair the four ``(beta_i, beta_j)`` rectangle
    vertices are all added, making the constraint robust to
    ``tau_i in [tau_min, tau_max]``.
    """
    drone_ids = sorted(positions)
    scale = dt if command_scale is None else command_scale
    if scale <= 0.0:
        raise ValueError("command_scale must be positive")
    n_agents = len(drone_ids)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    fixed_mask, fixed_values, fixed = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )

    x = np.zeros(2 * n_agents, dtype=np.float64)
    for idx, drone in enumerate(drone_ids):
        x[2 * idx : 2 * idx + 2] = np.asarray(a_nom[drone], dtype=np.float64)
    x[fixed_mask] = fixed_values[fixed_mask]

    halfspaces: list[tuple[np.ndarray, float]] = []
    for (i, j) in sorted(s_now):
        ii = index[i]
        jj = index[j]
        r = np.asarray(positions[i], dtype=np.float64) - np.asarray(
            positions[j], dtype=np.float64
        )
        v = np.asarray(velocities[i], dtype=np.float64) - np.asarray(
            velocities[j], dtype=np.float64
        )
        dist = float(np.linalg.norm(r))
        n = r / dist if dist > 1e-9 else np.array([1.0, 0.0])
        d_now = s_now[(i, j)]
        d_next = s_next[(i, j)]
        h_now = float(np.dot(n, r)) - d_now
        bi_lo, bi_hi = beta[i]
        bj_lo, bj_hi = beta[j]

        for bi in (bi_lo, bi_hi):
            for bj in (bj_lo, bj_hi):
                c = np.zeros(2 * n_agents, dtype=np.float64)
                c[2 * ii : 2 * ii + 2] = scale * bi * n
                c[2 * jj : 2 * jj + 2] = -scale * bj * n
                b = (
                    (1.0 - gamma) * h_now
                    - float(np.dot(n, r))
                    - dt * float(np.dot(n, v))
                    + d_next
                )
                halfspaces.append((c, b))

    corrections = [np.zeros_like(x) for _ in range(len(halfspaces) + 1)]
    x0 = x.copy()
    for _ in range(max_iters):
        cur = x0
        for idx in range(len(halfspaces) + 1):
            y = cur + corrections[idx]
            if idx == 0:
                proj = _project_box(
                    y,
                    a_max,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                )
            else:
                c, b = halfspaces[idx - 1]
                proj = _project_halfspace(y, c, b)
            corrections[idx] = y - proj
            cur = proj
        x0 = cur

    feasible = bool(np.all(np.abs(x0[~fixed_mask]) <= a_max + 1e-6))
    feasible = feasible and bool(
        np.allclose(x0[fixed_mask], fixed_values[fixed_mask], atol=1e-6, rtol=0.0)
    )
    for c, b in halfspaces:
        if float(np.dot(c, x0)) < b - 1e-6:
            feasible = False
            break

    a_safe = _fallback_accelerations(
        x=x0,
        feasible=feasible,
        drone_ids=drone_ids,
        velocities=velocities,
        a_max=a_max,
        infeasible_fallback="velocity_cancel",
        fixed_accelerations=fixed,
    )
    return a_safe, feasible, max_iters


def solve_sampled_data_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    s_now: dict[tuple[int, int], float],
    s_next: dict[tuple[int, int], float],
    dt: float,
    gamma: float,
    a_max: float,
    alpha: float = 1.0,
    max_iters: int = 3000,
    fixed_accelerations: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """Centralized sampled-data acceleration QP.

    For each pair, the barrier is evaluated at the next sample:

        r_next = r + v*dt + 0.5*a_rel*dt^2
        h_next = ||r_next||^2 - s_next^2 >= (1 - gamma) * h_now

    The quadratic term in ``a`` is linearized around ``a_nom``, keeping the
    problem a convex QP over all agents simultaneously.
    """
    drone_ids = sorted(positions)
    n_agents = len(drone_ids)
    index = {drone: idx for idx, drone in enumerate(drone_ids)}
    fixed_mask, fixed_values, fixed = _fixed_coordinates(
        drone_ids=drone_ids, fixed_accelerations=fixed_accelerations
    )

    x = np.zeros(2 * n_agents, dtype=np.float64)
    for idx, drone in enumerate(drone_ids):
        x[2 * idx : 2 * idx + 2] = np.asarray(a_nom[drone], dtype=np.float64)
    x[fixed_mask] = fixed_values[fixed_mask]
    x_nom = x.copy()

    halfspaces: list[tuple[np.ndarray, float]] = []
    for (i, j), sk in s_now.items():
        ii = index[i]
        jj = index[j]
        r = np.asarray(positions[i], dtype=np.float64) - np.asarray(
            positions[j], dtype=np.float64
        )
        v = np.asarray(velocities[i], dtype=np.float64) - np.asarray(
            velocities[j], dtype=np.float64
        )
        a_rel_nom = np.asarray(a_nom[i], dtype=np.float64) - np.asarray(
            a_nom[j], dtype=np.float64
        )
        r_pred = r + v * dt + 0.5 * alpha * a_rel_nom * dt * dt
        h_now = float(np.dot(r, r)) - sk * sk
        h_next_nom = float(np.dot(r_pred, r_pred)) - s_next[(i, j)] ** 2

        grad = r_pred * (alpha * dt * dt)
        c = np.zeros(2 * n_agents, dtype=np.float64)
        c[2 * ii : 2 * ii + 2] = grad
        c[2 * jj : 2 * jj + 2] = -grad
        b = (1.0 - gamma) * h_now - h_next_nom + float(np.dot(c, x_nom))
        halfspaces.append((c, b))

    corrections = [np.zeros_like(x) for _ in range(len(halfspaces) + 1)]
    x0 = x.copy()
    for _ in range(max_iters):
        cur = x0
        for idx in range(len(halfspaces) + 1):
            y = cur + corrections[idx]
            if idx == 0:
                proj = _project_box(
                    y,
                    a_max,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                )
            else:
                c, b = halfspaces[idx - 1]
                proj = _project_halfspace(y, c, b)
            corrections[idx] = y - proj
            cur = proj
        x0 = cur

    feasible = bool(np.all(np.abs(x0[~fixed_mask]) <= a_max + 1e-6))
    feasible = feasible and bool(
        np.allclose(x0[fixed_mask], fixed_values[fixed_mask], atol=1e-6, rtol=0.0)
    )
    for c, b in halfspaces:
        if float(np.dot(c, x0)) < b - 1e-6:
            feasible = False
            break

    a_safe = _fallback_accelerations(
        x=x0,
        feasible=feasible,
        drone_ids=drone_ids,
        velocities=velocities,
        a_max=a_max,
        infeasible_fallback="velocity_cancel",
        fixed_accelerations=fixed,
    )
    return a_safe, feasible, max_iters
