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


def _project_box(x: np.ndarray, a_max: float) -> np.ndarray:
    return np.clip(x, -a_max, a_max)


def _project_halfspace(x: np.ndarray, c: np.ndarray, b: float) -> np.ndarray:
    residual = b - float(np.dot(c, x))
    if residual > 1e-9:
        norm_sq = float(np.dot(c, c))
        if norm_sq > 1e-12:
            x = x + (residual / norm_sq) * c
    return x


def solve_acceleration_qp(
    *,
    a_nom: dict[int, np.ndarray],
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    d_safe: dict[tuple[int, int], float],
    k1: float,
    k2: float,
    a_max: float,
    max_iters: int = 400,
    tol: float = 1e-7,
) -> tuple[dict[int, np.ndarray], bool, int]:
    """Solve one centralized QP for every drone's safe acceleration.

    Returns ``(a_safe, feasible, iterations)``.  If the QP is infeasible the
    returned accelerations are a deterministic hard-brake fallback and
    ``feasible`` is ``False``.
    """
    drone_ids = sorted(positions)
    n_agents = len(drone_ids)

    x = np.zeros(2 * n_agents, dtype=np.float64)
    for idx, drone in enumerate(drone_ids):
        x[2 * idx : 2 * idx + 2] = np.asarray(a_nom[drone], dtype=np.float64)

    halfspaces: list[tuple[np.ndarray, float]] = []
    for (i, j), s in d_safe.items():
        c, b = _pair_constraint(
            i=i,
            j=j,
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

    # Dykstra's alternating projection onto the intersection of the box and
    # every pairwise half-space converges to the closest point to a_nom.
    corrections = [np.zeros_like(x) for _ in range(len(halfspaces) + 1)]
    x0 = x.copy()
    prev = np.full_like(x, np.inf)
    iterations = 0
    for iterations in range(1, max_iters + 1):
        cur = x0
        for idx in range(len(halfspaces) + 1):
            y = cur + corrections[idx]
            if idx == 0:
                proj = _project_box(y, a_max)
            else:
                c, b = halfspaces[idx - 1]
                proj = _project_halfspace(y, c, b)
            corrections[idx] = y - proj
            cur = proj
        x0 = cur
        if float(np.max(np.abs(x0 - prev))) < tol:
            break
        prev = x0.copy()

    feasible = True
    box_ok = np.all(np.abs(x0) <= a_max + 1e-6)
    for c, b in halfspaces:
        if float(np.dot(c, x0)) < b - 1e-6:
            feasible = False
            break
    if not box_ok:
        feasible = False

    a_safe: dict[int, np.ndarray] = {}
    for idx, drone in enumerate(drone_ids):
        if feasible:
            a_safe[drone] = x0[2 * idx : 2 * idx + 2].copy()
        else:
            a_safe[drone] = np.clip(
                -np.asarray(velocities[drone][:2], dtype=np.float64),
                -a_max,
                a_max,
            )
    return a_safe, feasible, iterations
