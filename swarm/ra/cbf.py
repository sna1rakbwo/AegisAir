"""Time-varying CBF / QP velocity filter (plan.md section 19).

The lightweight environment controls 2D velocity, so the filter solves a small
quadratic program: minimally perturb the nominal MARL velocity while keeping
every pairwise CBF constraint satisfied.

    h_ij = ||p_i - p_j||^2 - d_safe^2
    2 (p_i - p_j) . u_i >= 2 (p_i - p_j) . v_j - alpha * h_ij
"""

from __future__ import annotations

import numpy as np


def cbf_constraint(
    p_i: np.ndarray,
    p_j: np.ndarray,
    v_j: np.ndarray,
    d_safe: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Return the linear half-space ``a . u_i >= b`` for the (i, j) pair."""
    p_ij = np.asarray(p_i, dtype=np.float64) - np.asarray(p_j, dtype=np.float64)
    v_j = np.asarray(v_j, dtype=np.float64)
    d2 = float(np.dot(p_ij, p_ij))
    h = d2 - d_safe * d_safe
    a = 2.0 * p_ij
    b = 2.0 * float(np.dot(p_ij, v_j)) - alpha * h
    return a, b


def project_safe_action(
    u_nom: np.ndarray,
    constraints: list[tuple[np.ndarray, float]],
    v_max: float,
    max_iters: int = 32,
) -> np.ndarray:
    """Sequential projection of u_nom onto the intersection of CBF half-spaces."""
    u = np.asarray(u_nom, dtype=np.float64).copy()
    for _ in range(max_iters):
        changed = False
        for a, b in constraints:
            residual = b - float(np.dot(a, u))
            if residual > 1e-9:
                norm_sq = float(np.dot(a, a))
                if norm_sq > 1e-9:
                    u = u + (residual / norm_sq) * a
                    changed = True
        if not changed:
            break
    return np.clip(u, -v_max, v_max)
