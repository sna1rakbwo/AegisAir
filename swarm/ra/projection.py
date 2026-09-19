"""Small deterministic Euclidean projection primitives for RA QPs."""

from __future__ import annotations

import numpy as np


def project_box(
    x: np.ndarray,
    a_max: float,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
) -> np.ndarray:
    projected = np.clip(x, -a_max, a_max)
    if fixed_mask is not None:
        if fixed_values is None:
            raise ValueError("fixed_values are required with fixed_mask")
        projected[fixed_mask] = fixed_values[fixed_mask]
    return projected


def project_halfspace(x: np.ndarray, c: np.ndarray, b: float) -> np.ndarray:
    residual = b - float(np.dot(c, x))
    if residual > 1e-9:
        norm_sq = float(np.dot(c, c))
        if norm_sq > 1e-12:
            return x + residual * c / norm_sq
    return x


def project_box_and_single_halfspace(
    *,
    x_nom: np.ndarray,
    c: np.ndarray,
    b: float,
    a_max: float,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    """Exactly project onto a box intersected with one half-space.

    The KKT solution has ``x = clip(x_nom + lambda*c)`` on free
    coordinates. Solving the one-dimensional monotone dual avoids thousands
    of alternating-projection iterations for the common two-UAV case.
    """
    projected = project_box(
        x_nom,
        a_max,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    if float(np.dot(c, projected)) >= b - 1e-9:
        return projected, True, 0

    free = ~fixed_mask
    maximizing = projected.copy()
    positive = free & (c > 0.0)
    negative = free & (c < 0.0)
    maximizing[positive] = a_max
    maximizing[negative] = -a_max
    if float(np.dot(c, maximizing)) < b - 1e-9:
        return projected, False, 0

    nonzero = free & (np.abs(c) > 1e-15)
    targets = np.where(c >= 0.0, a_max, -a_max)
    breakpoints = (targets[nonzero] - x_nom[nonzero]) / c[nonzero]
    upper = max(0.0, float(np.max(breakpoints, initial=0.0)))
    if upper == 0.0:
        upper = 1.0
        while float(
            np.dot(
                c,
                project_box(
                    x_nom + upper * c,
                    a_max,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                ),
            )
        ) < b:
            upper *= 2.0

    lower = 0.0
    iterations = 0
    for iterations in range(1, 65):
        dual = 0.5 * (lower + upper)
        candidate = project_box(
            x_nom + dual * c,
            a_max,
            fixed_mask=fixed_mask,
            fixed_values=fixed_values,
        )
        if float(np.dot(c, candidate)) >= b:
            upper = dual
            projected = candidate
        else:
            lower = dual
        if upper - lower <= 1e-12 * max(1.0, upper):
            break
    feasible = float(np.dot(c, projected)) >= b - 1e-6
    return projected, feasible, iterations
