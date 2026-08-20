#!/usr/bin/env python3
"""Find one-step states where a PX4 tau interval changes the RA decision.

The search uses the same projected sampled-data barrier as the robust QP.  It
also evaluates the original squared-distance barrier on a dense tau grid; the
latter is an audit, not an assumption that endpoint checks are sufficient for
the non-affine squared formulation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swarm.ra.hocbf import beta_of_tau, solve_robust_sampled_data_qp
from swarm.ra.margins import (
    RuntimeAssuranceParams,
    dynamics_margin,
    communication_margin,
    perception_margin,
)


def _alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / tau) if tau > 0.0 else 1.0


def _safe(closing_speed: float, aoi: float, params: RuntimeAssuranceParams) -> float:
    return (
        params.d0
        + dynamics_margin(closing_speed, params)
        + perception_margin(0.0, 0.0, params)
        + communication_margin(aoi, params)
    )


def _state(distance: float, closing_speed: float, yielding_speed: float) -> tuple[dict, dict, dict]:
    """Moving UAV i approaches yielding UAV j along +x."""
    i, j = 0, 1
    d = distance
    positions = {i: np.array([0.0, 0.0]), j: np.array([d, 0.0])}
    velocities = {
        i: np.array([closing_speed + yielding_speed, 0.0]),
        j: np.array([yielding_speed, 0.0]),
    }
    # Right-of-way vehicle keeps its current command; the yielding vehicle is
    # commanded to stop while its actual velocity is still nonzero.
    nominal_velocity = {
        i: velocities[i].copy(),
        j: np.zeros(2),
    }
    return positions, velocities, nominal_velocity


def _next_safe(
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    accelerations: dict[int, np.ndarray],
    dt: float,
    tau_i: float,
    tau_j: float,
    params: RuntimeAssuranceParams,
    aoi: float,
) -> tuple[np.ndarray, float]:
    ids = (0, 1)
    taus = (tau_i, tau_j)
    next_positions = {}
    next_velocities = {}
    for drone, tau in zip(ids, taus):
        alpha = _alpha(dt, tau)
        beta = beta_of_tau(dt, tau)
        v = velocities[drone]
        a = accelerations[drone]
        next_velocities[drone] = v + alpha * a * dt
        next_positions[drone] = positions[drone] + v * dt + beta * a * dt
    r = next_positions[0] - next_positions[1]
    v = next_velocities[0] - next_velocities[1]
    distance = float(np.linalg.norm(r))
    closing = max(0.0, -float(np.dot(r, v)) / distance) if distance > 1e-12 else 0.0
    return r, _safe(closing, aoi, params)


def _projected_g(
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    accelerations: dict[int, np.ndarray],
    dt: float,
    gamma: float,
    tau_i: float,
    tau_j: float,
    s_now: float,
    s_next: float,
) -> float:
    r = positions[0] - positions[1]
    v = velocities[0] - velocities[1]
    distance = float(np.linalg.norm(r))
    n = r / distance
    r_next = r + v * dt + dt * (
        beta_of_tau(dt, tau_i) * accelerations[0]
        - beta_of_tau(dt, tau_j) * accelerations[1]
    )
    return float(np.dot(n, r_next) - s_next - (1.0 - gamma) * (np.dot(n, r) - s_now))


def _exact_g(
    positions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    accelerations: dict[int, np.ndarray],
    dt: float,
    gamma: float,
    tau_i: float,
    tau_j: float,
    s_now: float,
    params: RuntimeAssuranceParams,
    aoi: float,
) -> float:
    r_next, s_next = _next_safe(
        positions, velocities, accelerations, dt, tau_i, tau_j, params, aoi
    )
    h_now = float(np.dot(positions[0] - positions[1], positions[0] - positions[1])) - s_now**2
    h_next = float(np.dot(r_next, r_next)) - s_next**2
    return h_next - (1.0 - gamma) * h_now


def _projected_metrics(
    *, delta: float, closing_speed: float, yielding_speed: float,
    dt: float, gamma: float, tau_hat: float, tau_min: float, tau_max: float,
    params: RuntimeAssuranceParams, kv: float, a_max: float,
) -> tuple[dict, dict, dict, float, float, float, float]:
    s_now = _safe(closing_speed, 0.0, params)
    positions, velocities, nominal_velocity = _state(
        s_now + delta, closing_speed, yielding_speed
    )
    a_nom = {
        drone: np.clip(kv * (nominal_velocity[drone] - velocities[drone]), -a_max, a_max)
        for drone in (0, 1)
    }
    alpha_hat = _alpha(dt, tau_hat)
    v_hat = {
        drone: velocities[drone] + alpha_hat * a_nom[drone] * dt
        for drone in (0, 1)
    }
    r_hat = positions[0] - positions[1]
    vcl_hat = max(0.0, -float(np.dot(r_hat, v_hat[0] - v_hat[1])) / np.linalg.norm(r_hat))
    s_next = _safe(vcl_hat, 0.0, params)
    g_nominal = _projected_g(
        positions, velocities, a_nom, dt, gamma, tau_hat, tau_hat, s_now, s_next
    )
    vertex_g = [
        _projected_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, s_next)
        for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
    ]
    return positions, velocities, a_nom, s_now, s_next, g_nominal, min(vertex_g)


def _exact_metrics(*, delta: float, closing_speed: float, yielding_speed: float,
                   dt: float, gamma: float, tau_hat: float, tau_min: float,
                   tau_max: float, params: RuntimeAssuranceParams, kv: float,
                   a_max: float) -> tuple[float, float]:
    positions, velocities, a_nom, s_now, _, _, _ = _projected_metrics(
        delta=delta, closing_speed=closing_speed, yielding_speed=yielding_speed,
        dt=dt, gamma=gamma, tau_hat=tau_hat, tau_min=tau_min, tau_max=tau_max,
        params=params, kv=kv, a_max=a_max,
    )
    nominal = _exact_g(
        positions, velocities, a_nom, dt, gamma, tau_hat, tau_hat,
        s_now, params, 0.0,
    )
    worst = min(
        _exact_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, params, 0.0)
        for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
    )
    return nominal, worst


def _robust_projected_metrics(*, delta: float, closing_speed: float,
                              yielding_speed: float, dt: float, gamma: float,
                              tau_hat: float, tau_min: float, tau_max: float,
                              params: RuntimeAssuranceParams, kv: float,
                              a_max: float) -> tuple[float, float]:
    positions, velocities, a_nom, s_now, s_next, nominal, _ = _projected_metrics(
        delta=delta, closing_speed=closing_speed, yielding_speed=yielding_speed,
        dt=dt, gamma=gamma, tau_hat=tau_hat, tau_min=tau_min, tau_max=tau_max,
        params=params, kv=kv, a_max=a_max,
    )
    s_next_robust = max(
        _next_safe(positions, velocities, a_nom, dt, ti, tj, params, 0.0)[1]
        for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
    )
    worst = min(
        _projected_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, s_next_robust)
        for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)
    )
    return nominal, worst


def evaluate_case(
    *,
    delta: float,
    closing_speed: float,
    yielding_speed: float,
    dt: float,
    gamma: float,
    tau_hat: float,
    tau_min: float,
    tau_max: float,
    dense_points: int,
    params: RuntimeAssuranceParams,
    kv: float,
    a_max: float,
) -> dict | None:
    positions, velocities, a_nom, s_now, s_next, g_nominal, worst_g = _projected_metrics(
        delta=delta, closing_speed=closing_speed, yielding_speed=yielding_speed,
        dt=dt, gamma=gamma, tau_hat=tau_hat, tau_min=tau_min, tau_max=tau_max,
        params=params, kv=kv, a_max=a_max,
    )
    vertices = [(ti, tj) for ti in (tau_min, tau_max) for tj in (tau_min, tau_max)]
    vertex_g = {
        f"{ti:.6g},{tj:.6g}": _projected_g(
            positions, velocities, a_nom, dt, gamma, ti, tj, s_now, s_next
        )
        for ti, tj in vertices
    }
    worst_vertex = min(vertex_g, key=vertex_g.get)
    worst_g = vertex_g[worst_vertex]
    exact_g_nominal, exact_g_worst = _exact_metrics(
        delta=delta, closing_speed=closing_speed, yielding_speed=yielding_speed,
        dt=dt, gamma=gamma, tau_hat=tau_hat, tau_min=tau_min, tau_max=tau_max,
        params=params, kv=kv, a_max=a_max,
    )
    projected_g_nominal, projected_g_worst_robust = _robust_projected_metrics(
        delta=delta, closing_speed=closing_speed, yielding_speed=yielding_speed,
        dt=dt, gamma=gamma, tau_hat=tau_hat, tau_min=tau_min, tau_max=tau_max,
        params=params, kv=kv, a_max=a_max,
    )
    exact_vertex_g = {
        f"{ti:.6g},{tj:.6g}": _exact_g(
            positions, velocities, a_nom, dt, gamma, ti, tj, s_now, params, 0.0
        )
        for ti, tj in vertices
    }
    exact_worst_vertex = min(exact_vertex_g, key=exact_vertex_g.get)
    if not (projected_g_nominal >= 0.0 and projected_g_worst_robust < 0.0):
        return None

    nominal_safe, nominal_feasible, _ = solve_robust_sampled_data_qp(
        a_nom=a_nom, positions=positions, velocities=velocities,
        s_now={(0, 1): s_now}, s_next={(0, 1): s_next}, dt=dt,
        gamma=gamma, a_max=a_max, beta={0: (beta_of_tau(dt, tau_hat),) * 2, 1: (beta_of_tau(dt, tau_hat),) * 2},
    )
    s_next_robust = max(
        _next_safe(positions, velocities, a_nom, dt, ti, tj, params, 0.0)[1]
        for ti, tj in vertices
    )
    robust_safe, robust_feasible, _ = solve_robust_sampled_data_qp(
        a_nom=a_nom, positions=positions, velocities=velocities,
        s_now={(0, 1): s_now}, s_next={(0, 1): s_next_robust}, dt=dt,
        gamma=gamma, a_max=a_max,
        beta={0: (beta_of_tau(dt, tau_min), beta_of_tau(dt, tau_max)),
              1: (beta_of_tau(dt, tau_min), beta_of_tau(dt, tau_max))},
    )
    delta_u = max(float(np.linalg.norm(robust_safe[i] - nominal_safe[i])) for i in (0, 1))
    taus = np.linspace(tau_min, tau_max, dense_points)
    dense_nominal_action = np.array([
        _exact_g(positions, velocities, nominal_safe, dt, gamma, ti, tj, s_now, params, 0.0)
        for ti in taus for tj in taus
    ])
    dense_robust_action = np.array([
        _exact_g(positions, velocities, robust_safe, dt, gamma, ti, tj, s_now, params, 0.0)
        for ti in taus for tj in taus
    ])
    r_now = positions[0] - positions[1]
    n_now = r_now / np.linalg.norm(r_now)
    b_now = float(np.dot(n_now, r_now) - s_now)
    dense_projected_robust = []
    dense_hsq_robust = []
    dense_projected_nominal = []
    dense_hsq_nominal = []
    for ti in taus:
        for tj in taus:
            r_next_robust, s_next_tau = _next_safe(
                positions, velocities, robust_safe, dt, ti, tj, params, 0.0
            )
            r_next_nominal, _ = _next_safe(
                positions, velocities, nominal_safe, dt, ti, tj, params, 0.0
            )
            dense_projected_robust.append(
                float(np.dot(n_now, r_next_robust) - s_next_robust - (1.0 - gamma) * b_now)
            )
            dense_projected_nominal.append(
                float(np.dot(n_now, r_next_nominal) - s_next_robust - (1.0 - gamma) * b_now)
            )
            dense_hsq_robust.append(float(np.dot(r_next_robust, r_next_robust) - s_next_tau**2))
            dense_hsq_nominal.append(float(np.dot(r_next_nominal, r_next_nominal) - s_next_tau**2))
    dense = np.array([
        _exact_g(positions, velocities, a_nom, dt, gamma, ti, tj, s_now, params, 0.0)
        for ti in taus for tj in taus
    ])
    robust_idx = int(np.argmin(dense_robust_action))
    return {
        "delta_m": delta,
        "closing_speed_mps": closing_speed,
        "yielding_speed_mps": yielding_speed,
        "s_now_m": s_now,
        "s_next_nominal_m": s_next,
        "s_next_robust_m": s_next_robust,
        "tau_hat": tau_hat,
        "g_nominal_projected": g_nominal,
        "g_worst_vertex_projected": worst_g,
        "g_nominal_exact": exact_g_nominal,
        "g_worst_vertex_exact": exact_g_worst,
        "g_nominal_projected_controller": projected_g_nominal,
        "g_worst_projected_robust_controller": projected_g_worst_robust,
        "worst_vertex_tau_projected": [float(x) for x in map(float, worst_vertex.split(","))],
        "worst_vertex_tau_exact": [float(x) for x in map(float, exact_worst_vertex.split(","))],
        "dense_exact_g_min": float(dense.min()),
        "dense_exact_argmin_tau": [float(taus[idx // dense_points]), float(taus[idx % dense_points])]
        if (idx := int(np.argmin(dense))) >= 0 else None,
        "dense_exact_g_min_nominal_action": float(dense_nominal_action.min()),
        "dense_exact_g_min_robust_action": float(dense_robust_action.min()),
        "dense_exact_robust_action_argmin_tau": [
            float(taus[robust_idx // dense_points]),
            float(taus[robust_idx % dense_points]),
        ],
        "b_projected_now": b_now,
        "dense_projected_margin_min_robust_action": float(min(dense_projected_robust)),
        "dense_projected_margin_min_nominal_action": float(min(dense_projected_nominal)),
        "dense_hsq_min_robust_action": float(min(dense_hsq_robust)),
        "dense_hsq_min_nominal_action": float(min(dense_hsq_nominal)),
        "nominal_feasible": bool(nominal_feasible),
        "robust_feasible": bool(robust_feasible),
        "nominal_acceleration": {str(i): nominal_safe[i].tolist() for i in (0, 1)},
        "robust_acceleration": {str(i): robust_safe[i].tolist() for i in (0, 1)},
        "delta_u_mps2": delta_u,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-min", type=float, default=0.53)
    parser.add_argument("--tau-max", type=float, default=1.76)
    parser.add_argument("--tau-hat", type=float, default=0.7)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--dense-points", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    params = RuntimeAssuranceParams()
    cases = []
    for yielding_speed in (0.1, 0.3, 0.6):
        for closing_speed in np.linspace(0.2, 1.4, 25):
            kwargs = dict(
                closing_speed=float(closing_speed), yielding_speed=yielding_speed,
                dt=args.dt, gamma=args.gamma, tau_hat=args.tau_hat,
                tau_min=args.tau_min, tau_max=args.tau_max,
                params=params, kv=2.0, a_max=2.0,
            )
            grid = np.linspace(0.05, 1.0, 40)
            values = [
                _robust_projected_metrics(delta=float(delta), **kwargs)[0]
                for delta in grid
            ]
            for left, right, f_left, f_right in zip(
                grid[:-1], grid[1:], values[:-1], values[1:]
            ):
                if f_left * f_right > 0.0:
                    continue
                lo, hi = float(left), float(right)
                for _ in range(50):
                    mid = (lo + hi) / 2.0
                    f_mid = _robust_projected_metrics(delta=mid, **kwargs)[0]
                    if f_left * f_mid <= 0.0:
                        hi, f_right = mid, f_mid
                    else:
                        lo, f_left = mid, f_mid
                # Move just inside the nominal-safe side.  The robust gap is
                # intentionally narrow; a coarse grid would miss it.
                root = (lo + hi) / 2.0
                for delta in (root + 1e-6, root + 1e-5, root + 5e-5):
                    case = evaluate_case(delta=delta, dense_points=args.dense_points, **kwargs)
                    if case is not None:
                        cases.append(case)
    cases.sort(key=lambda c: (c["delta_u_mps2"] <= 1e-8, -c["delta_u_mps2"], c["g_worst_vertex_projected"]))
    feasible_cases = [c for c in cases if c["robust_feasible"]]
    projected_gate_cases = [
        c for c in feasible_cases
        if c["dense_projected_margin_min_robust_action"] >= -1e-9
    ]
    hsq_gate_cases = [
        c for c in feasible_cases if c["dense_hsq_min_robust_action"] >= -1e-9
    ]
    result = {
        "tau_interval": [args.tau_min, args.tau_max],
        "tau_hat": args.tau_hat,
        "dt": args.dt,
        "gamma": args.gamma,
        "dense_points": args.dense_points,
        "activation_count": len(cases),
        "delta_u_positive_count": sum(c["delta_u_mps2"] > 1e-8 for c in cases),
        "robust_feasible_count": len(feasible_cases),
        "robust_infeasible_count": len(cases) - len(feasible_cases),
        "projected_dense_gate_pass_count": len(projected_gate_cases),
        "squared_h_dense_gate_pass_count": len(hsq_gate_cases),
        "all_cases_projected_dense_gate_pass": len(projected_gate_cases) == len(cases),
        "feasible_cases_projected_dense_gate_pass": len(projected_gate_cases) == len(feasible_cases),
        "feasible_cases_squared_h_dense_gate_pass": len(hsq_gate_cases) == len(feasible_cases),
        "cases": cases[: args.top_k],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "activation_count": result["activation_count"],
        "delta_u_positive_count": result["delta_u_positive_count"],
        "robust_feasible_count": result["robust_feasible_count"],
        "robust_infeasible_count": result["robust_infeasible_count"],
        "projected_dense_gate_pass_count": result["projected_dense_gate_pass_count"],
        "squared_h_dense_gate_pass_count": result["squared_h_dense_gate_pass_count"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
