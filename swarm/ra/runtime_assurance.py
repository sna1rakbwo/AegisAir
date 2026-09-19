"""Risk-Adaptive Runtime Assurance orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swarm.ra.cbf import cbf_constraint, project_safe_action
from swarm.ra.hocbf import (
    beta_of_tau,
    minimum_acceleration_box_reserve,
    solve_acceleration_qp,
    solve_robust_sampled_data_qp,
    solve_sampled_data_qp,
)
from swarm.ra.margin import PairMarginTracker, normalized_margin
from swarm.ra.margins import (
    RuntimeAssuranceParams,
    closing_speed,
    communication_margin,
    dynamic_safety_boundary,
    dynamics_margin,
    perception_margin,
)
from swarm.ra.predictor import PredictiveMonitor
from swarm.ra.pcbf import PCBFConfig, solve_pcbf
from swarm.ra.sota_cbf import solve_prediction_based_cbf_qp, solve_zocbf_qp


@dataclass
class FilterResult:
    drone: int
    mode: str
    nominal_action: tuple[float, float]
    safe_action: tuple[float, float]
    safety_margin: float
    predicted_margin: float
    time_to_min_margin: float
    time_to_safety_boundary: float | None
    prediction_reliability_score: float
    recovery_buffer: float | None
    semantic_recovery_feasible: bool
    degradation: float
    worst_pair: int | None
    intervened: bool
    proactive: bool
    # Diagnostics (populated by the sampled-data path; optional elsewhere).
    a_nom: tuple[float, float] | None = None
    a_safe: tuple[float, float] | None = None
    d_safe: float | None = None
    accel_saturated: bool = False
    vel_saturated: bool = False
    feasible: bool | None = None
    selected_filter: str | None = None
    primary_feasible: bool | None = None
    predictive_feasible: bool | None = None
    recovery_active: bool = False
    recovery_reason: str | None = None
    feasibility_reserve: float | None = None
    pcbf_status: str | None = None
    pcbf_stage1_status: str | None = None
    pcbf_stage2_status: str | None = None
    pcbf_terminal_feasible: bool | None = None
    pcbf_value: float | None = None
    pcbf_slack_sum: float | None = None
    pcbf_tracking_cost: float | None = None
    pcbf_max_constraint_violation: float | None = None
    pcbf_tie_break_applied: bool | None = None
    pcbf_fail_closed_reason: str | None = None
    control_authority: bool = True
    fixed_action: tuple[float, float] | None = None


class RuntimeAssurance:
    """Filter nominal MARL velocity commands through the CBF safety set."""

    def __init__(
        self,
        params: RuntimeAssuranceParams | None = None,
        v_max: float = 2.0,
        perception_sigma: float = 0.1,
        rho_warn: float = 0.2,
        use_hocbf: bool = False,
        hocbf_k1: float = 1.0,
        hocbf_k2: float = 1.0,
        a_max: float = 2.0,
        kv: float = 2.0,
        sampled_data: bool = False,
        gamma: float = 0.1,
        tau_px4: float = 0.0,
        tau_px4_min: float | None = None,
        tau_px4_max: float | None = None,
        execution_model: str = "exact_zoh",
        sampled_data_method: str = "aegis",
        pcbf_horizon: int = 20,
        pcbf_terminal_buffer_m: float = 0.10,
        pcbf_terminal_velocity_tolerance_mps: float = 0.0,
        pcbf_position_bound_m: float = 20.0,
        pcbf_velocity_bound_mps: float = 5.0,
        pcbf_max_iterations: int = 300,
        pcbf_multistart_count: int = 3,
        pcbf_tolerance: float = 1e-7,
        pcbf_acceptable_tolerance: float = 1e-5,
        pcbf_lexicographic_tolerance: float = 1e-7,
        zocbf_delta: float = 0.0,
        pb_alpha: float = 2.0,
        pb_braking_accel: float | None = None,
        command_feedforward_tau_s: float | None = None,
        constraint_boundary: str = "full",
        hocbf_boundary_guard: float = 0.0,
        hocbf_boundary_buffer_m: float = 0.0,
        hocbf_infeasible_fallback: str = "velocity_cancel",
        hocbf_pb_recovery: bool = False,
        hocbf_predictive_recovery: bool = True,
        hocbf_prediction_execution_fraction: float = 0.0,
        hocbf_prediction_steps: int = 1,
        hocbf_recovery_reserve_threshold: float | None = None,
        hocbf_recovery_alpha: float = 0.5,
        hocbf_recovery_braking_accel: float | None = None,
        hocbf_recovery_boundary_buffer_m: float = 0.0,
        hocbf_recovery_clear_steps: int = 5,
        sampled_data_boundary_buffer_m: float = 0.0,
        sampled_data_infeasible_fallback: str = "velocity_cancel",
        qp_max_iters: int = 3000,
    ) -> None:
        self.params = params or RuntimeAssuranceParams()
        self.v_max = v_max
        self.perception_sigma = perception_sigma
        self.rho_warn = rho_warn
        self.use_hocbf = use_hocbf
        self.k1 = hocbf_k1
        self.k2 = hocbf_k2
        self.a_max = a_max
        self.kv = kv
        self.sampled_data = sampled_data
        self.gamma = gamma
        self.tau_px4 = tau_px4
        self.tau_px4_min = tau_px4_min if tau_px4_min is not None else tau_px4
        self.tau_px4_max = tau_px4_max if tau_px4_max is not None else tau_px4
        if execution_model not in {"exact_zoh", "legacy_trapezoidal"}:
            raise ValueError("unknown execution_model")
        self.execution_model = execution_model
        if sampled_data_method not in {"aegis", "zocbf", "pb_cbf", "pcbf"}:
            raise ValueError("unknown sampled_data_method")
        if zocbf_delta < 0.0:
            raise ValueError("zocbf_delta must be non-negative")
        if pb_alpha <= 0.0:
            raise ValueError("pb_alpha must be positive")
        self.sampled_data_method = sampled_data_method
        self.pcbf_config = PCBFConfig(
            horizon=pcbf_horizon,
            dt=self.params.degradation_dt,
            a_max=a_max,
            terminal_buffer_m=pcbf_terminal_buffer_m,
            terminal_velocity_tolerance_mps=pcbf_terminal_velocity_tolerance_mps,
            position_bound_m=pcbf_position_bound_m,
            velocity_bound_mps=pcbf_velocity_bound_mps,
            max_iterations=pcbf_max_iterations,
            multistart_count=pcbf_multistart_count,
            tolerance=pcbf_tolerance,
            acceptable_tolerance=pcbf_acceptable_tolerance,
            lexicographic_tolerance=pcbf_lexicographic_tolerance,
        )
        self.zocbf_delta = zocbf_delta
        self.pb_alpha = pb_alpha
        self.pb_braking_accel = (
            self.params.a_eff if pb_braking_accel is None else pb_braking_accel
        )
        if self.pb_braking_accel <= 0.0:
            raise ValueError("pb_braking_accel must be positive")
        if command_feedforward_tau_s is not None and command_feedforward_tau_s <= 0.0:
            raise ValueError("command_feedforward_tau_s must be positive")
        self.command_feedforward_tau_s = command_feedforward_tau_s
        if constraint_boundary not in {"full", "static"}:
            raise ValueError("constraint_boundary must be full or static")
        self.constraint_boundary = constraint_boundary
        if hocbf_boundary_guard < 0.0 or hocbf_boundary_buffer_m < 0.0:
            raise ValueError("HOCBF boundary guard and buffer must be non-negative")
        self.hocbf_boundary_guard = hocbf_boundary_guard
        self.hocbf_boundary_buffer_m = hocbf_boundary_buffer_m
        if hocbf_infeasible_fallback not in {"velocity_cancel", "max_brake"}:
            raise ValueError("unknown HOCBF infeasible fallback")
        self.hocbf_infeasible_fallback = hocbf_infeasible_fallback
        if hocbf_recovery_alpha <= 0.0:
            raise ValueError("HOCBF recovery alpha must be positive")
        recovery_braking = (
            self.a_max
            if hocbf_recovery_braking_accel is None
            else hocbf_recovery_braking_accel
        )
        if recovery_braking <= 0.0:
            raise ValueError("HOCBF recovery braking acceleration must be positive")
        if hocbf_recovery_boundary_buffer_m < 0.0:
            raise ValueError("HOCBF recovery boundary buffer must be non-negative")
        if hocbf_recovery_clear_steps < 1:
            raise ValueError("HOCBF recovery clear steps must be positive")
        self.hocbf_pb_recovery = hocbf_pb_recovery
        self.hocbf_predictive_recovery = hocbf_predictive_recovery
        if not 0.0 <= hocbf_prediction_execution_fraction <= 1.0:
            raise ValueError("HOCBF prediction execution fraction must be in [0, 1]")
        self.hocbf_prediction_execution_fraction = (
            hocbf_prediction_execution_fraction
        )
        if hocbf_prediction_steps < 1:
            raise ValueError("HOCBF prediction steps must be positive")
        self.hocbf_prediction_steps = hocbf_prediction_steps
        if (
            hocbf_recovery_reserve_threshold is not None
            and not np.isfinite(hocbf_recovery_reserve_threshold)
        ):
            raise ValueError("HOCBF recovery reserve threshold must be finite")
        self.hocbf_recovery_reserve_threshold = (
            hocbf_recovery_reserve_threshold
        )
        self.hocbf_recovery_alpha = hocbf_recovery_alpha
        self.hocbf_recovery_braking_accel = recovery_braking
        self.hocbf_recovery_boundary_buffer_m = hocbf_recovery_boundary_buffer_m
        self.hocbf_recovery_clear_steps = hocbf_recovery_clear_steps
        if sampled_data_boundary_buffer_m < 0.0:
            raise ValueError("sampled-data boundary buffer must be non-negative")
        if sampled_data_infeasible_fallback not in {"velocity_cancel", "max_brake"}:
            raise ValueError("unknown sampled-data infeasible fallback")
        self.sampled_data_boundary_buffer_m = sampled_data_boundary_buffer_m
        self.sampled_data_infeasible_fallback = sampled_data_infeasible_fallback
        if qp_max_iters < 1:
            raise ValueError("qp_max_iters must be positive")
        self.qp_max_iters = qp_max_iters
        self.trackers: dict[tuple[int, int], PairMarginTracker] = {}
        self.monitor = PredictiveMonitor(q_pred=self.params.q_pred)
        self._last_t: float | None = None
        self.qp_solve_count = 0
        self.qp_infeasible_count = 0
        self.last_qp_feasible: bool | None = None
        self.last_primary_hocbf_feasible: bool | None = None
        self.last_predictive_hocbf_feasible: bool | None = None
        self.hocbf_primary_infeasible_count = 0
        self.hocbf_predictive_infeasible_count = 0
        self.hocbf_recovery_infeasible_count = 0
        self.hocbf_recovery_steps = 0
        self.hocbf_recovery_entries = 0
        self._hocbf_recovery_active = False
        self._hocbf_recovery_clean_steps = 0
        self._hocbf_recovery_reason: str | None = None

    def _tracker(self, i: int, j: int) -> PairMarginTracker:
        key = (min(i, j), max(i, j))
        if key not in self.trackers:
            self.trackers[key] = PairMarginTracker(self.params)
        return self.trackers[key]

    def _pair_sigmas(self, snap_i, snap_j) -> tuple[float, float]:
        """Project each estimate's covariance onto the pair line-of-sight."""
        p_i = np.asarray(snap_i.position[:2], dtype=np.float64)
        p_j = np.asarray(snap_j.position[:2], dtype=np.float64)
        delta = p_i - p_j
        dist = float(np.linalg.norm(delta))
        direction = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
        return (
            _projected_sigma(snap_i, direction, self.perception_sigma),
            _projected_sigma(snap_j, direction, self.perception_sigma),
        )

    def filter(
        self,
        snapshots: dict[int, object],
        nominal_actions: dict[int, np.ndarray],
        t: float,
        aoi: dict[tuple[int, int], float] | None = None,
        fixed_actions: dict[int, np.ndarray] | None = None,
    ) -> dict[int, FilterResult]:
        """Filter commands while keeping revoked agents as fixed obstacles.

        ``fixed_actions`` contains the exact planar velocity commands that the
        execution layer will publish for agents without control authority.
        These agents remain in every pairwise constraint, but joint solvers
        cannot change their inputs or credit their control budget.
        """
        fixed_actions = _normalize_fixed_actions(snapshots, fixed_actions)
        if self.sampled_data:
            return self._filter_sampled_data(
                snapshots, nominal_actions, t, aoi, fixed_actions
            )
        if self.use_hocbf:
            return self._filter_hocbf(
                snapshots, nominal_actions, t, aoi, fixed_actions
            )
        aoi = aoi or {}
        results: dict[int, FilterResult] = {}

        dt = self.params.degradation_dt
        if self._last_t is not None:
            dt = max(1e-3, t - self._last_t)
        self._last_t = t
        for agent_id, snapshot in snapshots.items():
            if snapshot.velocity is not None:
                self.monitor.update_acceleration(agent_id, snapshot.velocity, dt)

        for drone_id, snapshot in snapshots.items():
            p_i = np.asarray(snapshot.position[:2], dtype=np.float64)
            v_i = np.asarray(snapshot.velocity[:2], dtype=np.float64) if snapshot.velocity else np.zeros(2)
            u_nom = np.asarray(nominal_actions[drone_id], dtype=np.float64)

            constraints: list[tuple[np.ndarray, float]] = []
            worst_rho = float("inf")
            worst_g = 0.0
            worst_pair: int | None = None
            worst_pred_rho = float("inf")
            worst_pred_tau = 0.0
            worst_ttsb: float | None = None
            worst_confidence = 1.0

            for other_id, other in snapshots.items():
                if other_id == drone_id:
                    continue
                p_j = np.asarray(other.position[:2], dtype=np.float64)
                v_j = np.asarray(other.velocity[:2], dtype=np.float64) if other.velocity else np.zeros(2)

                distance = float(np.linalg.norm(p_i - p_j))
                v_cl = closing_speed(snapshot.position, other.position, snapshot.velocity or (0.0, 0.0, 0.0), other.velocity or (0.0, 0.0, 0.0))
                pair_aoi = aoi.get((drone_id, other_id), 0.0)
                sigma_i, sigma_j = self._pair_sigmas(snapshot, other)
                d_reference = dynamic_safety_boundary(
                    closing_speed=v_cl,
                    perception_sigma_i=sigma_i,
                    perception_sigma_j=sigma_j,
                    aoi=pair_aoi,
                    params=self.params,
                )
                d_constraint = (
                    d_reference
                    if self.constraint_boundary == "full"
                    else self.params.d0
                    + perception_margin(sigma_i, sigma_j, self.params)
                    + communication_margin(pair_aoi, self.params)
                )
                rho = normalized_margin(distance, d_reference)
                tracker = self._tracker(drone_id, other_id)
                g = tracker.update(rho, t)

                a, b = cbf_constraint(
                    p_i, p_j, v_j, d_constraint, self.params.alpha
                )
                constraints.append((a, b))

                if rho < worst_rho:
                    worst_rho = rho
                    worst_pair = other_id
                    worst_g = g

                pred = self.monitor.predict(
                    agent_i=drone_id,
                    agent_j=other_id,
                    p_i=snapshot.position,
                    p_j=other.position,
                    v_i=snapshot.velocity or (0.0, 0.0, 0.0),
                    v_j=other.velocity or (0.0, 0.0, 0.0),
                    sigma_i=sigma_i,
                    sigma_j=sigma_j,
                    aoi=pair_aoi,
                    params=self.params,
                    horizon=self.params.prediction_horizon,
                )
                if pred.rho_min_pred < worst_pred_rho:
                    worst_pred_rho = pred.rho_min_pred
                    worst_pred_tau = pred.tau_star
                    worst_ttsb = pred.ttsb
                    worst_confidence = pred.reliability_score

            u_safe = (
                fixed_actions[drone_id]
                if drone_id in fixed_actions
                else project_safe_action(u_nom, constraints, self.v_max)
            )
            intervened = bool(np.linalg.norm(u_safe - u_nom) > 1e-6)

            if intervened:
                mode = "override"
            elif worst_rho < self.rho_warn:
                mode = "warning"
            else:
                mode = "normal"

            results[drone_id] = FilterResult(
                drone=drone_id,
                mode=mode,
                nominal_action=(float(u_nom[0]), float(u_nom[1])),
                safe_action=(float(u_safe[0]), float(u_safe[1])),
                safety_margin=float(worst_rho),
                predicted_margin=float(worst_pred_rho),
                time_to_min_margin=float(worst_pred_tau),
                time_to_safety_boundary=worst_ttsb,
                prediction_reliability_score=float(worst_confidence),
                recovery_buffer=(
                    worst_ttsb - self.params.estimated_recovery_latency
                    if worst_ttsb is not None
                    else None
                ),
                semantic_recovery_feasible=(
                    worst_ttsb is not None
                    and worst_ttsb > self.params.estimated_recovery_latency
                ),
                degradation=float(worst_g),
                worst_pair=worst_pair,
                intervened=intervened,
                proactive=worst_pred_rho < self.params.rho_pred_threshold,
                control_authority=drone_id not in fixed_actions,
                fixed_action=(
                    tuple(float(value) for value in fixed_actions[drone_id])
                    if drone_id in fixed_actions
                    else None
                ),
            )
        return results

    def _filter_sampled_data(
        self,
        snapshots: dict[int, object],
        nominal_actions: dict[int, np.ndarray],
        t: float,
        aoi: dict[tuple[int, int], float] | None,
        fixed_actions: dict[int, np.ndarray],
    ) -> dict[int, FilterResult]:
        """Sampled-data acceleration-aware filter."""
        aoi = aoi or {}
        dt = self.params.degradation_dt
        if self._last_t is not None:
            dt = max(1e-3, t - self._last_t)
        self._last_t = t
        for agent_id, snapshot in snapshots.items():
            if snapshot.velocity is not None:
                self.monitor.update_acceleration(agent_id, snapshot.velocity, dt)

        drone_ids = sorted(snapshots)
        command_scale = (
            dt
            if self.command_feedforward_tau_s is None
            else self.command_feedforward_tau_s
        )
        beta = {
            i: (
                beta_of_tau(dt, self.tau_px4_max),
                beta_of_tau(dt, self.tau_px4_min),
            )
            for i in drone_ids
        }
        positions = {
            i: np.asarray(snapshots[i].position[:2], dtype=np.float64)
            for i in drone_ids
        }
        velocities = {
            i: (
                np.asarray(snapshots[i].velocity[:2], dtype=np.float64)
                if snapshots[i].velocity is not None
                else np.zeros(2)
            )
            for i in drone_ids
        }
        fixed_accelerations = _fixed_accelerations_from_actions(
            fixed_actions=fixed_actions,
            velocities=velocities,
            command_scale=command_scale,
        )

        a_nom: dict[int, np.ndarray] = {}
        for i in drone_ids:
            v_nom = np.asarray(nominal_actions[i][:2], dtype=np.float64)
            a_nom[i] = (
                fixed_accelerations[i].copy()
                if i in fixed_accelerations
                else np.clip(
                    self.kv * (v_nom - velocities[i]),
                    -self.a_max,
                    self.a_max,
                )
            )

        s_now: dict[tuple[int, int], float] = {}
        s_next: dict[tuple[int, int], float] = {}
        s_static: dict[tuple[int, int], float] = {}
        pair_rho: dict[tuple[int, int], float] = {}
        pair_deg: dict[tuple[int, int], float] = {}
        pair_pred: dict[tuple[int, int], object] = {}

        for a in range(len(drone_ids)):
            for b in range(a + 1, len(drone_ids)):
                i = drone_ids[a]
                j = drone_ids[b]
                snap_i = snapshots[i]
                snap_j = snapshots[j]
                pair_aoi = aoi.get((i, j), aoi.get((j, i), 0.0))
                sigma_i, sigma_j = self._pair_sigmas(snap_i, snap_j)
                v_cl_now = _closing_speed_2d(
                    positions[i], positions[j], velocities[i], velocities[j]
                )
                s_now[(i, j)] = _d_safe_2d(
                    v_cl_now, pair_aoi, sigma_i, sigma_j, self.params
                )
                # PB-CBF supplies its own bounded-braking margin.  Keep only
                # the physical, perception, and communication terms here to
                # avoid counting the dynamics margin twice.
                s_static[(i, j)] = (
                    self.params.d0
                    + perception_margin(sigma_i, sigma_j, self.params)
                    + communication_margin(pair_aoi, self.params)
                )

                # The execution interval changes both the position coefficient
                # beta and the next-sample closing speed.  Use the worst next
                # safety boundary over the four tau rectangle vertices; the
                # QP then remains affine in beta while retaining a robust
                # scalar D_next for every vertex constraint.
                next_boundaries = []
                tau_vertices = (
                    (self.tau_px4_min, self.tau_px4_min),
                    (self.tau_px4_min, self.tau_px4_max),
                    (self.tau_px4_max, self.tau_px4_min),
                    (self.tau_px4_max, self.tau_px4_max),
                )
                for tau_i, tau_j in tau_vertices:
                    alpha_i = (
                        1.0 - float(np.exp(-dt / tau_i))
                        if tau_i > 0 else 1.0
                    )
                    alpha_j = (
                        1.0 - float(np.exp(-dt / tau_j))
                        if tau_j > 0 else 1.0
                    )
                    v_pred_i = velocities[i] + alpha_i * a_nom[i] * command_scale
                    v_pred_j = velocities[j] + alpha_j * a_nom[j] * command_scale
                    v_cl_next = _closing_speed_2d(
                        positions[i], positions[j], v_pred_i, v_pred_j
                    )
                    next_boundaries.append(
                        _d_safe_2d(
                            v_cl_next, pair_aoi, sigma_i, sigma_j, self.params
                        )
                    )
                s_next[(i, j)] = max(next_boundaries)

                distance = float(np.linalg.norm(positions[i] - positions[j]))
                rho = normalized_margin(distance, s_now[(i, j)])
                tracker = self._tracker(i, j)
                degradation = tracker.update(rho, t)
                pred = self.monitor.predict(
                    agent_i=i,
                    agent_j=j,
                    p_i=snap_i.position,
                    p_j=snap_j.position,
                    v_i=snap_i.velocity or (0.0, 0.0, 0.0),
                    v_j=snap_j.velocity or (0.0, 0.0, 0.0),
                    sigma_i=sigma_i,
                    sigma_j=sigma_j,
                    aoi=pair_aoi,
                    params=self.params,
                    horizon=self.params.prediction_horizon,
                )
                pair_rho[(i, j)] = rho
                pair_deg[(i, j)] = degradation
                pair_pred[(i, j)] = pred

        pcbf_result = None
        if self.sampled_data_method == "pcbf":
            # PCBF's state constraint is held fixed within one finite-horizon
            # solve.  For the full-boundary condition this is the current
            # dynamic boundary; it is deliberately not AegisAir's forecast or
            # reserve monitor, which the external baseline must not access.
            pcbf_result = solve_pcbf(
                nominal_accelerations=a_nom,
                positions=positions,
                velocities=velocities,
                safe_distances={
                    pair: distance + self.sampled_data_boundary_buffer_m
                    for pair, distance in (
                        s_now
                        if self.constraint_boundary == "full"
                        else s_static
                    ).items()
                },
                config=PCBFConfig(
                    horizon=self.pcbf_config.horizon,
                    dt=dt,
                    # The command interface publishes ``v + command_scale*a``,
                    # but PX4 only realizes an alpha fraction during this 20 Hz
                    # sample under its first-order velocity tracking dynamics.
                    # PCBF must predict the realized state increment, not the
                    # larger setpoint jump.
                    control_scale_s=(
                        (
                            1.0 - float(np.exp(-dt / self.tau_px4))
                            if self.tau_px4 > 0.0
                            else 1.0
                        ) * command_scale
                    ),
                    a_max=self.a_max,
                    terminal_buffer_m=self.pcbf_config.terminal_buffer_m,
                    terminal_velocity_tolerance_mps=self.pcbf_config.terminal_velocity_tolerance_mps,
                    position_bound_m=self.pcbf_config.position_bound_m,
                    velocity_bound_mps=self.pcbf_config.velocity_bound_mps,
                    max_iterations=self.pcbf_config.max_iterations,
                    multistart_count=self.pcbf_config.multistart_count,
                    tolerance=self.pcbf_config.tolerance,
                    acceptable_tolerance=self.pcbf_config.acceptable_tolerance,
                    lexicographic_tolerance=self.pcbf_config.lexicographic_tolerance,
                ),
                fixed_accelerations=fixed_accelerations,
            )
            a_safe = pcbf_result.accelerations
            feasible = pcbf_result.feasible
        elif self.sampled_data_method == "zocbf":
            point_beta = beta_of_tau(dt, self.tau_px4)
            a_safe, feasible, _ = solve_zocbf_qp(
                a_nom=a_nom,
                positions=positions,
                velocities=velocities,
                s_now=(s_now if self.constraint_boundary == "full" else {p: d + self.sampled_data_boundary_buffer_m for p, d in s_static.items()}),
                s_next=(s_next if self.constraint_boundary == "full" else {p: d + self.sampled_data_boundary_buffer_m for p, d in s_static.items()}),
                dt=dt,
                gamma=self.gamma,
                delta=self.zocbf_delta,
                a_max=self.a_max,
                beta={i: point_beta for i in drone_ids},
                command_scale=command_scale,
                max_iters=self.qp_max_iters,
                infeasible_fallback=self.sampled_data_infeasible_fallback,
                fixed_accelerations=fixed_accelerations,
            )
        elif self.sampled_data_method == "pb_cbf":
            a_safe, feasible, _ = solve_prediction_based_cbf_qp(
                a_nom=a_nom,
                positions=positions,
                velocities=velocities,
                static_distance={p: d + self.sampled_data_boundary_buffer_m for p, d in s_static.items()},
                alpha=self.pb_alpha,
                braking_accel=self.pb_braking_accel,
                a_max=self.a_max,
                max_iters=self.qp_max_iters,
                infeasible_fallback=self.sampled_data_infeasible_fallback,
                fixed_accelerations=fixed_accelerations,
            )
        elif self.execution_model == "legacy_trapezoidal":
            alpha = (
                1.0 - float(np.exp(-dt / self.tau_px4))
                if self.tau_px4 > 0.0
                else 1.0
            )
            a_safe, feasible, _ = solve_sampled_data_qp(
                a_nom=a_nom,
                positions=positions,
                velocities=velocities,
                s_now=s_now,
                s_next=s_next,
                dt=dt,
                gamma=self.gamma,
                a_max=self.a_max,
                alpha=alpha,
                max_iters=self.qp_max_iters,
                fixed_accelerations=fixed_accelerations,
            )
        else:
            a_safe, feasible, _ = solve_robust_sampled_data_qp(
                a_nom=a_nom,
                positions=positions,
                velocities=velocities,
                s_now=s_now,
                s_next=s_next,
                dt=dt,
                gamma=self.gamma,
                a_max=self.a_max,
                beta=beta,
                command_scale=command_scale,
                max_iters=self.qp_max_iters,
                fixed_accelerations=fixed_accelerations,
            )
        self.qp_solve_count += 1
        if not feasible:
            self.qp_infeasible_count += 1
        self.last_qp_feasible = feasible

        results: dict[int, FilterResult] = {}
        for i in drone_ids:
            worst_rho = float("inf")
            worst_g = 0.0
            worst_pair: int | None = None
            worst_pred_rho = float("inf")
            worst_tau = 0.0
            worst_ttsb: float | None = None
            worst_conf = 1.0
            for (ii, jj), rho in pair_rho.items():
                if i != ii and i != jj:
                    continue
                other = jj if ii == i else ii
                if rho < worst_rho:
                    worst_rho = rho
                    worst_g = pair_deg[(ii, jj)]
                    worst_pair = other
                pred = pair_pred[(ii, jj)]
                if pred.rho_min_pred < worst_pred_rho:
                    worst_pred_rho = pred.rho_min_pred
                    worst_tau = pred.tau_star
                    worst_ttsb = pred.ttsb
                    worst_conf = pred.reliability_score

            v_nom = np.asarray(nominal_actions[i][:2], dtype=np.float64)
            v_safe = (
                fixed_actions[i].copy()
                if i in fixed_actions
                else np.clip(
                    velocities[i] + a_safe[i] * command_scale,
                    -self.v_max,
                    self.v_max,
                )
            )
            a_nom_i = np.asarray(a_nom[i], dtype=np.float64)
            a_safe_i = np.asarray(a_safe[i], dtype=np.float64)
            accel_saturated = bool(np.linalg.norm(a_safe_i) >= self.a_max - 1e-6)
            vel_saturated = bool(np.linalg.norm(v_safe) >= self.v_max - 1e-6)
            local_boundaries = [
                s_now[(ii, jj)] for (ii, jj) in s_now if i == ii or i == jj
            ]
            d_safe_i = min(local_boundaries) if local_boundaries else None
            intervened = bool(np.linalg.norm(a_safe[i] - a_nom[i]) > 1e-6)
            mode = (
                "override"
                if intervened
                else ("warning" if worst_rho < self.rho_warn else "normal")
            )
            results[i] = FilterResult(
                drone=i,
                mode=mode,
                nominal_action=(float(v_nom[0]), float(v_nom[1])),
                safe_action=(float(v_safe[0]), float(v_safe[1])),
                safety_margin=float(worst_rho),
                predicted_margin=float(worst_pred_rho),
                time_to_min_margin=float(worst_tau),
                time_to_safety_boundary=worst_ttsb,
                prediction_reliability_score=float(worst_conf),
                recovery_buffer=(
                    worst_ttsb - self.params.estimated_recovery_latency
                    if worst_ttsb is not None
                    else None
                ),
                semantic_recovery_feasible=(
                    worst_ttsb is not None
                    and worst_ttsb > self.params.estimated_recovery_latency
                ),
                degradation=float(worst_g),
                worst_pair=worst_pair,
                intervened=intervened,
                proactive=worst_pred_rho < self.params.rho_pred_threshold,
                a_nom=(float(a_nom_i[0]), float(a_nom_i[1])),
                a_safe=(float(a_safe_i[0]), float(a_safe_i[1])),
                d_safe=float(d_safe_i) if d_safe_i is not None else None,
                accel_saturated=accel_saturated,
                vel_saturated=vel_saturated,
                feasible=bool(feasible),
                selected_filter=("pcbf" if self.sampled_data_method == "pcbf" else None),
                pcbf_status=(pcbf_result.status if pcbf_result is not None else None),
                pcbf_stage1_status=(pcbf_result.stage1_status if pcbf_result is not None else None),
                pcbf_stage2_status=(pcbf_result.stage2_status if pcbf_result is not None else None),
                pcbf_terminal_feasible=(pcbf_result.terminal_feasible if pcbf_result is not None else None),
                pcbf_value=(pcbf_result.value if pcbf_result is not None else None),
                pcbf_slack_sum=(pcbf_result.slack_sum if pcbf_result is not None else None),
                pcbf_tracking_cost=(pcbf_result.tracking_cost if pcbf_result is not None else None),
                pcbf_max_constraint_violation=(
                    pcbf_result.max_constraint_violation
                    if pcbf_result is not None
                    else None
                ),
                pcbf_tie_break_applied=(
                    pcbf_result.tie_break_applied
                    if pcbf_result is not None
                    else None
                ),
                pcbf_fail_closed_reason=(pcbf_result.fail_closed_reason if pcbf_result is not None else None),
                control_authority=i not in fixed_actions,
                fixed_action=(
                    tuple(float(value) for value in fixed_actions[i])
                    if i in fixed_actions
                    else None
                ),
            )
        return results

    def _filter_hocbf(
        self,
        snapshots: dict[int, object],
        nominal_actions: dict[int, np.ndarray],
        t: float,
        aoi: dict[tuple[int, int], float] | None,
        fixed_actions: dict[int, np.ndarray],
    ) -> dict[int, FilterResult]:
        """Acceleration-aware centralized HOCBF filter."""
        aoi = aoi or {}
        dt = self.params.degradation_dt
        if self._last_t is not None:
            dt = max(1e-3, t - self._last_t)
        self._last_t = t
        for agent_id, snapshot in snapshots.items():
            if snapshot.velocity is not None:
                self.monitor.update_acceleration(agent_id, snapshot.velocity, dt)

        drone_ids = sorted(snapshots)
        command_scale = (
            dt
            if self.command_feedforward_tau_s is None
            else self.command_feedforward_tau_s
        )
        positions = {
            i: np.asarray(snapshots[i].position[:2], dtype=np.float64)
            for i in drone_ids
        }
        velocities = {
            i: (
                np.asarray(snapshots[i].velocity[:2], dtype=np.float64)
                if snapshots[i].velocity is not None
                else np.zeros(2)
            )
            for i in drone_ids
        }
        fixed_accelerations = _fixed_accelerations_from_actions(
            fixed_actions=fixed_actions,
            velocities=velocities,
            command_scale=command_scale,
        )

        d_safe_map: dict[tuple[int, int], float] = {}
        static_safe_map: dict[tuple[int, int], float] = {}
        pair_metrics: dict[tuple[int, int], tuple[float, float, object]] = {}
        pair_context: dict[tuple[int, int], tuple[float, float, float]] = {}
        for a in range(len(drone_ids)):
            for b in range(a + 1, len(drone_ids)):
                i = drone_ids[a]
                j = drone_ids[b]
                snap_i = snapshots[i]
                snap_j = snapshots[j]
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                v_cl = closing_speed(
                    snap_i.position,
                    snap_j.position,
                    snap_i.velocity or (0.0, 0.0, 0.0),
                    snap_j.velocity or (0.0, 0.0, 0.0),
                )
                pair_aoi = aoi.get((i, j), aoi.get((j, i), 0.0))
                sigma_i, sigma_j = self._pair_sigmas(snap_i, snap_j)
                d_safe_pair = dynamic_safety_boundary(
                    closing_speed=v_cl,
                    perception_sigma_i=sigma_i,
                    perception_sigma_j=sigma_j,
                    aoi=pair_aoi,
                    params=self.params,
                )
                rho = normalized_margin(distance, d_safe_pair)
                tracker = self._tracker(i, j)
                degradation = tracker.update(rho, t)
                pred = self.monitor.predict(
                    agent_i=i,
                    agent_j=j,
                    p_i=snap_i.position,
                    p_j=snap_j.position,
                    v_i=snap_i.velocity or (0.0, 0.0, 0.0),
                    v_j=snap_j.velocity or (0.0, 0.0, 0.0),
                    sigma_i=sigma_i,
                    sigma_j=sigma_j,
                    aoi=pair_aoi,
                    params=self.params,
                    horizon=self.params.prediction_horizon,
                )
                d_safe_map[(i, j)] = d_safe_pair
                static_safe_map[(i, j)] = (
                    self.params.d0
                    + perception_margin(sigma_i, sigma_j, self.params)
                    + communication_margin(pair_aoi, self.params)
                )
                pair_metrics[(i, j)] = (rho, degradation, pred)
                pair_context[(i, j)] = (pair_aoi, sigma_i, sigma_j)

        a_nom: dict[int, np.ndarray] = {}
        for i in drone_ids:
            v_nom = np.asarray(nominal_actions[i][:2], dtype=np.float64)
            a_nom[i] = (
                fixed_accelerations[i].copy()
                if i in fixed_accelerations
                else np.clip(
                    self.kv * (v_nom - velocities[i]),
                    -self.a_max,
                    self.a_max,
                )
            )

        if self.hocbf_boundary_guard > 0.0 or self.hocbf_boundary_buffer_m > 0.0:
            execution_alpha = (
                1.0 - float(np.exp(-dt / self.tau_px4))
                if self.tau_px4 > 0.0
                else 1.0
            )
            for (i, j), current_boundary in list(d_safe_map.items()):
                v_pred_i = velocities[i] + execution_alpha * a_nom[i] * command_scale
                v_pred_j = velocities[j] + execution_alpha * a_nom[j] * command_scale
                v_cl_next = _closing_speed_2d(
                    positions[i], positions[j], v_pred_i, v_pred_j
                )
                pair_aoi, sigma_i, sigma_j = pair_context[(i, j)]
                next_boundary = _d_safe_2d(
                    v_cl_next, pair_aoi, sigma_i, sigma_j, self.params
                )
                growth = max(0.0, next_boundary - current_boundary)
                d_safe_map[(i, j)] = (
                    current_boundary
                    + self.hocbf_boundary_guard * growth
                    + self.hocbf_boundary_buffer_m
                )

        a_hocbf, primary_feasible, _ = solve_acceleration_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            d_safe=d_safe_map,
            k1=self.k1,
            k2=self.k2,
            a_max=self.a_max,
            infeasible_fallback=self.hocbf_infeasible_fallback,
            fixed_accelerations=fixed_accelerations,
        )
        feasibility_reserve = minimum_acceleration_box_reserve(
            positions=positions,
            velocities=velocities,
            d_safe=d_safe_map,
            k1=self.k1,
            k2=self.k2,
            a_max=self.a_max,
            fixed_accelerations=fixed_accelerations,
        )

        predictive_feasible: bool | None = None
        if (
            self.hocbf_pb_recovery
            and self.hocbf_predictive_recovery
            and primary_feasible
        ):
            execution_alpha = (
                1.0 - float(np.exp(-dt / self.tau_px4))
                if self.tau_px4 > 0.0
                else 1.0
            )
            execution_beta = beta_of_tau(dt, self.tau_px4)
            probe_positions = positions
            probe_velocities = velocities
            probe_acceleration = a_hocbf
            predictive_feasible = True
            for prediction_step in range(self.hocbf_prediction_steps):
                # Only the first prediction interval carries the pipeline
                # uncertainty.  Once the delayed command reaches PX4, later
                # intervals use the identified first-order execution model.
                execution_fraction = (
                    self.hocbf_prediction_execution_fraction
                    if prediction_step == 0
                    else 1.0
                )
                applied_acceleration = {
                    i: execution_fraction * probe_acceleration[i]
                    for i in drone_ids
                }
                predicted_positions = {
                    i: probe_positions[i]
                    + probe_velocities[i] * dt
                    + execution_beta * applied_acceleration[i] * command_scale
                    for i in drone_ids
                }
                predicted_velocities = {
                    i: probe_velocities[i]
                    + execution_alpha * applied_acceleration[i] * command_scale
                    for i in drone_ids
                }
                predicted_fixed_accelerations = _fixed_accelerations_from_actions(
                    fixed_actions=fixed_actions,
                    velocities=predicted_velocities,
                    command_scale=command_scale,
                )
                predicted_a_nom = {
                    i: (
                        predicted_fixed_accelerations[i].copy()
                        if i in predicted_fixed_accelerations
                        else np.clip(
                            self.kv
                            * (
                                np.asarray(
                                    nominal_actions[i][:2], dtype=np.float64
                                )
                                - predicted_velocities[i]
                            ),
                            -self.a_max,
                            self.a_max,
                        )
                    )
                    for i in drone_ids
                }
                predicted_boundaries: dict[tuple[int, int], float] = {}
                for (i, j), (pair_aoi, sigma_i, sigma_j) in pair_context.items():
                    closing_now = _closing_speed_2d(
                        predicted_positions[i],
                        predicted_positions[j],
                        predicted_velocities[i],
                        predicted_velocities[j],
                    )
                    boundary_now = _d_safe_2d(
                        closing_now, pair_aoi, sigma_i, sigma_j, self.params
                    )
                    future_velocity_i = (
                        predicted_velocities[i]
                        + execution_alpha * predicted_a_nom[i] * command_scale
                    )
                    future_velocity_j = (
                        predicted_velocities[j]
                        + execution_alpha * predicted_a_nom[j] * command_scale
                    )
                    closing_next = _closing_speed_2d(
                        predicted_positions[i],
                        predicted_positions[j],
                        future_velocity_i,
                        future_velocity_j,
                    )
                    boundary_next = _d_safe_2d(
                        closing_next, pair_aoi, sigma_i, sigma_j, self.params
                    )
                    predicted_boundaries[(i, j)] = (
                        boundary_now
                        + self.hocbf_boundary_guard
                        * max(0.0, boundary_next - boundary_now)
                        + self.hocbf_boundary_buffer_m
                    )
                probe_acceleration, predictive_feasible, _ = solve_acceleration_qp(
                    a_nom=predicted_a_nom,
                    positions=predicted_positions,
                    velocities=predicted_velocities,
                    d_safe=predicted_boundaries,
                    k1=self.k1,
                    k2=self.k2,
                    a_max=self.a_max,
                    infeasible_fallback=self.hocbf_infeasible_fallback,
                    fixed_accelerations=predicted_fixed_accelerations,
                )
                if not predictive_feasible:
                    break
                probe_positions = predicted_positions
                probe_velocities = predicted_velocities
        elif self.hocbf_pb_recovery and self.hocbf_predictive_recovery:
            predictive_feasible = False

        recovery_trigger: str | None = None
        if self.hocbf_pb_recovery:
            if not primary_feasible:
                recovery_trigger = "primary_infeasible"
            elif (
                self.hocbf_recovery_reserve_threshold is not None
                and feasibility_reserve
                < self.hocbf_recovery_reserve_threshold
            ):
                recovery_trigger = "feasibility_reserve_low"
            elif self.hocbf_predictive_recovery and predictive_feasible is False:
                recovery_trigger = "predictive_infeasible"

            if recovery_trigger is not None:
                if not self._hocbf_recovery_active:
                    self.hocbf_recovery_entries += 1
                self._hocbf_recovery_active = True
                self._hocbf_recovery_clean_steps = 0
                self._hocbf_recovery_reason = recovery_trigger
            elif self._hocbf_recovery_active:
                self._hocbf_recovery_clean_steps += 1
                if self._hocbf_recovery_clean_steps >= self.hocbf_recovery_clear_steps:
                    self._hocbf_recovery_active = False
                    self._hocbf_recovery_clean_steps = 0
                    self._hocbf_recovery_reason = None

        recovery_active = self.hocbf_pb_recovery and self._hocbf_recovery_active
        recovery_feasible: bool | None = None
        if recovery_active:
            a_safe, recovery_feasible, _ = solve_prediction_based_cbf_qp(
                a_nom=a_nom,
                positions=positions,
                velocities=velocities,
                static_distance={
                    pair: boundary + self.hocbf_recovery_boundary_buffer_m
                    for pair, boundary in static_safe_map.items()
                },
                alpha=self.hocbf_recovery_alpha,
                braking_accel=self.hocbf_recovery_braking_accel,
                a_max=self.a_max,
                max_iters=self.qp_max_iters,
                infeasible_fallback="max_brake",
                fixed_accelerations=fixed_accelerations,
            )
            feasible = recovery_feasible
            selected_filter = "pb_recovery"
            self.hocbf_recovery_steps += 1
            if not recovery_feasible:
                self.hocbf_recovery_infeasible_count += 1
        else:
            a_safe = a_hocbf
            feasible = primary_feasible
            selected_filter = "hocbf"

        self.last_primary_hocbf_feasible = primary_feasible
        self.last_predictive_hocbf_feasible = predictive_feasible
        if not primary_feasible:
            self.hocbf_primary_infeasible_count += 1
        if predictive_feasible is False:
            self.hocbf_predictive_infeasible_count += 1
        self.qp_solve_count += 1
        if not feasible:
            self.qp_infeasible_count += 1
        self.last_qp_feasible = feasible

        results: dict[int, FilterResult] = {}
        for i in drone_ids:
            worst_rho = float("inf")
            worst_g = 0.0
            worst_pair: int | None = None
            worst_pred_rho = float("inf")
            worst_tau = 0.0
            worst_ttsb: float | None = None
            worst_conf = 1.0
            for (ii, jj), (rho, degradation, pred) in pair_metrics.items():
                if i != ii and i != jj:
                    continue
                other = jj if ii == i else ii
                if rho < worst_rho:
                    worst_rho = rho
                    worst_g = degradation
                    worst_pair = other
                if pred.rho_min_pred < worst_pred_rho:
                    worst_pred_rho = pred.rho_min_pred
                    worst_tau = pred.tau_star
                    worst_ttsb = pred.ttsb
                    worst_conf = pred.reliability_score

            v_nom = np.asarray(nominal_actions[i][:2], dtype=np.float64)
            v_safe = (
                fixed_actions[i].copy()
                if i in fixed_actions
                else np.clip(
                    velocities[i] + a_safe[i] * command_scale,
                    -self.v_max,
                    self.v_max,
                )
            )
            intervened = bool(np.linalg.norm(a_safe[i] - a_nom[i]) > 1e-6)
            if intervened:
                mode = "override"
            elif worst_rho < self.rho_warn:
                mode = "warning"
            else:
                mode = "normal"
            local_boundaries = [
                boundary
                for pair, boundary in d_safe_map.items()
                if i in pair
            ]

            results[i] = FilterResult(
                drone=i,
                mode=mode,
                nominal_action=(float(v_nom[0]), float(v_nom[1])),
                safe_action=(float(v_safe[0]), float(v_safe[1])),
                safety_margin=float(worst_rho),
                predicted_margin=float(worst_pred_rho),
                time_to_min_margin=float(worst_tau),
                time_to_safety_boundary=worst_ttsb,
                prediction_reliability_score=float(worst_conf),
                recovery_buffer=(
                    worst_ttsb - self.params.estimated_recovery_latency
                    if worst_ttsb is not None
                    else None
                ),
                semantic_recovery_feasible=(
                    worst_ttsb is not None
                    and worst_ttsb > self.params.estimated_recovery_latency
                ),
                degradation=float(worst_g),
                worst_pair=worst_pair,
                intervened=intervened,
                proactive=worst_pred_rho < self.params.rho_pred_threshold,
                a_nom=(float(a_nom[i][0]), float(a_nom[i][1])),
                a_safe=(float(a_safe[i][0]), float(a_safe[i][1])),
                d_safe=(
                    float(min(local_boundaries)) if local_boundaries else None
                ),
                accel_saturated=bool(
                    np.linalg.norm(a_safe[i]) >= self.a_max - 1e-6
                ),
                vel_saturated=bool(
                    np.linalg.norm(v_safe) >= self.v_max - 1e-6
                ),
                feasible=bool(feasible),
                selected_filter=selected_filter,
                primary_feasible=bool(primary_feasible),
                predictive_feasible=predictive_feasible,
                recovery_active=recovery_active,
                recovery_reason=(
                    self._hocbf_recovery_reason if recovery_active else None
                ),
                feasibility_reserve=float(feasibility_reserve),
                control_authority=i not in fixed_actions,
                fixed_action=(
                    tuple(float(value) for value in fixed_actions[i])
                    if i in fixed_actions
                    else None
                ),
            )
        return results


def _normalize_fixed_actions(
    snapshots: dict[int, object],
    fixed_actions: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    normalized = {
        drone: np.asarray(action, dtype=np.float64)
        for drone, action in (fixed_actions or {}).items()
    }
    unknown = set(normalized) - set(snapshots)
    if unknown:
        raise ValueError(f"fixed action contains unknown drones: {sorted(unknown)}")
    if any(
        action.shape != (2,) or not np.all(np.isfinite(action))
        for action in normalized.values()
    ):
        raise ValueError("fixed actions must be finite planar vectors")
    return normalized


def _fixed_accelerations_from_actions(
    *,
    fixed_actions: dict[int, np.ndarray],
    velocities: dict[int, np.ndarray],
    command_scale: float,
) -> dict[int, np.ndarray]:
    if command_scale <= 0.0:
        raise ValueError("command scale must be positive")
    return {
        drone: (action - velocities[drone]) / command_scale
        for drone, action in fixed_actions.items()
    }


def _projected_sigma(snapshot, direction: np.ndarray, default: float) -> float:
    """Return the covariance-derived sigma along ``direction``, or the default."""
    cov = getattr(snapshot, "covariance", None)
    if cov is None:
        return default
    P = np.asarray(cov, dtype=np.float64).reshape(2, 2)
    variance = float(direction @ P @ direction)
    if variance <= 0.0:
        return 0.0
    return float(np.sqrt(variance))


def _closing_speed_2d(
    p_i: np.ndarray,
    p_j: np.ndarray,
    v_i: np.ndarray,
    v_j: np.ndarray,
) -> float:
    delta = np.asarray(p_i, dtype=np.float64) - np.asarray(p_j, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        return 0.0
    rel_v = np.asarray(v_i, dtype=np.float64) - np.asarray(v_j, dtype=np.float64)
    return max(0.0, -float(np.dot(delta, rel_v)) / distance)


def _d_safe_2d(
    closing_speed: float,
    aoi: float,
    sigma_i: float,
    sigma_j: float,
    params: RuntimeAssuranceParams,
) -> float:
    return (
        params.d0
        + dynamics_margin(closing_speed, params)
        + perception_margin(sigma_i, sigma_j, params)
        + communication_margin(aoi, params)
    )
