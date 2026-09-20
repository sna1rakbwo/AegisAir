"""Phase 5 closed-loop runner.

The first gate is deterministic and does not require PX4/Gazebo:

    MultiUAVEnv telemetry -> DroneSnapshot -> nominal go-to-goal pilot ->
    RuntimeAssurance CBF -> AsyncMissionReplanner -> FLU move_to command ->
    px4_adapter codec -> LocalSafetyState

The same command/telemetry codecs are reused by the live MQTT path, which is
intended to be run against an already-launched PX4 SITL + ROS 2 adapter stack.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import ScenarioConfig
from marllib.envs.multi_uav import MultiUAVEnv
if TYPE_CHECKING:
    from marllib.policies.mappo import MappoPilot
from px4_adapter.mqtt_codec import (
    TelemetryState,
    decode_command,
    normalize_command_to_ned,
)
from px4_adapter.px4_codec import flu_to_px4_ned
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits
from swarm.estimation import (
    EstimatedState,
    SharedStateEstimator,
    SharedStateEstimatorConfig,
)
from swarm.ra.margin import normalized_margin
from swarm.ra.margins import RuntimeAssuranceParams, dynamic_safety_boundary
from swarm.ra.execution_supervisor import (
    ExecutionConformanceSupervisor,
    ExecutionSupervisorConfig,
    TelemetryFreshnessConfig,
    TelemetryFreshnessGate,
    deterministic_backup_velocity,
)
from swarm.ra.c1_px4_supervisor import (
    C1Px4AdmissionSupervisor,
    C1Px4SupervisorConfig,
)
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.interfaces import ExecutionAssuranceDecision
from swarm.recovery import (
    C3AdmissionCoordinator,
    C3GroupSlotCoordinator,
    C3ReservationCoordinator,
    C3SpaceTimeReservationCoordinator,
    AsyncMissionReplanner,
    DeterministicRecoveryClient,
    LLMRecoveryClient,
    LLMRecoveryResult,
    MlxLmClient,
    RecoveryContext,
    RecoveryOverrides,
    RecoverabilityAdmissionCoordinator,
    ReplanConfig,
    RuleMissionPlanner,
)
from swarm.safety import DroneSnapshot, safe_holding_point


class _FaultedLlmClient(LLMRecoveryClient):
    """Deterministic invalid-LLM stand-in for live fault injection."""

    name = "faulted"

    def __init__(self, raw) -> None:
        self.raw = raw

    def generate(self, context) -> LLMRecoveryResult:
        return LLMRecoveryResult(
            plan=None,
            raw=self.raw,
            latency_s=0.0,
            timeout=False,
            valid=False,
            errors=[],
            backend=self.name,
        )


CRUISE_ALTITUDE_M = 2.5
COMMAND_HORIZON_S = 0.5
COMMAND_TTL_S = 0.15
NATIVE_COMMAND_TTL_S = 0.5
NOMINAL_GAIN = 1.5
ALTITUDE_HOLD_KP = 1.0
MAX_VERTICAL_SPEED_MPS = 1.0
RESET_POSITION_TOLERANCE_M = 0.20


def _latency_summary_ms(
    values: list[float], *, deadline_ms: float
) -> dict[str, float | int] | None:
    """汇总逐步 RA wall-clock 延迟，不删除慢样本。"""
    if not values:
        return None
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
        "deadline_ms": deadline_ms,
        "deadline_misses": sum(value > deadline_ms for value in values),
    }


def _command_authority(*, failed_or_aborted: bool) -> tuple[str, bool]:
    """标记命令权限路径；当前 live runner 不允许高层绕过 RA。"""
    return ("failed_zero" if failed_or_aborted else "ra_filtered", False)
RESET_SPEED_TOLERANCE_MPS = 0.15
RESET_MAX_SPEED_MPS = 1.0
RESET_MAX_VERTICAL_SPEED_MPS = 0.5
RESET_STABLE_S = 1.0
RESET_TIMEOUT_S = 45.0

OBSERVATION_LEGACY_ASYMMETRIC = "legacy_current_nominal_stale_ra"
OBSERVATION_SHARED_CURRENT = "shared_current"
OBSERVATION_SHARED_STALE = "shared_stale"
OBSERVATION_LOCAL_FRESH_SELF = "local_fresh_self_stale_peers"
OBSERVATION_MODES = frozenset(
    {
        OBSERVATION_LEGACY_ASYMMETRIC,
        OBSERVATION_SHARED_CURRENT,
        OBSERVATION_SHARED_STALE,
        OBSERVATION_LOCAL_FRESH_SELF,
    }
)


# ---------------------------------------------------------------------------
# Frozen adapter limits (must match px4_adapter/config.yaml).
# ---------------------------------------------------------------------------

DEFAULT_SAFETY_LIMITS = SafetyLimits(
    command_ttl_sec=1.0,
    telemetry_stale_sec=1.5,
    max_horizontal_speed_mps=3.0,
    max_vertical_speed_mps=2.0,
    min_altitude_m=0.0,
    max_altitude_m=10.0,
    max_position_distance_m=30.0,
    fail_closed_action="LAND",
)


# ---------------------------------------------------------------------------
# Pure command/telemetry codecs.
# ---------------------------------------------------------------------------


def snapshot_from_telemetry(payload: dict[str, Any]) -> DroneSnapshot:
    """Build a FLU DroneSnapshot from an adapter ``swarm/drone/{id}/telemetry``."""
    return DroneSnapshot.from_telemetry(payload)


def flu_snapshot_to_telemetry_state(
    drone: int,
    position_flu: tuple[float, float, float],
    velocity_flu: tuple[float, float, float],
    *,
    timestamp_ms: int,
) -> TelemetryState:
    """Synthesize an adapter TelemetryState (PX4 NED) for codec validation."""
    position_ned = flu_to_px4_ned(*position_flu)
    velocity_ned = flu_to_px4_ned(*velocity_flu)
    return TelemetryState(
        instance_id=drone,
        source_frame="PX4_NED",
        timestamp_ms=timestamp_ms,
        source_timestamp_us=timestamp_ms * 1000,
        armed=True,
        nav_state=14,  # PX4_NAV_STATE_OFFBOARD
        failsafe=False,
        connection_lost=False,
        position=position_ned,
        velocity=velocity_ned,
        xy_valid=True,
        z_valid=True,
        v_xy_valid=True,
        v_z_valid=True,
    )


def build_phase5_command(
    *,
    drone: int,
    safe_velocity: np.ndarray | tuple[float, float],
    position_flu: tuple[float, float, float],
    timestamp_ms: int,
    priority: str = "normal",
    cruise_altitude_m: float = CRUISE_ALTITUDE_M,
    horizon_s: float = COMMAND_HORIZON_S,
    ttl_sec: float = COMMAND_TTL_S,
) -> dict[str, Any]:
    """Encode a RA safe velocity as an adapter FLU ``move_to`` command.

    The RA emits a 2D velocity, but the frozen adapter interface uses position
    setpoints.  We integrate the safe velocity over a short horizon and keep the
    cruise altitude constant.  The adapter's local safety state remains the
    final fail-closed boundary.
    """
    vx = float(safe_velocity[0])
    vy = float(safe_velocity[1])
    target = (
        float(position_flu[0]) + vx * horizon_s,
        float(position_flu[1]) + vy * horizon_s,
        float(cruise_altitude_m),
    )
    return {
        "drone": drone,
        "action": "move_to",
        "target": list(target),
        "waypoint": list(target),
        "source_frame": "FLU",
        "ttl_sec": ttl_sec,
        "priority": priority,
        "command_id": f"phase5-move-{drone}-{timestamp_ms}",
        "timestamp_ms": timestamp_ms,
    }


def build_phase5_velocity_command(
    *,
    drone: int,
    safe_velocity: np.ndarray | tuple[float, float],
    vertical_velocity: float,
    timestamp_ms: int,
    priority: str = "normal",
    ttl_sec: float = COMMAND_TTL_S,
) -> dict[str, Any]:
    """Encode a RA safe velocity as an adapter FLU ``velocity`` command.

    PX4 Offboard interprets a ``TrajectorySetpoint`` with valid ``velocity``
    and ``NaN`` position as a velocity setpoint.  The altitude loop supplies
    the vertical velocity, keeping the 2D CBF semantics intact.
    """
    return {
        "drone": drone,
        "action": "velocity",
        "velocity": [
            float(safe_velocity[0]),
            float(safe_velocity[1]),
            float(vertical_velocity),
        ],
        "source_frame": "FLU",
        "ttl_sec": ttl_sec,
        "priority": priority,
        "command_id": f"phase5-vel-{drone}-{timestamp_ms}",
        "timestamp_ms": timestamp_ms,
    }


def build_phase5_hold_command(
    *,
    drone: int,
    timestamp_ms: int,
    priority: str = "safety",
    ttl_sec: float = COMMAND_TTL_S,
) -> dict[str, Any]:
    """Request adapter-local position hold without using central stale state."""
    return {
        "drone": drone,
        "action": "hold",
        "source_frame": "FLU",
        "ttl_sec": ttl_sec,
        "priority": priority,
        "command_id": f"phase5-hold-{drone}-{timestamp_ms}",
        "timestamp_ms": timestamp_ms,
    }


def _velocity_for_command_mode(result: object, velocity_command_mode: str) -> np.ndarray:
    """Map an RA result to the requested horizontal command exactly once."""
    if velocity_command_mode == "nominal":
        value = getattr(result, "nominal_action")
    elif velocity_command_mode in {"safe_action", "feedforward_tau"}:
        # ``safe_action`` was already formed from the same state and command
        # scale used by the acceleration QP.  Rebuilding it from raw telemetry
        # here can silently change the solver-selected joint input.
        value = getattr(result, "safe_action")
    else:
        raise ValueError(f"unknown velocity_command_mode: {velocity_command_mode}")
    return np.asarray(value, dtype=np.float64)


def _audit_published_command(
    *,
    solver_selected_velocity: np.ndarray | tuple[float, float],
    published_command: dict[str, Any],
    reference_velocity: np.ndarray | tuple[float, float],
    command_scale_s: float,
    fallback_reason: str | None,
) -> dict[str, Any]:
    """Separate solver feasibility from the command actually sent to PX4."""
    selected = np.asarray(solver_selected_velocity, dtype=np.float64)
    published_raw = published_command.get("velocity")
    published = (
        np.asarray(published_raw[:2], dtype=np.float64)
        if published_command.get("action") == "velocity" and published_raw is not None
        else None
    )
    matches = bool(
        published is not None
        and np.allclose(published, selected, atol=1e-9, rtol=0.0)
    )
    effective_acceleration = None
    if published is not None and command_scale_s > 0.0:
        reference = np.asarray(reference_velocity, dtype=np.float64)
        effective_acceleration = ((published - reference) / command_scale_s).tolist()
    return {
        "solver_selected_velocity": selected.tolist(),
        "published_action": published_command["action"],
        "published_velocity": published.tolist() if published is not None else None,
        "published_command_matches_selected": matches,
        "published_effective_acceleration": effective_acceleration,
        "fallback_reason": fallback_reason,
        "command_timestamp_ms": int(published_command["timestamp_ms"]),
        "command_ttl_s": float(published_command["ttl_sec"]),
    }


def _joint_published_constraint_status(
    *,
    command_audits: dict[int, dict[str, Any]],
    solver_feasible: dict[int, bool | None],
    solver_velocity_saturated: dict[int, bool],
) -> bool | None:
    """Transfer a joint solver claim only when the full command vector matches."""
    if not all(
        audit["published_command_matches_selected"]
        for audit in command_audits.values()
    ):
        return None
    feasibility = list(solver_feasible.values())
    if any(value is False for value in feasibility):
        return False
    if all(value is True for value in feasibility) and not any(
        solver_velocity_saturated.values()
    ):
        return True
    return None


def _joint_solver_feasible(solver_feasible: dict[int, bool | None]) -> bool:
    return bool(solver_feasible) and all(
        value is True for value in solver_feasible.values()
    )


def validate_command_path(
    command: dict[str, Any],
    telemetry_state: TelemetryState,
    *,
    now_ms: int,
    limits: SafetyLimits | None = None,
) -> tuple[bool, str]:
    """Run one command through the real adapter decode + local safety path."""
    try:
        decoded = normalize_command_to_ned(
            decode_command(command, default_source_frame="PX4_NED")
        )
    except ValueError as exc:
        return False, f"decode_error:{exc}"
    state = LocalSafetyState(limits or DEFAULT_SAFETY_LIMITS)
    decision = state.evaluate(now_ms, telemetry_state, decoded)
    return decision.allowed, decision.reason


# ---------------------------------------------------------------------------
# Scenarios (frozen for the first gate).
# ---------------------------------------------------------------------------


def _scenario(name: str) -> dict[str, Any]:
    if name == "head_on":
        return {
            "name": "head_on",
            "scenario": ScenarioConfig(
                name="head_on",
                num_agents=2,
                starts=((-4.0, 0.5), (4.0, -0.5)),
                goals=((4.0, -0.5), (-4.0, 0.5)),
                speed_limit=1.5,
            ),
            "mission_change": None,
            "change_step": None,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }

    if name == "crossing":
        return {
            "name": "crossing",
            "scenario": ScenarioConfig(
                name="crossing",
                num_agents=2,
                starts=((-4.0, 0.5), (0.5, -4.0)),
                goals=((4.0, -0.5), (-0.5, 4.0)),
                speed_limit=1.5,
            ),
            "mission_change": None,
            "change_step": None,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }

    if name == "high_closing":
        return {
            "name": "high_closing",
            "scenario": ScenarioConfig(
                name="high_closing",
                num_agents=2,
                starts=((-6.0, 0.0), (6.0, 0.0)),
                goals=((6.0, 0.0), (-6.0, 0.0)),
                speed_limit=2.0,
                accel_limit=3.0,
            ),
            "mission_change": None,
            "change_step": None,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }

    if name == "priority_conflict":
        return {
            "name": "priority_conflict",
            "scenario": ScenarioConfig(
                name="priority_conflict",
                num_agents=2,
                starts=((-6.0, 0.5), (6.0, -0.5)),
                goals=((6.0, -0.5), (-6.0, 0.5)),
            ),
            "mission_change": {"kind": "priority_change", "high": 0, "low": 1},
            "change_step": 0,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": 0,
        }

    if name == "corridor_blocked":
        zone = (2.0, 2.5, -1.0, 1.0)
        return {
            "name": "corridor_blocked",
            "scenario": ScenarioConfig(
                name="corridor_blocked",
                num_agents=1,
                starts=((-6.0, 0.0),),
                goals=((6.0, 0.0),),
            ),
            "mission_change": {
                "kind": "block_corridor",
                "drone": 0,
                "zone": list(zone),
            },
            "change_step": 0,
            "failed_drone": None,
            "blocked_zone": zone,
            "critical_goal": None,
            "high_drone": None,
        }

    if name == "drone_failure":
        return {
            "name": "drone_failure",
            "scenario": ScenarioConfig(
                name="drone_failure",
                num_agents=2,
                starts=((-4.0, 0.0), (-4.0, -2.0)),
                goals=((4.0, 0.0), (4.0, -2.0)),
            ),
            "mission_change": {"kind": "fail_drone", "drone": 0},
            "change_step": 15,
            "failed_drone": 0,
            "blocked_zone": None,
            "critical_goal": (4.0, 0.0),
            "high_drone": None,
        }

    if name == "reassign_3":
        # Greedy "nearest healthy drone takes over the orphan" is suboptimal here:
        # drone 1 is nearest to G0 but its own goal (2, 0) is almost free to keep,
        # while drone 2's own goal (5, -4) is expensive.  Reassigning drone 2 to G0
        # forfeits the expensive goal and cuts total path roughly 10.4 -> 6.1 m.
        return {
            "name": "reassign_3",
            "scenario": ScenarioConfig(
                name="reassign_3",
                num_agents=3,
                starts=((-5.0, 1.0), (0.0, 0.0), (0.0, 4.0)),
                goals=((1.0, 0.0), (2.0, 0.0), (5.0, -4.0)),
            ),
            "mission_change": {"kind": "fail_drone", "drone": 0},
            "change_step": 0,
            "failed_drone": 0,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }

    if name == "priority_reassign":
        # Qualitative semantic constraint: drone 1 is the nearest healthy drone to
        # the orphan, but it is "critical" and must keep its own mission.  The rule
        # planner's nearest-healthy heuristic ignores priority and would divert
        # drone 1; the correct recovery reassigns the normal-priority drone 2.
        return {
            "name": "priority_reassign",
            "scenario": ScenarioConfig(
                name="priority_reassign",
                num_agents=3,
                starts=((-5.0, 1.0), (0.0, 0.0), (0.0, 4.0)),
                goals=((1.0, 0.0), (2.0, 0.0), (5.0, -4.0)),
            ),
            "mission_change": {"kind": "fail_drone", "drone": 0},
            "change_step": 0,
            "failed_drone": 0,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
            "priorities": {1: "critical", 2: "normal"},
            "critical_drone": 1,
        }

    if name == "multi_uav":
        return {
            "name": "multi_uav",
            "scenario": ScenarioConfig(
                name="multi_uav",
                num_agents=4,
                starts=((-3.0, 0.8), (3.0, -0.8), (-3.0, -0.8), (3.0, 0.8)),
                goals=((3.0, -0.8), (-3.0, 0.8), (3.0, 0.8), (-3.0, -0.8)),
                speed_limit=1.5,
            ),
            "mission_change": None,
            "change_step": None,
            "failed_drone": None,
            "blocked_zone": None,
            "critical_goal": None,
            "high_drone": None,
        }

    raise SystemExit(f"unknown scenario: {name}")


# ---------------------------------------------------------------------------
# Shared step helpers.
# ---------------------------------------------------------------------------


def _snapshots(env: MultiUAVEnv) -> dict[int, DroneSnapshot]:
    return {
        i: DroneSnapshot(
            drone_id=i,
            position=(float(env.positions[i, 0]), float(env.positions[i, 1]), 0.0),
            velocity=(float(env.velocities[i, 0]), float(env.velocities[i, 1]), 0.0),
        )
        for i in env.agent_ids
    }


def _noisy_snapshots(
    snapshots: dict[int, DroneSnapshot],
    fault: dict[str, Any],
    rng: np.random.Generator,
) -> dict[int, DroneSnapshot]:
    """Perturb the perceived shared state (position/velocity) with Gaussian noise."""
    pos_sigma = float(fault.get("perception_noise_pos_m") or 0.0)
    vel_sigma = float(fault.get("perception_noise_vel_mps") or 0.0)
    noisy: dict[int, DroneSnapshot] = {}
    for i, snap in snapshots.items():
        velocity = snap.velocity or (0.0, 0.0, 0.0)
        position = tuple(
            float(snap.position[k]) + float(rng.normal(0.0, pos_sigma))
            for k in range(3)
        )
        velocity_noisy = tuple(
            float(velocity[k]) + float(rng.normal(0.0, vel_sigma))
            for k in range(3)
        )
        noisy[i] = replace(snap, position=position, velocity=velocity_noisy)
    return noisy


def _estimated_states_and_aoi(
    snapshots: dict[int, DroneSnapshot],
    estimator: SharedStateEstimator,
    now_ms: int,
    rng,
) -> tuple[dict[int, Any], dict[tuple[int, int], float]]:
    """Advance the shared-state estimator and derive pairwise AoI from it."""
    estimated = estimator.step(snapshots, now_ms, rng=rng)
    aoi: dict[tuple[int, int], float] = {}
    for i in snapshots:
        for j in snapshots:
            if i == j:
                continue
            age_i = max(0.0, (now_ms - estimated[i].timestamp_ms) / 1000.0)
            age_j = max(0.0, (now_ms - estimated[j].timestamp_ms) / 1000.0)
            aoi[(i, j)] = max(age_i, age_j)
    return estimated, aoi


def _estimator_config_for_fault(fault: dict[str, Any]) -> SharedStateEstimatorConfig:
    """Build the shared-state estimator config calibrated to the injected fault.

    The estimator's measurement covariance must match the injected perception
    noise so that ``perception_margin`` covers the actual zero-mean sensor noise
    instead of a hard-coded sigma.
    """

    pos_sigma = float(fault.get("perception_noise_pos_m") or 0.0)
    return SharedStateEstimatorConfig(
        delay_ms=int(fault.get("estimator_delay_ms") or 0),
        dropout_rate=float(fault.get("estimator_dropout_rate") or 0.0),
        measurement_cov_m2=pos_sigma * pos_sigma,
    )


def _propagate_states(
    states: dict[int, Any],
    ages: dict[int, float],
) -> dict[int, Any]:
    """Dead-reckon stale shared state to the current control time."""
    propagated: dict[int, Any] = {}
    for i, state in states.items():
        age = float(ages.get(i, 0.0))
        if age <= 0.0:
            propagated[i] = state
            continue
        velocity = state.velocity or (0.0, 0.0, 0.0)
        position = tuple(
            float(state.position[k]) + float(velocity[k]) * age for k in range(3)
        )
        propagated[i] = replace(state, position=position)
    return propagated


def _fresh_local_state(snapshot: DroneSnapshot, now_ms: int):
    """Return a covariance-zero current onboard state for a local RA view."""
    return EstimatedState(
        drone_id=snapshot.drone_id,
        position=snapshot.position,
        velocity=snapshot.velocity or (0.0, 0.0, 0.0),
        covariance=(0.0, 0.0, 0.0, 0.0),
        timestamp_ms=now_ms,
        dropped=False,
    )


def _local_ra_view(
    *,
    observer: int,
    fresh_states: dict[int, DroneSnapshot],
    peer_states: dict[int, Any],
    now_ms: int,
) -> tuple[dict[int, Any], dict[tuple[int, int], float]]:
    """Build one drone's local view: fresh self, propagated stale peers."""
    view = dict(peer_states)
    view[observer] = _fresh_local_state(fresh_states[observer], now_ms)
    aoi = {
        (observer, peer): max(
            0.0,
            (
                now_ms
                - (
                    now_ms
                    if isinstance(peer_states[peer], DroneSnapshot)
                    else int(getattr(peer_states[peer], "timestamp_ms", now_ms))
                )
            ) / 1000.0,
        )
        for peer in peer_states
        if peer != observer
    }
    return view, aoi


def _go_to_goal(
    env: MultiUAVEnv,
    *,
    base_goals: dict[int, tuple[float, float, float]],
    overrides: RecoveryOverrides,
    failed: set[int],
    aborted: set[int],
    pilot: MappoPilot | None = None,
    observed_states: dict[int, Any] | None = None,
) -> dict[int, np.ndarray]:
    """Create nominal actions from the caller's visible state.

    ``observed_states`` is optional to preserve the legacy current-truth pilot.
    It lets the deterministic runner freeze whether nominal and RA share an
    observation interface instead of silently mixing current truth with stale
    peer telemetry.
    """
    positions = (
        {
            i: np.asarray(observed_states[i].position[:2], dtype=np.float64)
            for i in env.agent_ids
        }
        if observed_states is not None
        else {i: env.positions[i] for i in env.agent_ids}
    )
    velocities = (
        {
            i: np.asarray(
                observed_states[i].velocity or (0.0, 0.0), dtype=np.float64
            )[:2]
            for i in env.agent_ids
        }
        if observed_states is not None
        else {i: env.velocities[i] for i in env.agent_ids}
    )
    if pilot is not None:
        goals = {
            i: np.asarray(overrides.goal_override.get(i, base_goals[i])[:2])
            for i in env.agent_ids
        }
        actions = pilot.actions(
            positions=positions, velocities=velocities, goals=goals
        )
    else:
        actions = {}
        for i in env.agent_ids:
            goal = overrides.goal_override.get(i, base_goals[i])
            delta = np.asarray(goal[:2]) - positions[i]
            actions[i] = np.clip(
                NOMINAL_GAIN * delta,
                -env.scenario.speed_limit,
                env.scenario.speed_limit,
            )

    for i in env.agent_ids:
        if i in failed or i in aborted:
            actions[i] = np.zeros(2)
            continue
        actions[i] = np.asarray(actions[i], dtype=np.float64) * overrides.velocity_scale.get(i, 1.0)
    return actions


def _in_zone(pos: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = zone
    return x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1


def _priority_order(
    drone_ids: list[int],
    urgent_drone: int | None = None,
) -> list[int]:
    """Mission-aware pass order: urgent drone first, then id-sorted rest."""
    if urgent_drone is not None and urgent_drone in drone_ids:
        return [urgent_drone] + sorted(d for d in drone_ids if d != urgent_drone)
    return sorted(drone_ids)


def _freeze_holding_points(
    drone_ids: list[int], positions: dict[int, Any]
) -> dict[int, tuple[float, float, float]]:
    """在一次顺序通行开始时冻结等待点，避免目标随当前位置向外漂移。"""
    return {
        drone: safe_holding_point(drone, positions)
        for drone in drone_ids
    }


def _resolve_priority_order(
    client,
    drone_ids: list[int],
    urgent_drone: int | None,
    snapshots: dict[int, DroneSnapshot],
    base_goals: dict[int, tuple[float, float, float]],
    timestamp_ms: int,
) -> list[int]:
    """Return a right-of-way order from the LLM, with deterministic fallback."""
    fallback = _priority_order(drone_ids, urgent_drone)
    if client is None:
        return fallback
    priorities = {
        d: ("urgent" if d == urgent_drone else "normal") for d in drone_ids
    }
    context = RecoveryContext(
        event="MISSION_PLAN_INVALIDATED",
        agent_i=drone_ids[0],
        agent_j=drone_ids[0],
        current_margin=0.0,
        predicted_min_margin=None,
        margin_degradation=None,
        intervention_count=0,
        cause="COORDINATION_DEGRADATION",
        severity="MEDIUM",
        snapshots=snapshots,
        current_goals=base_goals,
        base_goals=base_goals,
        priorities=priorities,
        timestamp_ms=timestamp_ms,
        mission_change=None,
    )
    try:
        result = client.generate(context)
        raw = result.raw if isinstance(result.raw, dict) else None
    except Exception:
        return fallback
    order = raw.get("priority_order") if raw else None
    if isinstance(order, list):
        try:
            order_int = [int(v) for v in order]
        except (TypeError, ValueError):
            order_int = None
        if order_int is not None and set(order_int) == set(drone_ids):
            print(f"[priority-order] source=llm order={order_int}", flush=True)
            return order_int
    print(
        f"[priority-order] source=fallback order={fallback} raw={raw}",
        flush=True,
    )
    return fallback


def _make_replanner(
    *,
    client,
    fallback,
    dt: float,
    config: ReplanConfig | None = None,
    initial_priorities: dict[int, str] | None = None,
) -> AsyncMissionReplanner:
    return AsyncMissionReplanner(
        client=client,
        fallback=fallback,
        config=config or ReplanConfig(),
        dt=dt,
        initial_priorities=initial_priorities,
    )


def _step_exact_zoh_execution(
    env: MultiUAVEnv,
    commands: dict[int, np.ndarray],
    tau_s: float,
) -> dict[int, dict[str, bool]]:
    """Advance the lightweight plant with exact first-order command tracking.

    This is intentionally opt-in for C2 execution-model ablations.  The
    default simulator retains its original instantaneous velocity-command
    transition, while C2 evaluates all controllers against this same lagged
    plant rather than accidentally evaluating them on ideal execution.
    """
    dt = env.scenario.dt
    alpha = 1.0 - float(np.exp(-dt / tau_s))
    beta = dt - tau_s * alpha
    for idx, agent_id in enumerate(env.agent_ids):
        velocity = env.velocities[idx].copy()
        command = np.clip(
            np.asarray(commands[agent_id], dtype=np.float64),
            -env.scenario.speed_limit,
            env.scenario.speed_limit,
        )
        requested_delta = alpha * (command - velocity)
        realized_delta = np.clip(
            requested_delta,
            -env.scenario.accel_limit * dt,
            env.scenario.accel_limit * dt,
        )
        env.positions[idx] = env.positions[idx] + velocity * dt + beta * (
            realized_delta / alpha
        )
        env.velocities[idx] = velocity + realized_delta
    arena = env.scenario.arena
    env.positions = np.clip(env.positions, (arena[0], arena[2]), (arena[1], arena[3]))
    distances = np.linalg.norm(
        env.positions[:, None, :] - env.positions[None, :, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    collisions = (distances < env.scenario.collision_radius).any(axis=1)
    env.step_count += 1
    return {
        agent_id: {"collided": bool(collisions[idx])}
        for idx, agent_id in enumerate(env.agent_ids)
    }


def _audit_exact_zoh_interval(
    env: MultiUAVEnv,
    commands: dict[int, np.ndarray],
    tau_s: float,
    ra: RuntimeAssurance,
    samples: int,
) -> dict[str, Any]:
    """Reconstruct one exact-ZOH control interval for the C2 dense audit."""
    if samples < 1:
        raise ValueError("samples must be positive")
    dt = env.scenario.dt
    alpha_dt = 1.0 - float(np.exp(-dt / tau_s))
    start_positions = env.positions.copy()
    start_velocities = env.velocities.copy()
    effective_commands: dict[int, np.ndarray] = {}
    for idx, agent_id in enumerate(env.agent_ids):
        command = np.clip(
            np.asarray(commands[agent_id], dtype=np.float64),
            -env.scenario.speed_limit,
            env.scenario.speed_limit,
        )
        requested_delta = alpha_dt * (command - start_velocities[idx])
        realized_delta = np.clip(
            requested_delta,
            -env.scenario.accel_limit * dt,
            env.scenario.accel_limit * dt,
        )
        effective_commands[agent_id] = start_velocities[idx] + realized_delta / alpha_dt

    min_distance = float("inf")
    min_rho = float("inf")
    min_projected = float("inf")
    min_squared = float("inf")
    endpoint_rho = float("inf")
    endpoint_distance = float("inf")
    for t in np.linspace(0.0, dt, samples + 1):
        alpha_t = 1.0 - float(np.exp(-t / tau_s))
        beta_t = t - tau_s * alpha_t
        positions = start_positions.copy()
        velocities = start_velocities.copy()
        for idx, agent_id in enumerate(env.agent_ids):
            delta_u = effective_commands[agent_id] - start_velocities[idx]
            positions[idx] += start_velocities[idx] * t + beta_t * delta_u
            velocities[idx] += alpha_t * delta_u
        for first in range(len(env.agent_ids)):
            for second in range(first + 1, len(env.agent_ids)):
                delta = positions[first] - positions[second]
                distance = float(np.linalg.norm(delta))
                direction = delta / distance if distance > 1e-9 else np.array([1.0, 0.0])
                relative_velocity = velocities[first] - velocities[second]
                closing = (
                    max(0.0, -float(np.dot(delta, relative_velocity)) / distance)
                    if distance > 1e-9
                    else 0.0
                )
                d_safe = dynamic_safety_boundary(
                    closing_speed=closing,
                    perception_sigma_i=ra.perception_sigma,
                    perception_sigma_j=ra.perception_sigma,
                    aoi=0.0,
                    params=ra.params,
                )
                rho = normalized_margin(distance, d_safe)
                projected = float(np.dot(direction, delta)) - d_safe
                squared = distance * distance - d_safe * d_safe
                min_distance = min(min_distance, distance)
                min_rho = min(min_rho, rho)
                min_projected = min(min_projected, projected)
                min_squared = min(min_squared, squared)
                if np.isclose(t, dt):
                    endpoint_rho = min(endpoint_rho, rho)
                    endpoint_distance = min(endpoint_distance, distance)
    return {
        "start_positions": start_positions.tolist(),
        "start_velocities": start_velocities.tolist(),
        "commands": {str(i): np.asarray(commands[i]).tolist() for i in env.agent_ids},
        "effective_commands": {str(i): effective_commands[i].tolist() for i in env.agent_ids},
        "endpoint_min_distance_m": endpoint_distance,
        "endpoint_min_rho": endpoint_rho,
        "intersample_min_distance_m": min_distance,
        "intersample_min_rho": min_rho,
        "intersample_min_projected_barrier": min_projected,
        "intersample_min_squared_barrier": min_squared,
    }


# ---------------------------------------------------------------------------
# Deterministic local closed-loop gate.
# ---------------------------------------------------------------------------


def run_sim_episode(
    *,
    spec: dict[str, Any],
    env: MultiUAVEnv,
    seed: int,
    ra: RuntimeAssurance,
    mode: str,
    llm_client,
    llm_fallback,
    max_steps: int,
    real_time: bool,
    nominal_noise: float = 0.0,
    sequential_pass: bool = False,
    urgent_drone: int | None = None,
    fault: dict[str, Any] | None = None,
    observation_mode: str = OBSERVATION_LEGACY_ASYMMETRIC,
    execution_tau_s: float = 0.0,
    execution_audit_samples: int = 0,
    replan_timeout_s: float | None = None,
    immediate_fallback: bool = False,
    pilot: MappoPilot | None = None,
    c1_px4_supervisor_config: C1Px4SupervisorConfig | None = None,
) -> dict[str, Any]:
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"unknown observation_mode: {observation_mode}")
    if execution_tau_s < 0.0:
        raise ValueError("execution_tau_s must be non-negative")
    if execution_audit_samples and execution_tau_s <= 0.0:
        raise ValueError("exact-ZOH audit requires execution_tau_s > 0")
    if observation_mode == OBSERVATION_LOCAL_FRESH_SELF and pilot is not None:
        raise ValueError("local fresh-self observation is not defined for a centralized pilot")
    env.reset(seed=seed)
    fault = fault or {}
    estimator_config = None
    if ("estimator_delay_ms" in fault or "estimator_dropout_rate" in fault):
        estimator_config = _estimator_config_for_fault(fault)
    estimator = SharedStateEstimator(estimator_config) if estimator_config else None
    estimator_rng = np.random.default_rng(seed + 10_000_019) if estimator else None
    local_ras = (
        {i: copy.deepcopy(ra) for i in env.agent_ids}
        if observation_mode == OBSERVATION_LOCAL_FRESH_SELF
        else None
    )
    replan_kwargs: dict[str, Any] = {"immediate_fallback": immediate_fallback}
    if replan_timeout_s is not None:
        replan_kwargs["replan_timeout_s"] = replan_timeout_s
    replan_config = ReplanConfig(**replan_kwargs)
    replanner = (
        _make_replanner(
            client=llm_client,
            fallback=llm_fallback,
            dt=env.scenario.dt,
            config=replan_config,
            initial_priorities=spec.get("priorities"),
        )
        if mode == "ASYNC"
        else None
    )
    base_goals = {
        i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
        for i in env.agent_ids
    }
    c1_px4_supervisor = (
        C1Px4AdmissionSupervisor(
            config=c1_px4_supervisor_config,
            drone_ids=list(env.agent_ids),
            base_goals=base_goals,
            reset_starts={
                i: (
                    float(env.positions[i, 0]),
                    float(env.positions[i, 1]),
                    0.0,
                )
                for i in env.agent_ids
            },
            rate_hz=1.0 / env.scenario.dt,
            ra_params=ra.params,
            v_max=ra.v_max,
        )
        if c1_px4_supervisor_config is not None
        else None
    )
    overrides = RecoveryOverrides()
    failed: set[int] = set()
    aborted: set[int] = set()
    change_step = spec.get("change_step")
    mission_change = spec.get("mission_change")
    failed_drone = spec.get("failed_drone")
    blocked_zone = spec.get("blocked_zone")
    critical_goal = spec.get("critical_goal")
    high_drone = spec.get("high_drone")
    critical_drone = spec.get("critical_drone")

    collision = False
    completed = False
    completion_step = max_steps
    zone_crossed = False
    critical_reached = False
    priority_violation = False
    high_reached_step = None
    cbf_events = 0
    min_rho = float("inf")
    min_pairwise_distance_m = float("inf")
    control_effort_sum = 0.0
    control_effort_n = 0
    emitted_commands = 0
    rejected_commands = 0
    rejected_reasons: dict[str, int] = {}
    qp_solve_steps = 0
    qp_infeasible_steps = 0
    qp_latency_ms: list[float] = []
    intersample_records: list[dict[str, Any]] = []
    first_recovery_step: int | None = None
    path_length_m = 0.0
    waiting_streak = 0
    max_waiting_streak = 0
    intervention_streak = 0
    max_intervention_streak = 0
    recovery_ra_veto_steps = 0
    seq_state = {
        "active": False,
        "order": _priority_order(env.agent_ids, urgent_drone),
        "idx": 0,
        "best": {i: float("inf") for i in env.agent_ids},
        "stall": {i: 0 for i in env.agent_ids},
        "holding": {},
        "completed": set(),
    }

    for step in range(max_steps):
        t = step * env.scenario.dt
        if change_step is not None and step >= change_step and failed_drone is not None:
            failed.add(failed_drone)

        for i in env.agent_ids:
            if i in failed:
                aborted.add(i)
            elif i in overrides.aborted:
                aborted.add(i)
            elif i in overrides.goal_override:
                env.set_goal(i, np.asarray(overrides.goal_override[i][:2]))
            else:
                env.set_goal(i, np.asarray(base_goals[i][:2]))

        if sequential_pass:
            for i in env.agent_ids:
                dist = float(
                    np.linalg.norm(env.positions[i] - np.asarray(base_goals[i][:2]))
                )
                if dist < env.scenario.goal_epsilon:
                    seq_state["completed"].add(i)
                    seq_state["stall"][i] = 0
                    continue
                if dist < seq_state["best"][i] - 0.02:
                    seq_state["best"][i] = dist
                    seq_state["stall"][i] = 0
                else:
                    seq_state["stall"][i] += 1
            pending = [
                i for i in env.agent_ids if i not in seq_state["completed"]
            ]
            if not seq_state["active"] and len(pending) > 1 and any(
                seq_state["stall"][i] >= 8 for i in pending
            ):
                seq_state["active"] = True
                seq_state["idx"] = 0
                seq_state["best"] = {i: float("inf") for i in env.agent_ids}
                seq_state["stall"] = {i: 0 for i in env.agent_ids}
                seq_state["order"] = _resolve_priority_order(
                    llm_client,
                    pending,
                    urgent_drone,
                    _snapshots(env),
                    base_goals,
                    int(t * 1000),
                )
                seq_state["holding"] = _freeze_holding_points(
                    pending, {i: env.positions[i].copy() for i in env.agent_ids}
                )
            if seq_state["active"]:
                while (
                    seq_state["idx"] < len(seq_state["order"])
                    and seq_state["order"][seq_state["idx"]]
                    in seq_state["completed"]
                ):
                    seq_state["idx"] += 1
                if seq_state["idx"] >= len(seq_state["order"]):
                    seq_state["active"] = False
                    for i in env.agent_ids:
                        overrides.goal_override.pop(i, None)
                        overrides.velocity_scale[i] = 1.0
                    continue
                right_of_way = seq_state["order"][seq_state["idx"]]
                for i in env.agent_ids:
                    if i == right_of_way or i in seq_state["completed"]:
                        overrides.velocity_scale[i] = 1.0
                        overrides.goal_override.pop(i, None)
                        continue
                    holding = seq_state["holding"][i]
                    if (
                        np.linalg.norm(
                            env.positions[i] - np.asarray(holding[:2])
                        )
                        < env.scenario.goal_epsilon
                    ):
                        overrides.goal_override.pop(i, None)
                        overrides.velocity_scale[i] = 0.0
                    else:
                        overrides.goal_override[i] = holding
                        overrides.velocity_scale[i] = 0.6
                if (
                    np.linalg.norm(
                        env.positions[right_of_way]
                        - np.asarray(base_goals[right_of_way][:2])
                    )
                    < env.scenario.goal_epsilon
                ):
                    seq_state["completed"].add(right_of_way)
                    seq_state["idx"] += 1
                    if seq_state["idx"] >= len(seq_state["order"]):
                        seq_state["active"] = False
                        for i in env.agent_ids:
                            overrides.goal_override.pop(i, None)
                            overrides.velocity_scale[i] = 1.0
        fresh_snapshots = _snapshots(env)
        if c1_px4_supervisor is not None:
            c1_goals, c1_scales = c1_px4_supervisor.directives(
                step=step,
                snapshots=fresh_snapshots,
                goal_epsilon=env.scenario.goal_epsilon,
            )
            for i in env.agent_ids:
                if i in c1_goals:
                    overrides.goal_override[i] = c1_goals[i]
                    env.set_goal(i, np.asarray(c1_goals[i][:2]))
                else:
                    overrides.goal_override.pop(i, None)
                    env.set_goal(i, np.asarray(base_goals[i][:2]))
                overrides.velocity_scale[i] = c1_scales[i]
        snapshots = fresh_snapshots
        pos_sigma = float(fault.get("perception_noise_pos_m") or 0.0)
        vel_sigma = float(fault.get("perception_noise_vel_mps") or 0.0)
        if pos_sigma > 0 or vel_sigma > 0:
            rng = np.random.default_rng(seed * 10_000_000 + step)
            snapshots = _noisy_snapshots(snapshots, fault, rng)

        stale_ms = int(fault.get("telemetry_stale_ms") or 0)
        aoi: dict[tuple[int, int], float] = {}
        if stale_ms > 0:
            aoi = {
                (i, j): stale_ms / 1000.0
                for i in env.agent_ids
                for j in env.agent_ids
                if i != j
            }
        if estimator is not None:
            now_ms = int(t * 1000)
            snapshots, estimator_aoi = _estimated_states_and_aoi(
                snapshots,
                estimator,
                now_ms,
                estimator_rng,
            )
            aoi.update(estimator_aoi)
            ages = {
                i: max(0.0, (now_ms - snapshots[i].timestamp_ms) / 1000.0)
                for i in snapshots
            }
            snapshots = _propagate_states(snapshots, ages)
        if observation_mode == OBSERVATION_LEGACY_ASYMMETRIC:
            nominal_states = None
        elif observation_mode == OBSERVATION_LOCAL_FRESH_SELF:
            # The go-to-goal pilot only consumes its own state.  Its own
            # odometry remains fresh; peer state is consumed only by local RA.
            nominal_states = fresh_snapshots
        else:
            nominal_states = snapshots
        nominal = _go_to_goal(
            env,
            base_goals=base_goals,
            overrides=overrides,
            failed=failed,
            aborted=aborted,
            pilot=pilot,
            observed_states=nominal_states,
        )
        if nominal_noise > 0:
            rng = np.random.default_rng(seed * 1_000_000 + step)
            for i in env.agent_ids:
                nominal[i] = nominal[i] + rng.normal(0.0, nominal_noise, 2)
        fixed_actions = {
            i: np.zeros(2, dtype=np.float64)
            for i in env.agent_ids
            if i in failed or i in aborted
        }

        if local_ras is None:
            filter_started = time.perf_counter()
            results = ra.filter(
                snapshots,
                nominal,
                t=t,
                aoi=aoi,
                fixed_actions=fixed_actions,
            )
            filter_elapsed_ms = (time.perf_counter() - filter_started) * 1000.0
            if ra.last_qp_feasible is not None:
                qp_solve_steps += 1
                qp_infeasible_steps += int(not ra.last_qp_feasible)
                qp_latency_ms.append(filter_elapsed_ms)
        else:
            now_ms = int(t * 1000)
            results = {}
            for i in env.agent_ids:
                local_view, local_aoi = _local_ra_view(
                    observer=i,
                    fresh_states=fresh_snapshots,
                    peer_states=snapshots,
                    now_ms=now_ms,
                )
                filter_started = time.perf_counter()
                results[i] = local_ras[i].filter(
                    local_view,
                    nominal,
                    t=t,
                    aoi=local_aoi,
                    fixed_actions=fixed_actions,
                )[i]
                filter_elapsed_ms = (time.perf_counter() - filter_started) * 1000.0
                if local_ras[i].last_qp_feasible is not None:
                    qp_solve_steps += 1
                    qp_infeasible_steps += int(not local_ras[i].last_qp_feasible)
                    qp_latency_ms.append(filter_elapsed_ms)
        if c1_px4_supervisor is not None:
            c1_px4_supervisor.observe_ra(results)
        cbf_events += sum(1 for r in results.values() if r.intervened)
        if first_recovery_step is not None and any(r.intervened for r in results.values()):
            recovery_ra_veto_steps += 1
        if any(r.intervened for r in results.values()):
            intervention_streak += 1
            max_intervention_streak = max(max_intervention_streak, intervention_streak)
        else:
            intervention_streak = 0
        min_rho = min(min_rho, min(r.safety_margin for r in results.values()))
        for i in env.agent_ids:
            control_effort_sum += float(
                np.linalg.norm(np.asarray(results[i].safe_action) - nominal[i])
            )
            control_effort_n += 1

        if blocked_zone is not None:
            for i in env.agent_ids:
                if _in_zone(env.positions[i], blocked_zone):
                    zone_crossed = True

        if replanner is not None:
            current_goals = {
                i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
                for i in env.agent_ids
            }
            overrides = replanner.step(
                t=t,
                snapshots=snapshots,
                results=results,
                current_goals=current_goals,
                base_goals=base_goals,
                mission_change=mission_change if step == change_step else None,
            )
            if first_recovery_step is None and replanner.counters.plans_committed:
                first_recovery_step = step

        # Exercise the exact adapter command path for every drone.
        timestamp_ms = int(time.time_ns() // 1_000_000)
        latency_ms = int(fault.get("command_latency_ms") or 0)
        telemetry_timestamp_ms = timestamp_ms - stale_ms
        for i in env.agent_ids:
            if i in failed or i in aborted:
                continue
            position_flu = (
                float(env.positions[i, 0]),
                float(env.positions[i, 1]),
                float(CRUISE_ALTITUDE_M),
            )
            command = build_phase5_command(
                drone=i,
                safe_velocity=results[i].safe_action,
                position_flu=position_flu,
                timestamp_ms=timestamp_ms - latency_ms,
                priority=overrides.priority.get(i, "normal"),
            )
            telemetry_state = flu_snapshot_to_telemetry_state(
                i,
                position_flu,
                (0.0, 0.0, 0.0),
                timestamp_ms=telemetry_timestamp_ms,
            )
            allowed, reason = validate_command_path(
                command,
                telemetry_state,
                now_ms=timestamp_ms,
            )
            emitted_commands += 1
            if not allowed:
                rejected_commands += 1
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        final = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
        if c1_px4_supervisor is not None and c1_px4_supervisor.brake_latched:
            final = {i: np.zeros(2, dtype=np.float64) for i in env.agent_ids}
        active_agents = [i for i in env.agent_ids if i not in failed and i not in aborted]
        if active_agents and all(np.linalg.norm(final[i]) < 0.05 for i in active_agents):
            waiting_streak += 1
            max_waiting_streak = max(max_waiting_streak, waiting_streak)
        else:
            waiting_streak = 0
        previous_positions = env.positions.copy()
        if execution_tau_s > 0.0:
            if execution_audit_samples:
                audit = _audit_exact_zoh_interval(
                    env, final, execution_tau_s, ra, execution_audit_samples
                )
                audit["step"] = step
                intersample_records.append(audit)
            infos = _step_exact_zoh_execution(env, final, execution_tau_s)
        else:
            _, _, _, _, infos = env.step(final)

        path_length_m += float(np.linalg.norm(env.positions - previous_positions, axis=1).sum())

        if len(env.agent_ids) > 1:
            positions = env.positions
            distances = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=-1
            )
            np.fill_diagonal(distances, np.inf)
            min_pairwise_distance_m = min(
                min_pairwise_distance_m, float(np.min(distances))
            )

        if critical_goal is not None:
            for i in env.agent_ids:
                if i in failed or i in aborted:
                    continue
                if np.linalg.norm(env.positions[i] - np.asarray(critical_goal)) < env.scenario.goal_epsilon:
                    critical_reached = True

        if high_drone is not None and high_reached_step is None:
            if np.linalg.norm(env.goals[high_drone] - env.positions[high_drone]) < env.scenario.goal_epsilon:
                high_reached_step = step

        reached_base = {
            i
            for i in env.agent_ids
            if np.linalg.norm(env.positions[i] - env.goals[i]) < env.scenario.goal_epsilon
        }
        if c1_px4_supervisor is not None:
            all_done = c1_px4_supervisor.complete
        else:
            all_done = (
                critical_reached
                if critical_goal is not None
                else all(i in aborted or i in reached_base for i in env.agent_ids)
            )
        if any(i["collided"] for i in infos.values()):
            collision = True
            completion_step = step + 1
            break
        if all_done:
            completed = True
            completion_step = step + 1
            break

        if real_time:
            time.sleep(env.scenario.dt)

    if replanner is not None:
        if critical_drone is not None:
            state = replanner.active.get(critical_drone)
            priority_violation = state is not None and state.goal_override is not None
        replanner.shutdown()

    ra_instances = [ra] if local_ras is None else list(local_ras.values())

    return {
        "observation_mode": observation_mode,
        "collision": collision,
        "completed": completed,
        "priority_violation": priority_violation,
        "completion_steps": completion_step,
        "cbf_events": cbf_events,
        "min_rho": round(min_rho, 6) if min_rho != float("inf") else None,
        "min_pairwise_distance_m": (
            round(min_pairwise_distance_m, 6)
            if min_pairwise_distance_m != float("inf")
            else None
        ),
        "control_effort": (
            round(control_effort_sum / control_effort_n, 4)
            if control_effort_n
            else None
        ),
        "zone_crossed": zone_crossed,
        "critical_reached": critical_reached,
        "high_reached_step": high_reached_step,
        "emitted_commands": emitted_commands,
        "rejected_commands": rejected_commands,
        "rejected_reasons": rejected_reasons,
        "qp_solve_steps": qp_solve_steps,
        "qp_infeasible_steps": qp_infeasible_steps,
        "hocbf_primary_infeasible_steps": sum(
            controller.hocbf_primary_infeasible_count for controller in ra_instances
        ),
        "hocbf_predictive_infeasible_steps": sum(
            controller.hocbf_predictive_infeasible_count for controller in ra_instances
        ),
        "hocbf_recovery_infeasible_steps": sum(
            controller.hocbf_recovery_infeasible_count for controller in ra_instances
        ),
        "hocbf_recovery_steps": sum(
            controller.hocbf_recovery_steps for controller in ra_instances
        ),
        "hocbf_recovery_entries": sum(
            controller.hocbf_recovery_entries for controller in ra_instances
        ),
        "qp_latency_ms": qp_latency_ms,
        "intersample_audit_samples": execution_audit_samples,
        "intersample_records": intersample_records if execution_audit_samples else None,
        "recovery_step": first_recovery_step,
        "recovery_time_s": (
            round((first_recovery_step - (change_step or 0)) * env.scenario.dt, 6)
            if first_recovery_step is not None
            else None
        ),
        "path_length_m": round(path_length_m, 6),
        "max_waiting_duration_s": round(max_waiting_streak * env.scenario.dt, 6),
        "max_repeated_cbf_duration_s": round(
            max_intervention_streak * env.scenario.dt, 6
        ),
        "recovery_ra_veto_steps": recovery_ra_veto_steps,
        "safety_bypass_count": 0,
        "c1_px4_supervisor": (
            c1_px4_supervisor.summary()
            if c1_px4_supervisor is not None
            else None
        ),
        "counters": (
            {
                "triggers": replanner.counters.triggers,
                "plans_committed": replanner.counters.plans_committed,
                "llm_plans_committed": replanner.counters.llm_plans_committed,
                "fallback_plans_committed": replanner.counters.fallback_plans_committed,
                "mission_changes": replanner.counters.mission_changes,
                "llm_timeouts": replanner.counters.llm_timeouts,
                "llm_syntactic_invalid": replanner.counters.llm_syntactic_invalid,
                "llm_schema_invalid": replanner.counters.llm_schema_invalid,
                "llm_semantic_invalid": replanner.counters.llm_semantic_invalid,
                "llm_execution_invalid": replanner.counters.llm_execution_invalid,
                "llm_stale_invalid": replanner.counters.llm_stale_invalid,
            }
            if replanner is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Live MQTT path (requires an already-running PX4 SITL + adapter + broker).
# ---------------------------------------------------------------------------


def _reset_velocity_command(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return a bounded FLU velocity that homes a vehicle without mode changes.

    Gazebo ``set_pose`` changes only the model pose; it does not reset PX4's
    EKF/offboard state.  Repositioning must therefore travel through the same
    velocity-offboard path as the measured crossing.  Horizontal speed is
    norm-bounded so diagonals cannot exceed the adapter safety limit.
    """
    error = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    horizontal = error[:2]
    distance = float(np.linalg.norm(horizontal))
    if distance > 0.0:
        horizontal *= min(RESET_MAX_SPEED_MPS / distance, 1.0)
    vertical = float(np.clip(error[2], -RESET_MAX_VERTICAL_SPEED_MPS, RESET_MAX_VERTICAL_SPEED_MPS))
    return (float(horizontal[0]), float(horizontal[1]), vertical)


def run_mqtt_loop(
    *,
    drone_ids: list[int],
    base_goals: dict[int, tuple[float, float, float]],
    mode: str,
    llm_client,
    llm_fallback,
    host: str,
    port: int,
    max_steps: int,
    trajectory: Path | None = None,
    reset_starts: dict[int, tuple[float, float, float]] | None = None,
    rate_hz: float = 10.0,
    use_hocbf: bool = False,
    hocbf_k1: float = 3.0,
    hocbf_k2: float = 3.0,
    a_max: float = 2.0,
    kv: float = 2.0,
    tau_ctrl: float = 0.0,
    tau_px4: float = 0.0,
    tau_px4_min: float | None = None,
    tau_px4_max: float | None = None,
    execution_model: str = "exact_zoh",
    sampled_data: bool = False,
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
    gamma: float = 0.1,
    zocbf_delta: float = 0.0,
    pb_alpha: float = 2.0,
    pb_braking_accel: float | None = None,
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
    ra_command_feedforward_tau_s: float | None = None,
    sequential_pass: bool = False,
    urgent_drone: int | None = None,
    estimator_config: SharedStateEstimatorConfig | None = None,
    estimator_seed: int = 0,
    replan_timeout_s: float | None = None,
    pilot: MappoPilot | None = None,
    coordination_intent_pilot: MappoPilot | None = None,
    execution_supervisor_config: ExecutionSupervisorConfig | None = None,
    freshness_gate_config: TelemetryFreshnessConfig | None = None,
    land_at_end: bool = True,
    mission_change: dict | None = None,
    change_step: int | None = None,
    failed_drone: int | None = None,
    blocked_zone: tuple[float, float, float, float] | None = None,
    critical_goal: tuple[float, float] | None = None,
    goal_epsilon: float = 0.5,
    velocity_command_mode: str = "safe_action",
    tau_command_s: float = 0.7,
    command_hold_jitter_probability: float = 0.0,
    command_hold_jitter_seed: int = 0,
    observation_mode: str = OBSERVATION_LEGACY_ASYMMETRIC,
    ra_params: RuntimeAssuranceParams | None = None,
    c1_px4_supervisor_config: C1Px4SupervisorConfig | None = None,
    admission_coordinator: C3AdmissionCoordinator | None = None,
    reservation_coordinator: C3ReservationCoordinator | None = None,
    space_time_reservation_coordinator: C3SpaceTimeReservationCoordinator | None = None,
    group_slot_coordinator: C3GroupSlotCoordinator | None = None,
    recoverability_admission_coordinator: Any | None = None,
) -> dict[str, Any]:
    """Live PX4 loop: arm/takeoff, then RA-filtered closed-loop control.

    ``arm``/``takeoff``/``land`` use the adapter-native ``px4/{id}/command``
    topic; the continuous ``move_to`` stream uses ``swarm/drone/{id}/command``
    with FLU coordinates, matching the frozen adapter contract.
    """
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("paho-mqtt is required for --mqtt mode") from exc

    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"unknown observation_mode: {observation_mode}")
    if not 0.0 <= command_hold_jitter_probability <= 1.0:
        raise ValueError("command_hold_jitter_probability must be in [0, 1]")
    coordination_count = sum(
        value is not None
        for value in (
            admission_coordinator,
            reservation_coordinator,
            space_time_reservation_coordinator,
            group_slot_coordinator,
        )
    ) + int(sequential_pass)
    if coordination_count > 1:
        raise ValueError(
            "SEQUENTIAL_PASS、C3-Admission、C3-Reservation、C3-STR 与 C3-GroupSlot 必须独立运行"
        )
    active_coordination_coordinator = (
        admission_coordinator
        or reservation_coordinator
        or space_time_reservation_coordinator
        or group_slot_coordinator
    )
    params = ra_params or RuntimeAssuranceParams(tau_ctrl=tau_ctrl)

    def make_ra() -> RuntimeAssurance:
        return RuntimeAssurance(
            params=params,
            v_max=1.5,
            use_hocbf=use_hocbf,
            hocbf_k1=hocbf_k1,
            hocbf_k2=hocbf_k2,
            a_max=a_max,
            kv=kv,
            tau_px4=tau_px4,
            tau_px4_min=tau_px4_min,
            tau_px4_max=tau_px4_max,
            execution_model=execution_model,
            sampled_data=sampled_data,
            sampled_data_method=sampled_data_method,
            pcbf_horizon=pcbf_horizon,
            pcbf_terminal_buffer_m=pcbf_terminal_buffer_m,
            pcbf_terminal_velocity_tolerance_mps=pcbf_terminal_velocity_tolerance_mps,
            pcbf_position_bound_m=pcbf_position_bound_m,
            pcbf_velocity_bound_mps=pcbf_velocity_bound_mps,
            pcbf_max_iterations=pcbf_max_iterations,
            pcbf_multistart_count=pcbf_multistart_count,
            pcbf_tolerance=pcbf_tolerance,
            pcbf_acceptable_tolerance=pcbf_acceptable_tolerance,
            pcbf_lexicographic_tolerance=pcbf_lexicographic_tolerance,
            gamma=gamma,
            zocbf_delta=zocbf_delta,
            pb_alpha=pb_alpha,
            pb_braking_accel=pb_braking_accel,
            constraint_boundary=constraint_boundary,
            hocbf_boundary_guard=hocbf_boundary_guard,
            hocbf_boundary_buffer_m=hocbf_boundary_buffer_m,
            hocbf_infeasible_fallback=hocbf_infeasible_fallback,
            hocbf_pb_recovery=hocbf_pb_recovery,
            hocbf_predictive_recovery=hocbf_predictive_recovery,
            hocbf_prediction_execution_fraction=hocbf_prediction_execution_fraction,
            hocbf_prediction_steps=hocbf_prediction_steps,
            hocbf_recovery_reserve_threshold=hocbf_recovery_reserve_threshold,
            hocbf_recovery_alpha=hocbf_recovery_alpha,
            hocbf_recovery_braking_accel=hocbf_recovery_braking_accel,
            hocbf_recovery_boundary_buffer_m=hocbf_recovery_boundary_buffer_m,
            hocbf_recovery_clear_steps=hocbf_recovery_clear_steps,
            sampled_data_boundary_buffer_m=sampled_data_boundary_buffer_m,
            sampled_data_infeasible_fallback=sampled_data_infeasible_fallback,
            command_feedforward_tau_s=ra_command_feedforward_tau_s,
        )

    ra = make_ra()
    local_ras = (
        {drone: make_ra() for drone in drone_ids}
        if observation_mode == OBSERVATION_LOCAL_FRESH_SELF
        else None
    )
    execution_supervisor = (
        ExecutionConformanceSupervisor(execution_supervisor_config)
        if execution_supervisor_config is not None
        else None
    )
    freshness_gate = TelemetryFreshnessGate(freshness_gate_config)
    if c1_px4_supervisor_config is not None and reset_starts is None:
        raise ValueError("C1 PX4 supervisor requires reset_starts")
    if c1_px4_supervisor_config is not None:
        if not math.isclose(
            tau_px4, c1_px4_supervisor_config.barrier_tau_px4_s, abs_tol=1e-12
        ):
            raise ValueError("C1 barrier_tau_px4_s must match RA tau_px4")
        if velocity_command_mode != "feedforward_tau":
            raise ValueError("C1 PX4 supervisor requires feedforward_tau commands")
        if not math.isclose(
            tau_command_s,
            c1_px4_supervisor_config.command_feedforward_tau_s,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "C1 command_feedforward_tau_s must match tau_command_s"
            )
    c1_px4_supervisor = (
        C1Px4AdmissionSupervisor(
            config=c1_px4_supervisor_config,
            drone_ids=drone_ids,
            base_goals=base_goals,
            reset_starts=reset_starts or {},
            rate_hz=rate_hz,
            ra_params=params,
            v_max=ra.v_max,
        )
        if c1_px4_supervisor_config is not None
        else None
    )
    replanner = (
        _make_replanner(
            client=llm_client,
            fallback=llm_fallback,
            dt=1.0 / rate_hz,
            config=(
                ReplanConfig(replan_timeout_s=replan_timeout_s)
                if replan_timeout_s is not None
                else ReplanConfig()
            ),
        )
        if mode == "ASYNC"
        else None
    )
    if replanner is not None and recoverability_admission_coordinator is not None:
        raise ValueError(
            "immediate replanner 与 recoverability admission 必须作为独立条件运行"
        )
    overrides = RecoveryOverrides()
    telemetry: dict[int, DroneSnapshot] = {}
    telemetry_health: dict[int, dict[str, bool | int]] = {}
    estimator = (
        SharedStateEstimator(estimator_config)
        if estimator_config is not None
        else None
    )
    estimator_rng = (
        np.random.default_rng(estimator_seed) if estimator is not None else None
    )

    def publish(
        drone: int,
        action: str,
        *,
        target: tuple[float, float, float] | None = None,
        velocity: tuple[float, float, float] | None = None,
        altitude_m: float | None = None,
        native: bool = False,
        priority: str = "normal",
    ) -> dict[str, Any]:
        timestamp_ms = int(time.time_ns() // 1_000_000)
        payload: dict[str, Any] = {
            "drone": drone,
            "action": action,
            "ttl_sec": NATIVE_COMMAND_TTL_S if native else COMMAND_TTL_S,
            "command_id": f"phase5-{action}-{drone}-{timestamp_ms}",
            "timestamp_ms": timestamp_ms,
            "source_frame": "PX4_NED" if native else "FLU",
            "priority": priority,
        }
        if target is not None:
            payload["target"] = list(target)
            payload["waypoint"] = list(target)
        if velocity is not None:
            payload["velocity"] = list(velocity)
        if altitude_m is not None:
            payload["altitude_m"] = altitude_m
        topic = f"px4/{drone}/command" if native else f"swarm/drone/{drone}/command"
        client.publish(topic, json.dumps(payload, separators=(",", ":")))
        return payload

    def publish_payload(payload: dict[str, Any], *, native: bool = False) -> None:
        topic = (
            f"px4/{payload['drone']}/command"
            if native
            else f"swarm/drone/{payload['drone']}/command"
        )
        client.publish(topic, json.dumps(payload, separators=(",", ":")))

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe("swarm/drone/+/telemetry")
        client.subscribe("px4/+/command_ack")

    def on_message(client, userdata, msg):
        if msg.topic.endswith("/telemetry"):
            try:
                payload = json.loads(msg.payload.decode())
            except (ValueError, UnicodeDecodeError):
                return
            snap = snapshot_from_telemetry(payload)
            telemetry[snap.drone_id] = snap
            telemetry_health[snap.drone_id] = {
                "armed": bool(payload.get("armed", False)),
                "failsafe": bool(payload.get("failsafe", False)),
                "connection_lost": bool(payload.get("connection_lost", False)),
                "nav_state": int(payload.get("nav_state", -1)),
            }

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="aegisair-phase5")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=30)
    client.loop_start()

    try:
        deadline = time.time() + 30.0
        while time.time() < deadline and not all(i in telemetry for i in drone_ids):
            time.sleep(1.0 / rate_hz)
        if not all(i in telemetry for i in drone_ids):
            raise SystemExit("telemetry not ready for all drones")

        # Arm, then takeoff to the frozen cruise altitude.
        arm_deadline = time.time() + 30.0
        while time.time() < arm_deadline and not all(
            telemetry.get(i) is not None
            and telemetry[i].status == "armed"
            and not telemetry_health.get(i, {}).get("failsafe", True)
            and not telemetry_health.get(i, {}).get("connection_lost", True)
            for i in drone_ids
        ):
            for i in drone_ids:
                publish(i, "arm", native=True)
            time.sleep(0.5)
        if not all(
            telemetry.get(i) is not None
            and telemetry[i].status == "armed"
            and not telemetry_health.get(i, {}).get("failsafe", True)
            and not telemetry_health.get(i, {}).get("connection_lost", True)
            for i in drone_ids
        ):
            raise SystemExit("PX4 did not arm cleanly before live episode")

        takeoff_deadline = time.time() + 30.0
        while time.time() < takeoff_deadline and not all(
            telemetry.get(i) is not None
            and telemetry[i].position[2] > CRUISE_ALTITUDE_M * 0.5
            for i in drone_ids
        ):
            for i in drone_ids:
                publish(i, "takeoff", altitude_m=CRUISE_ALTITUDE_M, native=True)
            time.sleep(0.5)
        if not all(
            telemetry.get(i) is not None
            and telemetry[i].position[2] > CRUISE_ALTITUDE_M * 0.5
            for i in drone_ids
        ):
            raise SystemExit("PX4 did not reach cruise altitude before live episode")

        # Reset through PX4 velocity-offboard, never Gazebo teleport.  A model
        # teleport leaves EKF/offboard state behind and can silently invalidate
        # the subsequent crossing.  A reset is accepted only after both the
        # position error and measured residual speed hold below frozen bounds.
        reset_elapsed_s: float | None = None
        if reset_starts is not None:
            reset_started = time.monotonic()
            reset_deadline = reset_started + RESET_TIMEOUT_S
            stable_since: float | None = None
            while time.monotonic() < reset_deadline:
                ready = True
                for i in drone_ids:
                    snap = telemetry.get(i)
                    health = telemetry_health.get(i, {})
                    if (
                        snap is None
                        or snap.status != "armed"
                        or bool(health.get("failsafe", True))
                        or bool(health.get("connection_lost", True))
                    ):
                        ready = False
                        continue
                    position_error = float(
                        np.linalg.norm(
                            np.asarray(reset_starts[i]) - np.asarray(snap.position)
                        )
                    )
                    speed = float(np.linalg.norm(snap.velocity or (float("inf"),) * 3))
                    if (
                        position_error > RESET_POSITION_TOLERANCE_M
                        or speed > RESET_SPEED_TOLERANCE_MPS
                    ):
                        ready = False
                    publish(
                        i,
                        "velocity",
                        velocity=_reset_velocity_command(snap.position, reset_starts[i]),
                        native=False,
                    )
                now = time.monotonic()
                if ready:
                    stable_since = stable_since or now
                    if now - stable_since >= RESET_STABLE_S:
                        reset_elapsed_s = now - reset_started
                        break
                else:
                    stable_since = None
                time.sleep(1.0 / rate_hz)
            if reset_elapsed_s is None:
                diagnostics = {
                    i: {
                        "position": telemetry[i].position if i in telemetry else None,
                        "velocity": telemetry[i].velocity if i in telemetry else None,
                        "health": telemetry_health.get(i),
                    }
                    for i in drone_ids
                }
                raise SystemExit(
                    "PX4 velocity reset failed to settle within "
                    f"{RESET_TIMEOUT_S:.0f}s: {diagnostics}"
                )

        period = 1.0 / rate_hz
        step = 0
        cbf_events = 0
        min_dist: float | None = None
        min_rho: float | None = None
        traj_rows: list[dict[str, Any]] = []
        failed: set[int] = set()
        critical_reached = False
        zone_crossed = False
        collision = False
        recovery_step: int | None = None
        path_length_m = 0.0
        ra_solve_latency_ms: list[float] = []
        previous_positions: dict[int, tuple[float, float, float]] = {}
        # This is an explicit test-side execution perturbation: a selected command
        # update is held for one control interval.  It never changes the RA state
        # estimate or controller parameters, and is logged per command.
        jitter_rng = np.random.default_rng(command_hold_jitter_seed)
        command_hold_jitter_count = 0
        safety_bypass_count = 0
        published_command_mismatch_count = 0
        published_command_constraint_unknown_count = 0
        published_command_constraint_failure_count = 0
        published_constraint_failure_while_solver_feasible_count = 0
        infrastructure_valid = True
        infrastructure_invalid_reasons: set[str] = set()
        previous_issued_velocity = {
            drone: np.zeros(2, dtype=np.float64) for drone in drone_ids
        }
        seq_state = {
            "active": False,
            "order": _priority_order(drone_ids, urgent_drone),
            "idx": 0,
            "best": {i: float("inf") for i in drone_ids},
            "stall": {i: 0 for i in drone_ids},
            "holding": {},
            "completed": set(),
        }
        previous_ra_results = None
        coordination_decision = None
        reservation_decision = None
        space_time_reservation_decision = None
        group_slot_decision = None
        next_deadline = time.monotonic()
        while step < max_steps:
            t = step * period
            timestamp_ms = int(time.time_ns() // 1_000_000)
            snapshots = {i: telemetry[i] for i in drone_ids if i in telemetry}
            if len(snapshots) != len(drone_ids):
                time.sleep(0.05)
                continue
            unhealthy = {
                i: telemetry_health.get(i, {})
                for i in drone_ids
                if telemetry_health.get(i, {}).get("failsafe", False)
                or telemetry_health.get(i, {}).get("connection_lost", False)
            }
            if unhealthy:
                raise SystemExit(
                    "PX4 health lost during live episode; trial is invalid: "
                    f"{unhealthy}"
                )

            if (
                change_step is not None
                and step >= change_step
                and failed_drone is not None
            ):
                failed.add(failed_drone)

            if recoverability_admission_coordinator is not None:
                overrides = recoverability_admission_coordinator.step(
                    step=step,
                    snapshots=snapshots,
                    base_goals=base_goals,
                    mission_change=(
                        mission_change if step == change_step else None
                    ),
                )
                if (
                    recovery_step is None
                    and recoverability_admission_coordinator.plans_committed > 0
                ):
                    recovery_step = step

            execution_residuals = (
                execution_supervisor.observe(snapshots, now_t_s=t)
                if execution_supervisor is not None
                else {}
            )

            if sequential_pass:
                for i in drone_ids:
                    dist = float(
                        np.linalg.norm(
                            np.asarray(snapshots[i].position[:2])
                            - np.asarray(base_goals[i][:2])
                        )
                    )
                    if dist < goal_epsilon:
                        seq_state["completed"].add(i)
                        seq_state["stall"][i] = 0
                        continue
                    if dist < seq_state["best"][i] - 0.02:
                        seq_state["best"][i] = dist
                        seq_state["stall"][i] = 0
                    else:
                        seq_state["stall"][i] += 1
                pending = [
                    i for i in drone_ids if i not in seq_state["completed"]
                ]
                if not seq_state["active"] and len(pending) > 1 and any(
                    seq_state["stall"][i] >= 8 for i in pending
                ):
                    seq_state["active"] = True
                    seq_state["idx"] = 0
                    seq_state["best"] = {i: float("inf") for i in drone_ids}
                    seq_state["stall"] = {i: 0 for i in drone_ids}
                    seq_state["order"] = _resolve_priority_order(
                        llm_client,
                        pending,
                        urgent_drone,
                        snapshots,
                        base_goals,
                        timestamp_ms,
                    )
                    seq_state["holding"] = _freeze_holding_points(
                        pending,
                        {i: np.asarray(snapshots[i].position) for i in drone_ids},
                    )
                if seq_state["active"]:
                    while (
                        seq_state["idx"] < len(seq_state["order"])
                        and seq_state["order"][seq_state["idx"]]
                        in seq_state["completed"]
                    ):
                        seq_state["idx"] += 1
                    if seq_state["idx"] >= len(seq_state["order"]):
                        seq_state["active"] = False
                        for i in drone_ids:
                            overrides.goal_override.pop(i, None)
                            overrides.velocity_scale[i] = 1.0
                        continue
                    right_of_way = seq_state["order"][seq_state["idx"]]
                    for i in drone_ids:
                        if i == right_of_way or i in seq_state["completed"]:
                            overrides.velocity_scale[i] = 1.0
                            overrides.goal_override.pop(i, None)
                            continue
                        holding = seq_state["holding"][i]
                        if (
                            np.linalg.norm(
                                np.asarray(snapshots[i].position[:2])
                                - np.asarray(holding[:2])
                            )
                            < 0.5
                        ):
                            overrides.goal_override.pop(i, None)
                            overrides.velocity_scale[i] = 0.0
                        else:
                            overrides.goal_override[i] = holding
                            overrides.velocity_scale[i] = 0.6
                    if (
                        np.linalg.norm(
                            np.asarray(snapshots[right_of_way].position[:2])
                            - np.asarray(base_goals[right_of_way][:2])
                        )
                        < 0.5
                    ):
                        seq_state["completed"].add(right_of_way)
                        seq_state["idx"] += 1
                        if seq_state["idx"] >= len(seq_state["order"]):
                            seq_state["active"] = False
                            for i in drone_ids:
                                overrides.goal_override.pop(i, None)
                                overrides.velocity_scale[i] = 1.0

            if active_coordination_coordinator is not None:
                intent_snapshots = {
                    i: replace(snapshots[i], target=base_goals[i]) for i in drone_ids
                }
                intent_pilot = coordination_intent_pilot or pilot
                if intent_pilot is not None:
                    intent_nominal = intent_pilot.actions(
                        positions={
                            i: np.asarray(snapshots[i].position) for i in drone_ids
                        },
                        velocities={
                            i: np.asarray(snapshots[i].velocity or (0.0, 0.0, 0.0))
                            for i in drone_ids
                        },
                        goals={i: np.asarray(base_goals[i][:2]) for i in drone_ids},
                    )
                else:
                    intent_nominal = {
                        i: np.clip(
                            NOMINAL_GAIN
                            * (
                                np.asarray(base_goals[i][:2])
                                - np.asarray(snapshots[i].position[:2])
                            ),
                            -ra.v_max,
                            ra.v_max,
                        )
                        for i in drone_ids
                    }
                directives = active_coordination_coordinator.directives(
                    step=step,
                    timestamp_ms=timestamp_ms,
                    snapshots=intent_snapshots,
                    nominal=intent_nominal,
                    previous_results=previous_ra_results,
                )
                if group_slot_coordinator is not None:
                    group_slot_decision = directives.decision
                    coordination_decision = directives.decision
                elif space_time_reservation_coordinator is not None:
                    space_time_reservation_decision = directives.decision
                elif reservation_coordinator is not None:
                    reservation_decision = directives.decision
                else:
                    coordination_decision = directives.decision
                for i in drone_ids:
                    if i in directives.goal_overrides:
                        overrides.goal_override[i] = directives.goal_overrides[i]
                    else:
                        overrides.goal_override.pop(i, None)
                    overrides.velocity_scale[i] = directives.velocity_scales.get(i, 1.0)

            if c1_px4_supervisor is not None:
                c1_goals, c1_scales = c1_px4_supervisor.directives(
                    step=step,
                    snapshots=snapshots,
                    goal_epsilon=goal_epsilon,
                )
                for i in drone_ids:
                    if i in c1_goals:
                        overrides.goal_override[i] = c1_goals[i]
                    else:
                        overrides.goal_override.pop(i, None)
                    overrides.velocity_scale[i] = c1_scales[i]

            if pilot is not None:
                positions = {
                    i: np.asarray(snapshots[i].position) for i in drone_ids
                }
                velocities = {
                    i: np.asarray(snapshots[i].velocity or (0.0, 0.0, 0.0))
                    for i in drone_ids
                }
                goals = {
                    i: np.asarray(overrides.goal_override.get(i, base_goals[i])[:2])
                    for i in drone_ids
                }
                nominal = pilot.actions(
                    positions=positions, velocities=velocities, goals=goals
                )
            else:
                nominal = {}
                for i in drone_ids:
                    snap = snapshots[i]
                    goal = overrides.goal_override.get(i, base_goals[i])
                    delta = np.asarray(goal[:2]) - np.asarray(snap.position[:2])
                    nominal[i] = np.clip(
                        NOMINAL_GAIN * delta,
                        -ra.v_max,
                        ra.v_max,
                    )

            for i in drone_ids:
                if i in overrides.aborted or i in failed:
                    nominal[i] = np.zeros(2)
                    continue
                nominal[i] = np.asarray(nominal[i], dtype=np.float64) * overrides.velocity_scale.get(i, 1.0)
            if reservation_coordinator is not None:
                reservation_coordinator.record_nominal(nominal)
            if space_time_reservation_coordinator is not None:
                space_time_reservation_coordinator.record_nominal(nominal)

            fixed_actions = {
                i: np.zeros(2, dtype=np.float64)
                for i in drone_ids
                if i in failed or i in overrides.aborted
            }

            # The CBF boundary grows with the age of the telemetry used for
            # filtering.  Feed the measured round-trip age instead of assuming
            # zero latency, otherwise the live PX4 position-control loop can
            # close faster than the lightweight point-mass simulation.  When
            # the SharedStateEstimator is enabled, RA consumes its delayed,
            # dropout-prone, covariance-bearing output instead of raw telemetry.
            raw_ages = {
                i: max(
                    0.0, (timestamp_ms - snapshots[i].timestamp_ms) / 1000.0
                )
                for i in drone_ids
            }
            if estimator is not None:
                ra_states, aoi = _estimated_states_and_aoi(
                    snapshots, estimator, timestamp_ms, estimator_rng
                )
                ages = {
                    i: max(
                        0.0, (timestamp_ms - ra_states[i].timestamp_ms) / 1000.0
                    )
                    for i in drone_ids
                }
            else:
                ra_states = snapshots
                ages = dict(raw_ages)
                aoi = {
                    (i, j): max(ages[i], ages[j])
                    for i in drone_ids
                    for j in drone_ids
                    if i != j
                }
            input_ages = {
                i: max(raw_ages[i], ages[i])
                for i in drone_ids
            }
            freshness_decision = freshness_gate.assess(
                required_drones=drone_ids,
                telemetry_ages_s=input_ages,
            )
            if not freshness_decision.fresh:
                infrastructure_valid = False
                infrastructure_invalid_reasons.update(freshness_decision.reasons)
            if freshness_decision.active:
                fixed_actions.update(
                    {
                        i: np.zeros(2, dtype=np.float64)
                        for i in drone_ids
                        if i not in failed and i not in overrides.aborted
                    }
                )
            ra_states = _propagate_states(ra_states, ages)
            solve_started = time.perf_counter()
            if local_ras is None:
                results = ra.filter(
                    ra_states,
                    nominal,
                    t=t,
                    aoi=aoi,
                    fixed_actions=fixed_actions,
                )
            else:
                results = {}
                for drone in drone_ids:
                    local_view, local_aoi = _local_ra_view(
                        observer=drone,
                        fresh_states=snapshots,
                        peer_states=ra_states,
                        now_ms=timestamp_ms,
                    )
                    results[drone] = local_ras[drone].filter(
                        local_view,
                        nominal,
                        t=t,
                        aoi=local_aoi,
                        fixed_actions=fixed_actions,
                    )[drone]
            solve_elapsed_s = time.perf_counter() - solve_started
            ra_solve_latency_ms.append(solve_elapsed_s * 1000.0)
            solver_selected_velocity = {
                i: np.asarray(results[i].safe_action, dtype=np.float64).copy()
                for i in drone_ids
            }
            solver_feasible = {
                i: results[i].feasible
                for i in drone_ids
            }
            solver_velocity_saturated = {
                i: bool(results[i].vel_saturated)
                for i in drone_ids
            }
            if c1_px4_supervisor is not None:
                c1_px4_supervisor.observe_ra(results)
            if active_coordination_coordinator is not None:
                observed_decision = active_coordination_coordinator.observe_ra(results)
                if group_slot_coordinator is not None:
                    group_slot_decision = observed_decision
                    coordination_decision = observed_decision
                elif space_time_reservation_coordinator is not None:
                    space_time_reservation_decision = observed_decision
                elif reservation_coordinator is not None:
                    reservation_decision = observed_decision
                else:
                    coordination_decision = observed_decision
                previous_ra_results = results
            execution_decision = None
            if execution_supervisor is not None:
                supervisor_state = execution_supervisor.assess(
                    residuals=execution_residuals,
                    qp_feasible=all(r.feasible is not False for r in results.values()),
                    solve_elapsed_s=solve_elapsed_s,
                    telemetry_ages_s=input_ages,
                    snapshots=snapshots,
                )
                execution_decision = ExecutionAssuranceDecision(
                    active=supervisor_state.active,
                    reasons=list(supervisor_state.reasons),
                    residuals={
                        drone: {
                            "dt_s": residual.dt_s,
                            "velocity_mps": residual.velocity_mps,
                            "position_m": residual.position_m,
                        }
                        for drone, residual in supervisor_state.residuals.items()
                    },
                    recoverability=[
                        {
                            "drones": pair.drones,
                            "separation_m": pair.separation_m,
                            "closing_speed_mps": pair.closing_speed_mps,
                            "required_distance_m": pair.required_distance_m,
                            "margin_m": pair.margin_m,
                        }
                        for pair in supervisor_state.recoverability
                    ],
                    backup_drones=list(supervisor_state.backup_drones),
                    qp_feasible=all(r.feasible is not False for r in results.values()),
                    solve_elapsed_s=solve_elapsed_s,
                    telemetry_ages_s=input_ages,
                    consecutive_bad=supervisor_state.consecutive_bad,
                    consecutive_clean=supervisor_state.consecutive_clean,
                    timestamp_ms=timestamp_ms,
                )
                if supervisor_state.active and not freshness_decision.active:
                    backup_candidates = (
                        supervisor_state.backup_drones or tuple(drone_ids)
                    )
                    controlled_drones = tuple(
                        drone
                        for drone in backup_candidates
                        if results[drone].control_authority
                    )
                    for i in controlled_drones:
                        backup = deterministic_backup_velocity(
                            drone=i,
                            snapshots=snapshots,
                            dt_s=period,
                            a_max=ra.a_max,
                            v_max=ra.v_max,
                            retreat_speed_mps=execution_supervisor.config.backup_speed_mps,
                        )
                        velocity = np.asarray(
                            snapshots[i].velocity[:2]
                            if snapshots[i].velocity is not None
                            else (0.0, 0.0),
                            dtype=np.float64,
                        )
                        backup_accel = (backup - velocity) / period
                        results[i] = replace(
                            results[i],
                            mode="override",
                            safe_action=(float(backup[0]), float(backup[1])),
                            a_safe=(float(backup_accel[0]), float(backup_accel[1])),
                            intervened=True,
                        )
            if freshness_decision.fresh:
                cbf_events += sum(1 for r in results.values() if r.intervened)
            positions = [snapshots[i].position for i in drone_ids]
            step_min_dist = float("inf")
            for a in range(len(positions)):
                for b in range(a + 1, len(positions)):
                    dist = math.hypot(
                        positions[a][0] - positions[b][0],
                        positions[a][1] - positions[b][1],
                        positions[a][2] - positions[b][2],
                    )
                    step_min_dist = min(step_min_dist, dist)
                    if freshness_decision.fresh:
                        min_dist = dist if min_dist is None else min(min_dist, dist)
            if (
                freshness_decision.fresh
                and step_min_dist != float("inf")
                and step_min_dist < 0.25
            ):
                collision = True
            if (
                freshness_decision.fresh
                and critical_goal is not None
                and not critical_reached
                and (change_step is None or step >= change_step)
            ):
                for i in drone_ids:
                    if i in failed:
                        continue
                    if (
                        np.linalg.norm(
                            np.asarray(snapshots[i].position[:2])
                            - np.asarray(critical_goal)
                        )
                        < goal_epsilon
                    ):
                        critical_reached = True
            if freshness_decision.fresh and blocked_zone is not None and not zone_crossed:
                for i in drone_ids:
                    if _in_zone(snapshots[i].position, blocked_zone):
                        zone_crossed = True
            for i in drone_ids:
                if i in previous_positions:
                    path_length_m += float(
                        np.linalg.norm(
                            np.asarray(snapshots[i].position)
                            - np.asarray(previous_positions[i])
                        )
                    )
                previous_positions[i] = snapshots[i].position
            if replanner is not None:
                current_goals = {
                    i: overrides.goal_override.get(i, base_goals[i])
                    for i in drone_ids
                }
                overrides = replanner.step(
                    t=t,
                    snapshots=snapshots,
                    results=results,
                    current_goals=current_goals,
                    base_goals=base_goals,
                    mission_change=(
                        mission_change if step == change_step else None
                    ),
                )
                if recovery_step is None and replanner.counters.plans_committed:
                    recovery_step = step

            step_min_rho = min(results[i].safety_margin for i in drone_ids)
            if freshness_decision.fresh:
                min_rho = (
                    step_min_rho
                    if min_rho is None
                    else min(min_rho, step_min_rho)
                )

            step_rows: dict[int, dict[str, Any]] = {}
            issued_commands: dict[int, np.ndarray] = {}
            command_audits: dict[int, dict[str, Any]] = {}
            for i in drone_ids:
                snap = snapshots[i]
                command_authority, ra_bypass = _command_authority(
                    failed_or_aborted=i in failed or i in overrides.aborted
                )
                safety_bypass_count += int(ra_bypass)
                vertical_velocity = max(
                    -MAX_VERTICAL_SPEED_MPS,
                    min(
                        MAX_VERTICAL_SPEED_MPS,
                        ALTITUDE_HOLD_KP * (CRUISE_ALTITUDE_M - snap.position[2]),
                    ),
                )
                fallback_reason = None
                if i in failed or i in overrides.aborted:
                    velocity_2d = np.zeros(2, dtype=np.float64)
                    fallback_reason = "CONTROL_AUTHORITY_REVOKED"
                else:
                    velocity_2d = _velocity_for_command_mode(
                        results[i], velocity_command_mode
                    )
                    if velocity_command_mode == "nominal":
                        fallback_reason = "NOMINAL_COMMAND_MODE"
                    elif not np.allclose(
                        velocity_2d,
                        solver_selected_velocity[i],
                        atol=1e-9,
                        rtol=0.0,
                    ):
                        fallback_reason = "EXECUTION_SUPERVISOR"
                if (
                    c1_px4_supervisor is not None
                    and c1_px4_supervisor.brake_latched
                ):
                    velocity_2d = np.zeros(2, dtype=np.float64)
                    fallback_reason = "C1_BRAKE_LATCH"
                requested_velocity = np.asarray(velocity_2d, dtype=np.float64)
                command_held = False
                if freshness_decision.active:
                    command_authority = "freshness_hold"
                    fallback_reason = "+".join(freshness_decision.reasons)
                    command = build_phase5_hold_command(
                        drone=i,
                        timestamp_ms=timestamp_ms,
                        priority="safety",
                    )
                else:
                    command_held = bool(
                        command_hold_jitter_probability > 0.0
                        and jitter_rng.random() < command_hold_jitter_probability
                    )
                    if command_held:
                        velocity_2d = previous_issued_velocity[i].copy()
                        command_hold_jitter_count += 1
                        fallback_reason = "COMMAND_HOLD_JITTER"
                    command = build_phase5_velocity_command(
                        drone=i,
                        safe_velocity=velocity_2d,
                        vertical_velocity=vertical_velocity,
                        timestamp_ms=timestamp_ms,
                        priority=overrides.priority.get(i, "normal"),
                    )
                    previous_issued_velocity[i] = np.asarray(
                        command["velocity"][:2], dtype=np.float64
                    )
                    issued_commands[i] = previous_issued_velocity[i].copy()
                publish_payload(command)
                command_scale_s = (
                    ra.command_feedforward_tau_s
                    if ra.command_feedforward_tau_s is not None
                    else (params.degradation_dt if step == 0 else period)
                )
                reference_velocity = np.asarray(
                    ra_states[i].velocity[:2]
                    if ra_states[i].velocity is not None
                    else (0.0, 0.0),
                    dtype=np.float64,
                )
                command_audit = _audit_published_command(
                    solver_selected_velocity=solver_selected_velocity[i],
                    published_command=command,
                    reference_velocity=reference_velocity,
                    command_scale_s=command_scale_s,
                    fallback_reason=fallback_reason,
                )
                command_audits[i] = command_audit
                execution_residual = execution_residuals.get(i)
                execution_residual_ok = None
                if (
                    execution_supervisor is not None
                    and execution_residual is not None
                ):
                    execution_residual_ok = bool(
                        execution_residual.velocity_mps
                        <= execution_supervisor.config.velocity_residual_limit_mps
                        and execution_residual.position_m
                        <= execution_supervisor.config.position_residual_limit_m
                    )
                if trajectory is not None:
                    step_rows[i] = {
                        "pos": list(snap.position),
                        "v_actual": list(snap.velocity or (0.0, 0.0, 0.0)),
                        "v_safe": command.get("velocity"),
                        "v_requested": [
                            float(requested_velocity[0]),
                            float(requested_velocity[1]),
                            0.0,
                        ],
                        "command_hold_jitter_applied": command_held,
                        "v_nom": [
                            float(results[i].nominal_action[0]),
                            float(results[i].nominal_action[1]),
                            0.0,
                        ],
                        "rho": round(results[i].safety_margin, 6),
                        "intervened": bool(results[i].intervened),
                        "age_s": round(ages[i], 4),
                        "a_nom": list(results[i].a_nom or (0.0, 0.0)),
                        "a_safe": list(results[i].a_safe or (0.0, 0.0)),
                        "d_safe": results[i].d_safe,
                        "accel_saturated": bool(results[i].accel_saturated),
                        "vel_saturated": bool(results[i].vel_saturated),
                        "feasible": results[i].feasible,
                        "selected_filter": results[i].selected_filter,
                        "primary_feasible": results[i].primary_feasible,
                        "predictive_feasible": results[i].predictive_feasible,
                        "recovery_active": results[i].recovery_active,
                        "recovery_reason": results[i].recovery_reason,
                        "feasibility_reserve": results[i].feasibility_reserve,
                        "pcbf_status": results[i].pcbf_status,
                        "pcbf_stage1_status": results[i].pcbf_stage1_status,
                        "pcbf_stage2_status": results[i].pcbf_stage2_status,
                        "pcbf_terminal_feasible": results[i].pcbf_terminal_feasible,
                        "pcbf_value": results[i].pcbf_value,
                        "pcbf_slack_sum": results[i].pcbf_slack_sum,
                        "pcbf_tracking_cost": results[i].pcbf_tracking_cost,
                        "pcbf_max_constraint_violation": results[i].pcbf_max_constraint_violation,
                        "pcbf_tie_break_applied": results[i].pcbf_tie_break_applied,
                        "pcbf_warm_start_used": results[i].pcbf_warm_start_used,
                        "pcbf_fail_closed_reason": results[i].pcbf_fail_closed_reason,
                        "control_authority": results[i].control_authority,
                        "fixed_action": (
                            list(results[i].fixed_action)
                            if results[i].fixed_action is not None
                            else None
                        ),
                        "command_authority": command_authority,
                        "ra_bypass": ra_bypass,
                        "measurement_timestamp_ms": snap.timestamp_ms,
                        "source_timestamp_us": snap.source_timestamp_us,
                        "raw_telemetry_age_s": round(raw_ages[i], 4),
                        "solver_input_age_s": round(ages[i], 4),
                        "telemetry_fresh": freshness_decision.fresh,
                        "freshness_gate_active": freshness_decision.active,
                        "freshness_reason": list(freshness_decision.reasons),
                        "execution_residual_ok": execution_residual_ok,
                        "published_min_constraint_slack": None,
                        "solver_feasible": solver_feasible[i],
                        **command_audit,
                        "c1_admission_phase": (
                            c1_px4_supervisor.phase
                            if c1_px4_supervisor is not None
                            else None
                        ),
                        "c1_brake_latched": (
                            c1_px4_supervisor.brake_latched
                            if c1_px4_supervisor is not None
                            else False
                        ),
                    }

            published_accelerations = {
                drone: np.asarray(
                    audit["published_effective_acceleration"],
                    dtype=np.float64,
                )
                for drone, audit in command_audits.items()
                if audit["published_effective_acceleration"] is not None
            }
            independently_checked = (
                local_ras is None
                and len(published_accelerations) == len(drone_ids)
            )
            if independently_checked:
                joint_constraint_ok, published_min_constraint_slack = (
                    ra.audit_published_accelerations(published_accelerations)
                )
            else:
                joint_constraint_ok = None
                published_min_constraint_slack = None
            if joint_constraint_ok is None:
                joint_constraint_ok = _joint_published_constraint_status(
                    command_audits=command_audits,
                    solver_feasible=solver_feasible,
                    solver_velocity_saturated=solver_velocity_saturated,
                )
            joint_command_matches = all(
                audit["published_command_matches_selected"]
                for audit in command_audits.values()
            )
            published_command_mismatch_count += sum(
                not audit["published_command_matches_selected"]
                for audit in command_audits.values()
            )
            published_command_constraint_unknown_count += (
                len(drone_ids) if joint_constraint_ok is None else 0
            )
            published_command_constraint_failure_count += (
                len(drone_ids) if joint_constraint_ok is False else 0
            )
            published_constraint_failure_while_solver_feasible_count += (
                len(drone_ids)
                if joint_constraint_ok is False
                and _joint_solver_feasible(solver_feasible)
                else 0
            )
            for row in step_rows.values():
                row["all_published_commands_match_selected"] = joint_command_matches
                row["published_command_constraint_ok"] = joint_constraint_ok
                row["published_min_constraint_slack"] = (
                    published_min_constraint_slack
                )

            if execution_supervisor is not None:
                execution_supervisor.record_commands(
                    snapshots,
                    issued_commands,
                    now_t_s=t,
                )

            if trajectory is not None:
                traj_rows.append(
                    {
                        "t": round(t, 4),
                        "step": step,
                        "min_rho": round(step_min_rho, 6),
                        "min_distance": round(
                            step_min_dist if step_min_dist != float("inf") else 0.0,
                            4,
                        ),
                        "ra_solve_latency_ms": solve_elapsed_s * 1000.0,
                        "drones": step_rows,
                        "input_freshness": {
                            "fresh": freshness_decision.fresh,
                            "active": freshness_decision.active,
                            "reasons": list(freshness_decision.reasons),
                            "stale_drones": list(freshness_decision.stale_drones),
                            "missing_drones": list(freshness_decision.missing_drones),
                            "consecutive_fresh": freshness_decision.consecutive_fresh,
                            "raw_telemetry_ages_s": raw_ages,
                            "solver_input_ages_s": ages,
                        },
                        "execution_assurance": (
                            execution_decision.model_dump(mode="json")
                            if execution_decision is not None
                            else None
                        ),
                        "coordination_admission": (
                            coordination_decision.model_dump(mode="json")
                            if coordination_decision is not None
                            else None
                        ),
                        "recoverability_admission": (
                            recoverability_admission_coordinator.summary()
                            if recoverability_admission_coordinator is not None
                            else None
                        ),
                        "coordination_reservation": (
                            reservation_decision.model_dump(mode="json")
                            if reservation_decision is not None
                            else None
                        ),
                        "coordination_space_time_reservation": (
                            space_time_reservation_decision.model_dump(mode="json")
                            if space_time_reservation_decision is not None
                            else None
                        ),
                    }
                )
            step += 1
            next_deadline += period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_deadline = time.monotonic() + period

        final_positions = {
            i: list(snapshots[i].position) for i in drone_ids if i in snapshots
        }
        if trajectory is not None:
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            trajectory.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in traj_rows)
                + "\n",
                encoding="utf-8",
            )
        if land_at_end:
            for i in drone_ids:
                publish(i, "land", native=True)
            time.sleep(2.0)
    finally:
        client.loop_stop()
        client.disconnect()
        if replanner is not None:
            replanner.shutdown()

    return {
        "steps": step,
        "mode": mode,
        "observation_mode": observation_mode,
        "drone_ids": drone_ids,
        "cbf_events": cbf_events,
        "min_distance_m": round(min_dist, 4) if min_dist is not None else None,
        "min_rho": round(min_rho, 6) if min_rho is not None else None,
        "failed_drone": failed_drone,
        "critical_reached": critical_reached,
        "zone_crossed": zone_crossed,
        "collision": collision,
        "recovery_step": recovery_step,
        "recovery_time_s": (
            round((recovery_step - change_step) * period, 6)
            if recovery_step is not None and change_step is not None
            else None
        ),
        "path_length_m": round(path_length_m, 6),
        "ra_solve_latency_ms": ra_solve_latency_ms,
        "ra_solve_latency_summary_ms": _latency_summary_ms(
            ra_solve_latency_ms, deadline_ms=period * 1000.0
        ),
        "command_hold_jitter_probability": command_hold_jitter_probability,
        "command_hold_jitter_count": command_hold_jitter_count,
        "safety_bypass_count": safety_bypass_count,
        "published_command_mismatch_count": published_command_mismatch_count,
        "published_command_constraint_unknown_count": published_command_constraint_unknown_count,
        "published_command_constraint_failure_count": published_command_constraint_failure_count,
        "published_constraint_failure_while_solver_feasible_count": (
            published_constraint_failure_while_solver_feasible_count
        ),
        "infrastructure_valid": infrastructure_valid,
        "infrastructure_invalid_reasons": sorted(infrastructure_invalid_reasons),
        "freshness_gate": {
            "max_age_s": freshness_gate.config.max_age_s,
            "release_samples": freshness_gate.config.release_samples,
            "trip_count": freshness_gate.trip_count,
            "release_count": freshness_gate.release_count,
            "active_steps": freshness_gate.active_steps,
            "stale_steps": freshness_gate.stale_steps,
            "max_observed_age_s": freshness_gate.max_observed_age_s,
            "reason_counts": freshness_gate.reason_counts,
            "active_at_end": freshness_gate.active,
        },
        "hocbf_primary_infeasible_steps": ra.hocbf_primary_infeasible_count,
        "hocbf_predictive_infeasible_steps": ra.hocbf_predictive_infeasible_count,
        "hocbf_recovery_infeasible_steps": ra.hocbf_recovery_infeasible_count,
        "hocbf_recovery_steps": ra.hocbf_recovery_steps,
        "hocbf_recovery_entries": ra.hocbf_recovery_entries,
        "final_positions": final_positions,
        "trajectory": str(trajectory) if trajectory is not None else None,
        "reset_elapsed_s": round(reset_elapsed_s, 3) if reset_elapsed_s is not None else None,
        "execution_supervisor": (
            {
                "trip_count": execution_supervisor.trip_count,
                "release_count": execution_supervisor.release_count,
                "active_steps": execution_supervisor.active_steps,
                "reason_counts": execution_supervisor.reason_counts,
                "active_at_end": execution_supervisor.active,
            }
            if execution_supervisor is not None
            else None
        ),
        "c1_px4_supervisor": (
            c1_px4_supervisor.summary()
            if c1_px4_supervisor is not None
            else None
        ),
        "c3_admission": (
            admission_coordinator.summary()
            if admission_coordinator is not None
            else None
        ),
        "c3_reservation": (
            reservation_coordinator.summary()
            if reservation_coordinator is not None
            else None
        ),
        "c3_space_time_reservation": (
            space_time_reservation_coordinator.summary()
            if space_time_reservation_coordinator is not None
            else None
        ),
        "c3_group_slot": (
            group_slot_coordinator.summary()
            if group_slot_coordinator is not None
            else None
        ),
        "recoverability_admission": (
            recoverability_admission_coordinator.summary()
            if recoverability_admission_coordinator is not None
            else None
        ),
        "counters": (
            {
                "triggers": replanner.counters.triggers,
                "plans_committed": replanner.counters.plans_committed,
                "llm_plans_committed": replanner.counters.llm_plans_committed,
                "fallback_plans_committed": replanner.counters.fallback_plans_committed,
                "mission_changes": replanner.counters.mission_changes,
                "llm_timeouts": replanner.counters.llm_timeouts,
                "llm_syntactic_invalid": replanner.counters.llm_syntactic_invalid,
                "llm_schema_invalid": replanner.counters.llm_schema_invalid,
                "llm_semantic_invalid": replanner.counters.llm_semantic_invalid,
                "llm_execution_invalid": replanner.counters.llm_execution_invalid,
                "llm_stale_invalid": replanner.counters.llm_stale_invalid,
            }
            if replanner is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on")
    parser.add_argument("--mode", choices=["CBF_ONLY", "ASYNC"], default="ASYNC")
    parser.add_argument(
        "--llm", choices=["rule", "qwen", "timeout", "invalid"], default="rule"
    )
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--qwen-max-tokens", type=int, default=48)
    parser.add_argument(
        "--pilot",
        choices=["go_to_goal", "checkpoint"],
        default="go_to_goal",
        help="Nominal pilot: rule-based go-to-goal or a trained MAPPO checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="MAPPO final.pt path (required with --pilot checkpoint).",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override the lightweight environment step time (seconds).",
    )
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--hocbf", action="store_true")
    parser.add_argument("--hocbf-k1", type=float, default=1.0)
    parser.add_argument("--hocbf-k2", type=float, default=1.0)
    parser.add_argument("--a-max", type=float, default=2.0)
    parser.add_argument("--kv", type=float, default=2.0)
    parser.add_argument("--tau-ctrl", type=float, default=0.0)
    parser.add_argument("--tau-px4", type=float, default=0.0)
    parser.add_argument("--tau-px4-min", type=float, default=None)
    parser.add_argument("--tau-px4-max", type=float, default=None)
    parser.add_argument("--sampled-data", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--execution-model",
        choices=["exact_zoh", "legacy_trapezoidal"],
        default="exact_zoh",
        help="Execution model for sampled-data RA (C2 E1/E2 axis).",
    )
    parser.add_argument("--sequential-pass", action="store_true")
    parser.add_argument("--urgent-drone", type=int, default=None)
    parser.add_argument("--replan-timeout-s", type=float, default=None)
    parser.add_argument(
        "--estimator",
        action="store_true",
        help="Enable SharedStateEstimator in --mqtt mode.",
    )
    parser.add_argument("--estimator-delay-ms", type=int, default=0)
    parser.add_argument("--estimator-dropout", type=float, default=0.0)
    parser.add_argument(
        "--estimator-measurement-cov-m2", type=float, default=0.01
    )
    parser.add_argument(
        "--estimator-process-noise-m2-per-s", type=float, default=0.02
    )
    parser.add_argument("--estimator-seed", type=int, default=0)
    parser.add_argument("--nominal-noise", type=float, default=0.0)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--mqtt", action="store_true")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Write per-step live telemetry/RA rows to this JSONL path.",
    )
    parser.add_argument(
        "--drone-ids",
        default="2,3",
        help="Comma-separated PX4 instance ids for live MQTT mode.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--multi-seed",
        type=int,
        default=0,
        help="Run one episode per seed 1..N with head-on lateral jitter.",
    )
    parser.add_argument(
        "--lateral",
        type=float,
        default=None,
        help="Head-on lateral offset for drone 2's goal y (drone 3 gets -y).",
    )
    parser.add_argument(
        "--starts",
        default=None,
        help="Semicolon-separated reset starts, e.g. 2=-3,0,2.5;3=3,0,2.5",
    )
    parser.add_argument(
        "--goals",
        default=None,
        help="Semicolon-separated live goals, e.g. 2=3,0,2.5;3=0,3,2.5",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    estimator_config = (
        SharedStateEstimatorConfig(
            delay_ms=args.estimator_delay_ms,
            dropout_rate=args.estimator_dropout,
            measurement_cov_m2=args.estimator_measurement_cov_m2,
            process_noise_m2_per_s=args.estimator_process_noise_m2_per_s,
        )
        if args.estimator
        else None
    )

    spec = _scenario(args.scenario)
    if args.dt is not None:
        spec["scenario"] = replace(spec["scenario"], dt=args.dt)

    if args.pilot == "checkpoint":
        if not args.checkpoint:
            parser.error("--checkpoint is required with --pilot checkpoint")
        try:
            from marllib.policies.mappo import MappoPilot
        except ModuleNotFoundError:
            parser.error(
                "--pilot checkpoint requires the non-public MARL checkpoint "
                "runtime, which is intentionally excluded from this reproduction release"
            )
        pilot = MappoPilot(
            args.checkpoint,
            obs_dim=4 + 5 * spec["scenario"].max_neighbors,
            num_agents=spec["scenario"].num_agents,
            speed_limit=spec["scenario"].speed_limit,
            max_neighbors=spec["scenario"].max_neighbors,
        )
    else:
        pilot = None

    if args.llm == "rule":
        llm_client = RuleMissionPlanner()
        llm_fallback = None
    elif args.llm == "qwen":
        llm_client = MlxLmClient(
            model_id=args.qwen_model,
            max_tokens=args.qwen_max_tokens,
            load=True,
        )
        llm_fallback = RuleMissionPlanner()
    elif args.llm == "timeout":
        llm_client = DeterministicRecoveryClient(plan_latency_s=0.3)
        llm_fallback = RuleMissionPlanner()
    else:  # invalid
        llm_client = _FaultedLlmClient({"action": "HOVER"})
        llm_fallback = RuleMissionPlanner()

    if args.mqtt:
        scenario: ScenarioConfig = spec["scenario"]
        drone_ids = [int(v) for v in args.drone_ids.split(",") if v != ""]
        if len(drone_ids) != scenario.num_agents:
            raise SystemExit(
                f"--drone-ids count {len(drone_ids)} != scenario agents "
                f"{scenario.num_agents}"
            )
        base_goals = {
            drone: (float(g[0]), float(g[1]), CRUISE_ALTITUDE_M)
            for drone, g in zip(drone_ids, scenario.goals)
        }
        if args.goals is not None:
            base_goals = {}
            for entry in args.goals.split(";"):
                if not entry:
                    continue
                id_part, vec = entry.split("=", 1)
                x, y, z = (float(v) for v in vec.split(","))
                base_goals[int(id_part)] = (x, y, z)
        if args.lateral is not None and args.scenario == "head_on":
            base_goals[drone_ids[0]] = (4.0, args.lateral, CRUISE_ALTITUDE_M)
            base_goals[drone_ids[1]] = (-4.0, -args.lateral, CRUISE_ALTITUDE_M)
        if args.starts is not None:
            reset_starts: dict[int, tuple[float, float, float]] = {}
            for entry in args.starts.split(";"):
                if not entry:
                    continue
                id_part, vec = entry.split("=", 1)
                x, y, z = (float(v) for v in vec.split(","))
                reset_starts[int(id_part)] = (x, y, z)
        else:
            reset_starts = {
                drone: (-3.0 if idx == 0 else 3.0, 0.0, CRUISE_ALTITUDE_M)
                for idx, drone in enumerate(drone_ids)
            }

        if args.multi_seed > 0:
            seed_results = []
            for seed in range(1, args.multi_seed + 1):
                lateral = random.Random(seed).uniform(-0.8, 0.8)
                seed_goals = dict(base_goals)
                if args.scenario == "head_on":
                    seed_goals[drone_ids[0]] = (
                        4.0,
                        lateral,
                        CRUISE_ALTITUDE_M,
                    )
                    seed_goals[drone_ids[1]] = (
                        -4.0,
                        -lateral,
                        CRUISE_ALTITUDE_M,
                    )
                trajectory = args.trajectory
                if trajectory is not None:
                    trajectory = trajectory.with_name(
                        f"{trajectory.stem}_seed{seed}{trajectory.suffix}"
                    )
                result = run_mqtt_loop(
                    drone_ids=drone_ids,
                    base_goals=seed_goals,
                    mode=args.mode,
                    llm_client=llm_client,
                    llm_fallback=llm_fallback,
                    host=args.host,
                    port=args.port,
                    max_steps=args.max_steps,
                    trajectory=trajectory,
                    reset_starts=reset_starts,
                    rate_hz=args.rate_hz,
                    use_hocbf=args.hocbf,
                    hocbf_k1=args.hocbf_k1,
                    hocbf_k2=args.hocbf_k2,
                    a_max=args.a_max,
                    kv=args.kv,
                    tau_ctrl=args.tau_ctrl,
                    tau_px4=args.tau_px4,
                    tau_px4_min=args.tau_px4_min,
                    tau_px4_max=args.tau_px4_max,
                    execution_model=args.execution_model,
                    sampled_data=args.sampled_data,
                    gamma=args.gamma,
                    sequential_pass=args.sequential_pass,
                    urgent_drone=args.urgent_drone,
                    estimator_config=estimator_config,
                    estimator_seed=args.estimator_seed,
                    replan_timeout_s=args.replan_timeout_s,
                    pilot=pilot,
                )
                seed_results.append(
                    {
                        "seed": seed,
                        "lateral": round(lateral, 4),
                        "min_rho": result["min_rho"],
                        "min_distance_m": result["min_distance_m"],
                        "cbf_events": result["cbf_events"],
                        "final_positions": result["final_positions"],
                    }
                )
                print(
                    json.dumps(seed_results[-1], ensure_ascii=False),
                    flush=True,
                )

            rho_vals = [r["min_rho"] for r in seed_results if r["min_rho"] is not None]
            print(
                json.dumps(
                    {
                        "seeds": args.multi_seed,
                        "min_rho_all": min(rho_vals) if rho_vals else None,
                        "results": seed_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        episodes = []
        for rep in range(args.repetitions):
            trajectory = args.trajectory
            if trajectory is not None and args.repetitions > 1:
                trajectory = trajectory.with_name(
                    f"{trajectory.stem}_rep{rep + 1}{trajectory.suffix}"
                )
            result = run_mqtt_loop(
                drone_ids=drone_ids,
                base_goals=base_goals,
                mode=args.mode,
                llm_client=llm_client,
                llm_fallback=llm_fallback,
                host=args.host,
                port=args.port,
                max_steps=args.max_steps,
                trajectory=trajectory,
                reset_starts=reset_starts,
                rate_hz=args.rate_hz,
                use_hocbf=args.hocbf,
                hocbf_k1=args.hocbf_k1,
                hocbf_k2=args.hocbf_k2,
                a_max=args.a_max,
                kv=args.kv,
                tau_ctrl=args.tau_ctrl,
                tau_px4=args.tau_px4,
                tau_px4_min=args.tau_px4_min,
                tau_px4_max=args.tau_px4_max,
                sampled_data=args.sampled_data,
                gamma=args.gamma,
                sequential_pass=args.sequential_pass,
                urgent_drone=args.urgent_drone,
                estimator_config=estimator_config,
                estimator_seed=args.estimator_seed,
                replan_timeout_s=args.replan_timeout_s,
                pilot=pilot,
            )
            episodes.append(result)
            print(
                json.dumps(
                    {
                        "rep": rep + 1,
                        "min_rho": result["min_rho"],
                        "min_distance_m": result["min_distance_m"],
                        "cbf_events": result["cbf_events"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        rho_vals = [e["min_rho"] for e in episodes if e["min_rho"] is not None]
        print(
            json.dumps(
                {
                    "repetitions": args.repetitions,
                    "min_rho_all": min(rho_vals) if rho_vals else None,
                    "episodes": episodes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    ra = RuntimeAssurance(
        params=RuntimeAssuranceParams(tau_ctrl=args.tau_ctrl),
        v_max=1.5,
        use_hocbf=args.hocbf,
        hocbf_k1=args.hocbf_k1,
        hocbf_k2=args.hocbf_k2,
        a_max=args.a_max,
        kv=args.kv,
        tau_px4=args.tau_px4,
        tau_px4_min=args.tau_px4_min,
        tau_px4_max=args.tau_px4_max,
        sampled_data=args.sampled_data,
        gamma=args.gamma,
    )
    env = MultiUAVEnv(spec["scenario"])
    runs = []
    for seed in range(1, args.seeds + 1):
        run = run_sim_episode(
            spec=spec,
            env=env,
            seed=seed,
            ra=ra,
            mode=args.mode,
            llm_client=llm_client,
            llm_fallback=llm_fallback,
            max_steps=args.max_steps,
            real_time=args.real_time,
            nominal_noise=args.nominal_noise,
            sequential_pass=args.sequential_pass,
            urgent_drone=args.urgent_drone,
            pilot=pilot,
        )
        runs.append(run)

    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "mode": args.mode,
                "llm": args.llm,
                "seeds": args.seeds,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
