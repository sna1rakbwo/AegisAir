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
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import ScenarioConfig
from marllib.envs.multi_uav import MultiUAVEnv
from px4_adapter.mqtt_codec import (
    TelemetryState,
    decode_command,
    normalize_command_to_ned,
)
from px4_adapter.px4_codec import flu_to_px4_ned
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    AsyncMissionReplanner,
    MlxLmClient,
    RecoveryOverrides,
    ReplanConfig,
    RuleMissionPlanner,
)
from swarm.safety import DroneSnapshot


CRUISE_ALTITUDE_M = 2.5
COMMAND_HORIZON_S = 0.5
COMMAND_TTL_S = 0.5
NOMINAL_GAIN = 1.5
ALTITUDE_HOLD_KP = 1.0
MAX_VERTICAL_SPEED_MPS = 1.0


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


def _go_to_goal(
    env: MultiUAVEnv,
    *,
    base_goals: dict[int, tuple[float, float, float]],
    overrides: RecoveryOverrides,
    failed: set[int],
    aborted: set[int],
) -> dict[int, np.ndarray]:
    actions: dict[int, np.ndarray] = {}
    for i in env.agent_ids:
        if i in failed or i in aborted:
            actions[i] = np.zeros(2)
            continue
        goal = overrides.goal_override.get(i, base_goals[i])
        delta = np.asarray(goal[:2]) - env.positions[i]
        speed = NOMINAL_GAIN * delta
        actions[i] = np.clip(speed, -env.scenario.speed_limit, env.scenario.speed_limit)
        actions[i] = actions[i] * overrides.velocity_scale.get(i, 1.0)
    return actions


def _in_zone(pos: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = zone
    return x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1


def _make_replanner(
    *,
    client,
    fallback,
    dt: float,
) -> AsyncMissionReplanner:
    return AsyncMissionReplanner(
        client=client,
        fallback=fallback,
        config=ReplanConfig(),
        dt=dt,
    )


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
) -> dict[str, Any]:
    env.reset(seed=seed)
    replanner = (
        _make_replanner(
            client=llm_client,
            fallback=llm_fallback,
            dt=env.scenario.dt,
        )
        if mode == "ASYNC"
        else None
    )
    base_goals = {
        i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0)
        for i in env.agent_ids
    }
    overrides = RecoveryOverrides()
    failed: set[int] = set()
    aborted: set[int] = set()
    change_step = spec.get("change_step")
    mission_change = spec.get("mission_change")
    failed_drone = spec.get("failed_drone")
    blocked_zone = spec.get("blocked_zone")
    critical_goal = spec.get("critical_goal")
    high_drone = spec.get("high_drone")

    collision = False
    completed = False
    completion_step = max_steps
    zone_crossed = False
    critical_reached = False
    high_reached_step = None
    cbf_events = 0
    emitted_commands = 0
    rejected_commands = 0

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

        nominal = _go_to_goal(
            env,
            base_goals=base_goals,
            overrides=overrides,
            failed=failed,
            aborted=aborted,
        )
        snapshots = _snapshots(env)
        results = ra.filter(snapshots, nominal, t=t)
        cbf_events += sum(1 for r in results.values() if r.intervened)

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

        # Exercise the exact adapter command path for every drone.
        timestamp_ms = int(time.time_ns() // 1_000_000)
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
                timestamp_ms=timestamp_ms,
                priority=overrides.priority.get(i, "normal"),
            )
            telemetry_state = flu_snapshot_to_telemetry_state(
                i,
                position_flu,
                (0.0, 0.0, 0.0),
                timestamp_ms=timestamp_ms,
            )
            allowed, reason = validate_command_path(
                command,
                telemetry_state,
                now_ms=timestamp_ms,
            )
            emitted_commands += 1
            if not allowed:
                rejected_commands += 1

        final = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
        _, _, terminated, _, infos = env.step(final)

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
            if np.linalg.norm(env.positions[i] - np.asarray(base_goals[i][:2])) < env.scenario.goal_epsilon
        }
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
        replanner.shutdown()

    return {
        "collision": collision,
        "completed": completed,
        "completion_steps": completion_step,
        "cbf_events": cbf_events,
        "zone_crossed": zone_crossed,
        "critical_reached": critical_reached,
        "high_reached_step": high_reached_step,
        "emitted_commands": emitted_commands,
        "rejected_commands": rejected_commands,
        "counters": (
            {
                "triggers": replanner.counters.triggers,
                "plans_committed": replanner.counters.plans_committed,
                "llm_plans_committed": replanner.counters.llm_plans_committed,
                "fallback_plans_committed": replanner.counters.fallback_plans_committed,
                "mission_changes": replanner.counters.mission_changes,
            }
            if replanner is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Live MQTT path (requires an already-running PX4 SITL + adapter + broker).
# ---------------------------------------------------------------------------


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

    ra = RuntimeAssurance(v_max=1.5)
    replanner = (
        _make_replanner(
            client=llm_client,
            fallback=llm_fallback,
            dt=0.1,
        )
        if mode == "ASYNC"
        else None
    )
    overrides = RecoveryOverrides()
    telemetry: dict[int, DroneSnapshot] = {}

    def publish(
        drone: int,
        action: str,
        *,
        target: tuple[float, float, float] | None = None,
        velocity: tuple[float, float, float] | None = None,
        altitude_m: float | None = None,
        native: bool = False,
        priority: str = "normal",
    ) -> None:
        payload: dict[str, Any] = {
            "drone": drone,
            "action": action,
            "ttl_sec": COMMAND_TTL_S,
            "command_id": f"phase5-{action}-{drone}-{int(time.time_ns() // 1_000_000)}",
            "timestamp_ms": int(time.time_ns() // 1_000_000),
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

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="aegisair-phase5")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=30)
    client.loop_start()

    try:
        deadline = time.time() + 30.0
        while time.time() < deadline and not all(i in telemetry for i in drone_ids):
            time.sleep(0.1)
        if not all(i in telemetry for i in drone_ids):
            raise SystemExit("telemetry not ready for all drones")

        # Arm, then takeoff to the frozen cruise altitude.
        arm_deadline = time.time() + 30.0
        while time.time() < arm_deadline and not all(
            telemetry.get(i) is not None and telemetry[i].status == "armed"
            for i in drone_ids
        ):
            for i in drone_ids:
                publish(i, "arm", native=True)
            time.sleep(0.5)

        takeoff_deadline = time.time() + 30.0
        while time.time() < takeoff_deadline and not all(
            telemetry.get(i) is not None
            and telemetry[i].position[2] > CRUISE_ALTITUDE_M * 0.5
            for i in drone_ids
        ):
            for i in drone_ids:
                publish(i, "takeoff", altitude_m=CRUISE_ALTITUDE_M, native=True)
            time.sleep(0.5)

        # Reset to the frozen start poses before the crossing so repeated
        # episodes do not start from the previous episode's goal positions.
        if reset_starts is not None:
            reset_deadline = time.time() + 45.0
            while time.time() < reset_deadline and not all(
                telemetry.get(i) is not None
                and math.hypot(
                    telemetry[i].position[0] - reset_starts[i][0],
                    telemetry[i].position[1] - reset_starts[i][1],
                )
                < 0.5
                for i in drone_ids
            ):
                for i in drone_ids:
                    publish(i, "move_to", target=reset_starts[i], native=False)
                time.sleep(0.3)

        start = time.monotonic()
        step = 0
        cbf_events = 0
        min_dist: float | None = None
        min_rho: float | None = None
        traj_rows: list[dict[str, Any]] = []
        while step < max_steps:
            t = time.monotonic() - start
            timestamp_ms = int(time.time_ns() // 1_000_000)
            snapshots = {i: telemetry[i] for i in drone_ids if i in telemetry}
            if len(snapshots) != len(drone_ids):
                time.sleep(0.05)
                continue

            nominal: dict[int, np.ndarray] = {}
            for i in drone_ids:
                snap = snapshots[i]
                if i in overrides.aborted:
                    nominal[i] = np.zeros(2)
                    continue
                goal = overrides.goal_override.get(i, base_goals[i])
                delta = np.asarray(goal[:2]) - np.asarray(snap.position[:2])
                nominal[i] = np.clip(
                    NOMINAL_GAIN * delta,
                    -ra.v_max,
                    ra.v_max,
                )
                nominal[i] = nominal[i] * overrides.velocity_scale.get(i, 1.0)

            # The CBF boundary grows with the age of the telemetry used for
            # filtering.  Feed the measured round-trip age instead of assuming
            # zero latency, otherwise the live PX4 position-control loop can
            # close faster than the lightweight point-mass simulation.
            ages = {
                i: max(0.0, (timestamp_ms - snapshots[i].timestamp_ms) / 1000.0)
                for i in drone_ids
            }
            aoi = {
                (i, j): max(ages[i], ages[j])
                for i in drone_ids
                for j in drone_ids
                if i != j
            }
            results = ra.filter(snapshots, nominal, t=t, aoi=aoi)
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
                    min_dist = dist if min_dist is None else min(min_dist, dist)
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
                )

            step_min_rho = min(results[i].safety_margin for i in drone_ids)
            min_rho = step_min_rho if min_rho is None else min(min_rho, step_min_rho)

            step_rows: dict[int, dict[str, Any]] = {}
            for i in drone_ids:
                snap = snapshots[i]
                vertical_velocity = max(
                    -MAX_VERTICAL_SPEED_MPS,
                    min(
                        MAX_VERTICAL_SPEED_MPS,
                        ALTITUDE_HOLD_KP * (CRUISE_ALTITUDE_M - snap.position[2]),
                    ),
                )
                command = build_phase5_velocity_command(
                    drone=i,
                    safe_velocity=results[i].safe_action,
                    vertical_velocity=vertical_velocity,
                    timestamp_ms=timestamp_ms,
                    priority=overrides.priority.get(i, "normal"),
                )
                publish(
                    i,
                    "velocity",
                    velocity=tuple(command["velocity"]),
                    native=False,
                    priority=command["priority"],
                )
                if trajectory is not None:
                    step_rows[i] = {
                        "pos": list(snap.position),
                        "v_actual": list(snap.velocity or (0.0, 0.0, 0.0)),
                        "v_safe": list(command["velocity"]),
                        "v_nom": [
                            float(results[i].nominal_action[0]),
                            float(results[i].nominal_action[1]),
                            0.0,
                        ],
                        "rho": round(results[i].safety_margin, 6),
                        "intervened": bool(results[i].intervened),
                        "age_s": round(ages[i], 4),
                    }

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
                        "drones": step_rows,
                    }
                )
            step += 1
            time.sleep(0.1)

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
        "drone_ids": drone_ids,
        "cbf_events": cbf_events,
        "min_distance_m": round(min_dist, 4) if min_dist is not None else None,
        "min_rho": round(min_rho, 6) if min_rho is not None else None,
        "final_positions": final_positions,
        "trajectory": str(trajectory) if trajectory is not None else None,
        "counters": (
            {
                "triggers": replanner.counters.triggers,
                "plans_committed": replanner.counters.plans_committed,
                "llm_plans_committed": replanner.counters.llm_plans_committed,
                "fallback_plans_committed": replanner.counters.fallback_plans_committed,
                "mission_changes": replanner.counters.mission_changes,
            }
            if replanner is not None
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="head_on")
    parser.add_argument("--mode", choices=["CBF_ONLY", "ASYNC"], default="ASYNC")
    parser.add_argument("--llm", choices=["rule", "qwen"], default="rule")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--qwen-max-tokens", type=int, default=48)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
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

    spec = _scenario(args.scenario)
    if args.llm == "rule":
        llm_client = RuleMissionPlanner()
        llm_fallback = None
    else:
        llm_client = MlxLmClient(
            model_id=args.qwen_model,
            max_tokens=args.qwen_max_tokens,
            load=True,
        )
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

    ra = RuntimeAssurance(v_max=1.5)
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
