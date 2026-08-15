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
) -> dict[str, Any]:
    """Best-effort live loop. Not part of the deterministic first gate."""
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
    stop = {"now": False}

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

        start = time.monotonic()
        step = 0
        while step < max_steps and not stop["now"]:
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

            results = ra.filter(snapshots, nominal, t=t)
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

            for i in drone_ids:
                command = build_phase5_command(
                    drone=i,
                    safe_velocity=results[i].safe_action,
                    position_flu=snapshots[i].position,
                    timestamp_ms=timestamp_ms,
                    priority=overrides.priority.get(i, "normal"),
                )
                client.publish(
                    f"swarm/drone/{i}/command",
                    json.dumps(command, separators=(",", ":")),
                )
            step += 1
            time.sleep(0.1)
    finally:
        client.loop_stop()
        client.disconnect()
        if replanner is not None:
            replanner.shutdown()

    return {"steps": step, "mode": mode}


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
        drone_ids = list(range(scenario.num_agents))
        base_goals = {
            i: (float(g[0]), float(g[1]), CRUISE_ALTITUDE_M)
            for i, g in zip(drone_ids, scenario.goals)
        }
        result = run_mqtt_loop(
            drone_ids=drone_ids,
            base_goals=base_goals,
            mode=args.mode,
            llm_client=llm_client,
            llm_fallback=llm_fallback,
            host=args.host,
            port=args.port,
            max_steps=args.max_steps,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
