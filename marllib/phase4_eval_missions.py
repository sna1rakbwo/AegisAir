"""Semantic-mission-replanning experiment harness.

Three mission scenarios from ``docs/decisions/llm_async_mission_replanning.md``:

    priority_conflict - a high-priority UAV crosses a normal-priority UAV.
    corridor_blocked  - a corridor suddenly becomes unavailable.
    drone_failure     - a UAV fails and its task must be reassigned.

Each is run with CBF only (System 1) versus the asynchronous Semantic Mission
Manager (System 1 + System 2), using a deterministic rule planner as the LLM
stand-in.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import ScenarioConfig
from marllib.envs.multi_uav import MultiUAVEnv
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    AsyncMissionReplanner,
    MlxLmClient,
    ReplanConfig,
    ReplanCounters,
    RecoveryOverrides,
    RuleMissionPlanner,
)
from swarm.safety import DroneSnapshot


def _snapshots(env: MultiUAVEnv) -> dict[int, DroneSnapshot]:
    return {
        i: DroneSnapshot(
            drone_id=i,
            position=(float(env.positions[i, 0]), float(env.positions[i, 1]), 0.0),
            velocity=(float(env.velocities[i, 0]), float(env.velocities[i, 1]), 0.0),
        )
        for i in env.agent_ids
    }


def _go_to_goal(env: MultiUAVEnv) -> dict[int, np.ndarray]:
    actions: dict[int, np.ndarray] = {}
    for i in env.agent_ids:
        delta = env.goals[i] - env.positions[i]
        actions[i] = np.clip(1.5 * delta, -env.speed_limit, env.speed_limit)
    return actions


def _in_zone(pos: np.ndarray, zone: tuple[float, float, float, float]) -> bool:
    x0, x1, y0, y1 = zone
    return x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1


def _priority_conflict() -> dict:
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
        "high_drone": 0,
    }


def _corridor_blocked() -> dict:
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
            "zone": [2.0, 2.5, -1.0, 1.0],
        },
        "change_step": 0,
        "blocked_zone": (2.0, 2.5, -1.0, 1.0),
    }


def _drone_failure() -> dict:
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
        "critical_goal": (4.0, 0.0),
    }


def _run_episode(
    *,
    spec: dict,
    env: MultiUAVEnv,
    seed: int,
    ra: RuntimeAssurance,
    make_replanner,
    max_steps: int,
    real_time: bool,
) -> tuple[dict, ReplanCounters | None]:
    env.reset(seed=seed)
    replanner = make_replanner() if make_replanner is not None else None
    base_goals = {i: (float(env.goals[i, 0]), float(env.goals[i, 1]), 0.0) for i in env.agent_ids}
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

        nominal = _go_to_goal(env)
        for i in env.agent_ids:
            if i in failed or i in aborted:
                nominal[i] = np.zeros(2)
            else:
                nominal[i] = nominal[i] * overrides.velocity_scale.get(i, 1.0)

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
        if critical_goal is not None:
            all_done = critical_reached
        else:
            all_done = all(i in aborted or i in reached_base for i in env.agent_ids)
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

    metrics = {
        "collision": collision,
        "completed": completed,
        "completion_steps": completion_step,
        "cbf_events": cbf_events,
        "zone_crossed": zone_crossed,
        "critical_reached": critical_reached,
        "high_reached_step": high_reached_step,
    }
    return metrics, (replanner.counters if replanner is not None else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument(
        "--modes", nargs="+", default=["CBF_ONLY", "ASYNC"], choices=["CBF_ONLY", "SYNC", "ASYNC"]
    )
    parser.add_argument("--llm", choices=["rule", "qwen"], default="rule")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    parser.add_argument("--qwen-max-tokens", type=int, default=300)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=["priority_conflict", "corridor_blocked", "drone_failure"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    specs = [_priority_conflict(), _corridor_blocked(), _drone_failure()]
    if args.scenarios is not None:
        specs = [s for s in specs if s["name"] in args.scenarios]
    params = RuntimeAssuranceParams(d0=0.5, tau_r=0.3, a_eff=2.0, alpha=1.0, v_max=1.5)
    ra = RuntimeAssurance(params, v_max=1.5, perception_sigma=0.0, rho_warn=0.2)
    qwen_client = None
    if args.llm == "qwen":
        qwen_client = MlxLmClient(
            model_id=args.qwen_model,
            max_tokens=args.qwen_max_tokens,
            load=True,
        )
    output = []

    for spec in specs:
        env = MultiUAVEnv(spec["scenario"], max_steps=args.max_steps)
        for mode in args.modes:
            if mode == "CBF_ONLY":
                make_replanner = None
            else:
                config = ReplanConfig(
                    route_deviation_threshold_m=1.5,
                    stall_window_s=3.0,
                    replan_cooldown_s=1.0,
                )

                def make_replanner():
                    if qwen_client is not None:
                        client = qwen_client
                        fallback = RuleMissionPlanner()
                    else:
                        client = RuleMissionPlanner()
                        fallback = None
                    return AsyncMissionReplanner(
                        client=client,
                        fallback=fallback,
                        config=config,
                        dt=env.scenario.dt,
                        blocking=(mode == "SYNC"),
                    )

            episodes = []
            counters: list[ReplanCounters] = []
            for seed in range(args.seeds):
                metrics, counter = _run_episode(
                    spec=spec,
                    env=env,
                    seed=seed,
                    ra=ra,
                make_replanner=make_replanner,
                max_steps=args.max_steps,
                real_time=args.real_time,
            )
                episodes.append(metrics)
                if counter is not None:
                    counters.append(counter)

            n = args.seeds
            result = {
                "scenario": spec["name"],
                "mode": mode,
                "seeds": n,
                "collision_rate": sum(e["collision"] for e in episodes) / n,
                "completion_rate": sum(e["completed"] for e in episodes) / n,
                "mean_completion_time_s": statistics.fmean(
                    [e["completion_steps"] * env.scenario.dt for e in episodes]
                ),
                "mean_cbf_events": statistics.fmean([e["cbf_events"] for e in episodes]),
            }
            if spec.get("blocked_zone") is not None:
                result["zone_cross_rate"] = sum(e["zone_crossed"] for e in episodes) / n
            if spec.get("critical_goal") is not None:
                result["critical_task_completion_rate"] = sum(
                    e["critical_reached"] for e in episodes
                ) / n
            if spec.get("high_drone") is not None:
                vals = [e["high_reached_step"] for e in episodes if e["high_reached_step"] is not None]
                result["high_priority_mean_reached_step"] = (
                    statistics.fmean(vals) if vals else None
                )
            if counters:
                result.update(
                    {
                        "llm_calls": sum(c.llm_calls for c in counters),
                        "llm_syntactic_invalid": sum(
                            c.llm_syntactic_invalid for c in counters
                        ),
                        "llm_schema_invalid": sum(c.llm_schema_invalid for c in counters),
                        "llm_semantic_invalid": sum(
                            c.llm_semantic_invalid for c in counters
                        ),
                        "llm_execution_invalid": sum(
                            c.llm_execution_invalid for c in counters
                        ),
                        "llm_timeouts": sum(c.llm_timeouts for c in counters),
                        "triggers": sum(c.triggers for c in counters),
                        "trigger_causes": [x for c in counters for x in c.trigger_causes],
                        "plans_committed": sum(c.plans_committed for c in counters),
                        "llm_plans_committed": sum(
                            c.llm_plans_committed for c in counters
                        ),
                        "fallback_plans_committed": sum(
                            c.fallback_plans_committed for c in counters
                        ),
                        "mission_changes": sum(c.mission_changes for c in counters),
                        "mean_replan_latency_s": statistics.fmean(
                            [x for c in counters for x in c.replan_latencies_s]
                        )
                        if any(c.replan_latencies_s for c in counters)
                        else None,
                    }
                )
            output.append(result)

    payload = {"results": output}
    print(json.dumps(payload, indent=2))
    out_path = args.output or "/Volumes/Expansion/safedrones_marllib_vec/phase4_missions_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
