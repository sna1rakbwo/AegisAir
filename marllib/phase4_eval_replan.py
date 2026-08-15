"""System-1 / System-2 mission-replanning comparison.

Compares:

    CBF_ONLY - Runtime Assurance alone (System 1, no LLM).
    SYNC     - Semantic Mission Manager that blocks the loop on the LLM.
    ASYNC    - Semantic Mission Manager that never blocks the loop.

The Mission Validity Monitor fires on route deviation (E_safety) or stalled
goal progress (E_coord); E_mission (external change) is exercised separately in
unit tests.  The LLM is judged by whether it restores mission consistency, not
whether it reacts inside one collision window.
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

from marllib.config import default_curriculum
from marllib.envs.multi_uav import MultiUAVEnv
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    AsyncMissionReplanner,
    DeterministicRecoveryClient,
    ReplanConfig,
    ReplanCounters,
    RecoveryOverrides,
)
from swarm.safety import DroneSnapshot


def _vec3(arr: np.ndarray) -> tuple[float, float, float]:
    return (float(arr[0]), float(arr[1]), 0.0)


def _snapshots(env: MultiUAVEnv) -> dict[int, DroneSnapshot]:
    return {
        i: DroneSnapshot(
            drone_id=i,
            position=_vec3(env.positions[i]),
            velocity=_vec3(env.velocities[i]),
        )
        for i in env.agent_ids
    }


def _go_to_goal(env: MultiUAVEnv) -> dict[int, np.ndarray]:
    actions: dict[int, np.ndarray] = {}
    for i in env.agent_ids:
        delta = env.goals[i] - env.positions[i]
        actions[i] = np.clip(1.5 * delta, -env.speed_limit, env.speed_limit)
    return actions


def _checkpoint_pilot(env: MultiUAVEnv, model) -> dict[int, np.ndarray]:
    import torch

    actions: dict[int, np.ndarray] = {}
    for i in env.agent_ids:
        obs = env.observation(i)
        action = (
            model.actor.deterministic(
                torch.from_numpy(obs).float().unsqueeze(0)
            )
            .squeeze(0)
            .numpy()
        )
        actions[i] = action
    return actions


def _cross_track(point: tuple[float, float], start: tuple[float, float], goal: tuple[float, float]) -> float:
    px, py = point[0] - start[0], point[1] - start[1]
    gx, gy = goal[0] - start[0], goal[1] - start[1]
    length_sq = gx * gx + gy * gy
    if length_sq == 0:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * gx + py * gy) / length_sq))
    cx, cy = start[0] + t * gx, start[1] + t * gy
    return math.hypot(point[0] - cx, point[1] - cy)


def _run_episode(
    *,
    env: MultiUAVEnv,
    seed: int,
    pilot,
    ra: RuntimeAssurance,
    make_replanner,
    max_steps: int,
    deviation_threshold_m: float,
    stall_window_s: float,
) -> tuple[dict, ReplanCounters | None]:
    env.reset(seed=seed)
    replanner = make_replanner() if make_replanner is not None else None
    base_goals = {i: _vec3(env.goals[i]) for i in env.agent_ids}
    base_starts = {i: _vec3(env.positions[i]) for i in env.agent_ids}
    overrides = RecoveryOverrides()
    aborted: set[int] = set()
    path_length = {i: 0.0 for i in env.agent_ids}
    straight = sum(
        math.hypot(base_goals[i][0] - base_starts[i][0], base_goals[i][1] - base_starts[i][1])
        for i in env.agent_ids
    )
    best_goal_dist = {
        i: math.hypot(env.positions[i][0] - base_goals[i][0], env.positions[i][1] - base_goals[i][1])
        for i in env.agent_ids
    }
    no_progress_since = {i: 0.0 for i in env.agent_ids}
    collision = False
    completed = False
    completion_step = max_steps
    intervention_events = 0
    intervention_duration_steps = 0
    had_route_deviation = False
    had_stall = False
    max_deviation = 0.0
    wall_start = time.perf_counter()

    for step in range(max_steps):
        t = step * env.scenario.dt
        prev_positions = env.positions.copy()

        for i in env.agent_ids:
            if i in overrides.aborted:
                aborted.add(i)
            elif i in overrides.goal_override:
                env.set_goal(i, np.asarray(overrides.goal_override[i][:2]))
            else:
                env.set_goal(i, np.asarray(base_goals[i][:2]))

        nominal = pilot(env)
        for i in env.agent_ids:
            if i in aborted:
                nominal[i] = np.zeros(2)
            else:
                nominal[i] = nominal[i] * overrides.velocity_scale.get(i, 1.0)

        snapshots = _snapshots(env)
        results = ra.filter(snapshots, nominal, t=t)

        step_intervened = {i for i in env.agent_ids if results[i].intervened}
        intervention_events += len(step_intervened)
        if step_intervened:
            intervention_duration_steps += 1

        # Mode-independent Mission Validity Monitor.
        for i in env.agent_ids:
            deviation = _cross_track(
                (env.positions[i, 0], env.positions[i, 1]),
                (base_starts[i][0], base_starts[i][1]),
                (base_goals[i][0], base_goals[i][1]),
            )
            max_deviation = max(max_deviation, deviation)
            if deviation > deviation_threshold_m:
                had_route_deviation = True

            goal_dist = math.hypot(
                env.positions[i, 0] - base_goals[i][0],
                env.positions[i, 1] - base_goals[i][1],
            )
            if goal_dist < best_goal_dist[i] - 0.05:
                best_goal_dist[i] = goal_dist
                no_progress_since[i] = t
            elif t - no_progress_since[i] >= stall_window_s:
                had_stall = True

        if replanner is not None:
            current_goals = {i: _vec3(env.goals[i]) for i in env.agent_ids}
            overrides = replanner.step(
                t=t,
                snapshots=snapshots,
                results=results,
                current_goals=current_goals,
                base_goals=base_goals,
            )

        final = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
        _, _, terminated, _, infos = env.step(final)

        for i in env.agent_ids:
            delta = env.positions[i] - prev_positions[i]
            path_length[i] += float(np.linalg.norm(delta))

        goal_epsilon = env.scenario.goal_epsilon
        reached_base = {
            i
            for i in env.agent_ids
            if np.linalg.norm(env.positions[i] - np.asarray(base_goals[i][:2]))
            < goal_epsilon
        }
        all_done = all(i in aborted or i in reached_base for i in env.agent_ids)
        if any(i["collided"] for i in infos.values()):
            collision = True
            completed = False
            completion_step = step + 1
            break
        if all_done:
            completed = True
            completion_step = step + 1
            break

    if replanner is not None:
        replanner.shutdown()

    total_path = sum(path_length.values()) or 1e-9
    metrics = {
        "collision": collision,
        "completed": completed,
        "completion_steps": completion_step,
        "intervention_events": intervention_events,
        "intervention_duration_steps": intervention_duration_steps,
        "had_route_deviation": had_route_deviation,
        "had_stall": had_stall,
        "had_mission_invalidation": had_route_deviation or had_stall,
        "max_deviation_m": max_deviation,
        "path_efficiency": straight / total_path,
        "wall_clock_s": time.perf_counter() - wall_start,
    }
    return metrics, (replanner.counters if replanner is not None else None)


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_8")
    parser.add_argument("--pilot", choices=["checkpoint", "go_to_goal"], default="checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["CBF_ONLY", "ASYNC"],
        choices=["CBF_ONLY", "SYNC", "ASYNC"],
    )
    parser.add_argument("--plan-latency-s", type=float, default=0.0)
    parser.add_argument("--deviation-threshold-m", type=float, default=1.5)
    parser.add_argument("--stall-window-s", type=float, default=3.0)
    parser.add_argument("--replan-cooldown-s", type=float, default=3.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scenario = next(s for s in default_curriculum() if s.name == args.scenario)
    env = MultiUAVEnv(scenario, max_steps=args.max_steps)

    if args.pilot == "checkpoint":
        if not args.checkpoint:
            parser.error("--checkpoint is required with --pilot checkpoint")
        import torch

        from marllib.policies.mappo import MAPPO

        model = MAPPO(env.obs_dim, 2, env.num_agents * 6, scenario.speed_limit)
        model.load_state_dict(torch.load(args.checkpoint, weights_only=False))

        def pilot(env: MultiUAVEnv) -> dict[int, np.ndarray]:
            return _checkpoint_pilot(env, model)
    else:

        def pilot(env: MultiUAVEnv) -> dict[int, np.ndarray]:
            return _go_to_goal(env)

    params = RuntimeAssuranceParams(
        d0=0.5,
        tau_r=0.3,
        a_eff=scenario.accel_limit,
        alpha=1.0,
        v_max=scenario.speed_limit,
    )
    ra = RuntimeAssurance(
        params,
        v_max=scenario.speed_limit,
        perception_sigma=0.0,
        rho_warn=0.2,
    )

    results = []
    for mode in args.modes:
        if mode == "CBF_ONLY":
            make_replanner = None
        else:
            config = ReplanConfig(
                route_deviation_threshold_m=args.deviation_threshold_m,
                stall_window_s=args.stall_window_s,
                replan_cooldown_s=args.replan_cooldown_s,
            )

            def make_replanner():
                return AsyncMissionReplanner(
                    client=DeterministicRecoveryClient(plan_latency_s=args.plan_latency_s),
                    config=config,
                    dt=scenario.dt,
                    blocking=(mode == "SYNC"),
                )

        episodes = []
        counters: list[ReplanCounters] = []
        for seed in range(args.eval_seeds):
            metrics, counter = _run_episode(
                env=env,
                seed=seed,
                pilot=pilot,
                ra=ra,
                make_replanner=make_replanner,
                max_steps=args.max_steps,
                deviation_threshold_m=args.deviation_threshold_m,
                stall_window_s=args.stall_window_s,
            )
            episodes.append(metrics)
            if counter is not None:
                counters.append(counter)

        n = args.eval_seeds
        result = {
            "mode": mode,
            "eval_seeds": n,
            "collision_rate": sum(e["collision"] for e in episodes) / n,
            "completion_rate": sum(e["completed"] for e in episodes) / n,
            "mean_completion_time_s": _mean(
                [e["completion_steps"] * scenario.dt for e in episodes]
            ),
            "mean_cbf_intervention_events": _mean(
                [e["intervention_events"] for e in episodes]
            ),
            "mean_cbf_intervention_duration_s": _mean(
                [e["intervention_duration_steps"] * scenario.dt for e in episodes]
            ),
            "mission_invalidation_rate": sum(
                e["had_mission_invalidation"] for e in episodes
            )
            / n,
            "route_deviation_rate": sum(e["had_route_deviation"] for e in episodes) / n,
            "stall_rate": sum(e["had_stall"] for e in episodes) / n,
            "mean_max_deviation_m": _mean([e["max_deviation_m"] for e in episodes]),
            "mean_path_efficiency": _mean([e["path_efficiency"] for e in episodes]),
            "mean_episode_wall_clock_s": _mean([e["wall_clock_s"] for e in episodes]),
        }
        if counters:
            result.update(
                {
                    "llm_calls": sum(c.llm_calls for c in counters),
                    "llm_syntactic_invalid": sum(
                        c.llm_syntactic_invalid for c in counters
                    ),
                    "llm_schema_invalid": sum(c.llm_schema_invalid for c in counters),
                    "llm_semantic_invalid": sum(c.llm_semantic_invalid for c in counters),
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
                    "mean_replan_latency_s": _mean(
                        [x for c in counters for x in c.replan_latencies_s]
                    ),
                }
            )
        results.append(result)

    payload = {"scenario": scenario.name, "results": results}
    print(json.dumps(payload, indent=2))
    output = args.output or "/Volumes/Expansion/safedrones_marllib_vec/phase4_replan_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2))
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
