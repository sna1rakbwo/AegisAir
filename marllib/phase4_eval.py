"""Phase 4 semantic-recovery comparison (R0 / R1 / R2).

Runs the same imperfect MARL (or a straight-to-goal) pilot through the Runtime
Assurance filter, then lets the semantic-recovery layer decide when to change
the mission.  The default backend is deterministic so the whole pipeline can be
verified without a large local model; ``--plan-latency-s`` can simulate LLM
generation latency to exercise the timeout budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
    DeterministicRecoveryClient,
    RecoveryConfig,
    RecoveryMode,
    RecoveryOverrides,
    SemanticRecovery,
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


def _run_episode(
    *,
    env: MultiUAVEnv,
    seed: int,
    pilot,
    ra: RuntimeAssurance,
    mode: str,
    client: DeterministicRecoveryClient,
    recovery_config: RecoveryConfig,
    max_steps: int,
) -> tuple[dict, RecoveryCounters]:
    from swarm.recovery import RecoveryCounters

    env.reset(seed=seed)
    base_goals = {i: _vec3(env.goals[i]) for i in env.agent_ids}
    recovery = SemanticRecovery(
        mode=mode,
        client=client,
        config=recovery_config,
        dt=env.scenario.dt,
    )
    overrides = RecoveryOverrides()
    aborted: set[int] = set()
    collision = False
    completed = False
    completion_step = max_steps
    intervention_steps = 0

    for step in range(max_steps):
        t = step * env.scenario.dt
        # 1. Apply the previous step's recovery intents to the environment.
        for i in env.agent_ids:
            if i in overrides.aborted:
                aborted.add(i)
            elif i in overrides.goal_override:
                env.set_goal(i, np.asarray(overrides.goal_override[i][:2]))
            else:
                env.set_goal(i, np.asarray(base_goals[i][:2]))

        # 2. Nominal pilot action (already sees any goal override).
        nominal = pilot(env)

        # 3. Apply velocity scaling / abort to the nominal action.
        for i in env.agent_ids:
            if i in aborted:
                nominal[i] = np.zeros(2)
            else:
                nominal[i] = nominal[i] * overrides.velocity_scale.get(i, 1.0)

        # 4. Runtime Assurance: evidence + minimally-invasive safe action.
        snapshots = _snapshots(env)
        results = ra.filter(snapshots, nominal, t=t)
        intervention_steps += int(any(r.intervened for r in results.values()))

        # 5. Feed evidence to semantic recovery (produces next-step overrides).
        current_goals = {i: _vec3(env.goals[i]) for i in env.agent_ids}
        overrides = recovery.step(
            t=t,
            snapshots=snapshots,
            results=results,
            current_goals=current_goals,
            base_goals=base_goals,
        )

        # 6. Final command is the RA-filtered action, never the raw LLM output.
        final = {i: np.asarray(results[i].safe_action) for i in env.agent_ids}
        _, _, terminated, _, infos = env.step(final)

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
        if terminated[0]:
            completed = False
            completion_step = step + 1
            break

    recovery.shutdown()
    metrics = {
        "collision": collision,
        "completed": completed,
        "completion_steps": completion_step,
        "intervention_steps": intervention_steps,
    }
    return metrics, recovery.counters


def _sum_counters(counters: list[RecoveryCounters]) -> dict:
    latencies = [x for c in counters for x in c.recovery_latencies_s]
    leads = [x for c in counters for x in c.candidate_lead_times_s]
    return {
        "llm_calls": sum(c.llm_calls for c in counters),
        "llm_timeouts": sum(c.llm_timeouts for c in counters),
        "llm_invalid": sum(c.llm_invalid for c in counters),
        "plans_executed": sum(c.plans_executed for c in counters),
        "mission_changes": sum(c.mission_changes for c in counters),
        "unnecessary_mission_changes": sum(
            c.unnecessary_mission_changes for c in counters
        ),
        "speculative_plans_generated": sum(
            c.speculative_plans_generated for c in counters
        ),
        "speculative_plans_discarded": sum(
            c.speculative_plans_discarded for c in counters
        ),
        "recovery_latency_s_mean": (
            statistics.fmean(latencies) if latencies else None
        ),
        "candidate_lead_time_s_mean": statistics.fmean(leads) if leads else None,
    }


def _run_mode(
    *,
    scenario,
    env: MultiUAVEnv,
    pilot,
    ra: RuntimeAssurance,
    mode: str,
    client: DeterministicRecoveryClient,
    recovery_config: RecoveryConfig,
    eval_seeds: int,
    max_steps: int,
) -> dict:
    counters: list[RecoveryCounters] = []
    collisions = 0
    completions = 0
    completion_times: list[float] = []
    intervention_totals: list[int] = []

    for seed in range(eval_seeds):
        metrics, counter = _run_episode(
            env=env,
            seed=seed,
            pilot=pilot,
            ra=ra,
            mode=mode,
            client=client,
            recovery_config=recovery_config,
            max_steps=max_steps,
        )
        counters.append(counter)
        collisions += int(metrics["collision"])
        completions += int(metrics["completed"])
        completion_times.append(metrics["completion_steps"] * scenario.dt)
        intervention_totals.append(metrics["intervention_steps"])

    result = _sum_counters(counters)
    result.update(
        {
            "mode": mode,
            "eval_seeds": eval_seeds,
            "collision_rate": collisions / eval_seeds,
            "completion_rate": completions / eval_seeds,
            "mean_completion_time_s": statistics.fmean(completion_times),
            "cbf_intervention_duration_s_mean": (
                statistics.fmean([x * scenario.dt for x in intervention_totals])
            ),
            "cbf_intervention_duration_s_total": (
                sum(x * scenario.dt for x in intervention_totals)
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="randomized_4")
    parser.add_argument("--pilot", choices=["checkpoint", "go_to_goal"], default="checkpoint")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["R0", "R1", "R2"],
        choices=["R0", "R1", "R2"],
    )
    parser.add_argument("--plan-latency-s", type=float, default=0.0)
    parser.add_argument("--latency-budget-s", type=float, default=0.6)
    parser.add_argument("--rho-warn", type=float, default=0.2)
    parser.add_argument("--g-th", type=float, default=0.3)
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
        rho_warn=args.rho_warn,
    )
    recovery_config = RecoveryConfig(
        rho_warn=args.rho_warn,
        g_th=args.g_th,
        latency_budget_s=args.latency_budget_s,
    )

    results = []
    for mode in args.modes:
        client = DeterministicRecoveryClient(plan_latency_s=args.plan_latency_s)
        result = _run_mode(
            scenario=scenario,
            env=env,
            pilot=pilot,
            ra=ra,
            mode=mode,
            client=client,
            recovery_config=recovery_config,
            eval_seeds=args.eval_seeds,
            max_steps=args.max_steps,
        )
        results.append(result)

    payload = {"scenario": scenario.name, "results": results}
    print(json.dumps(payload, indent=2))

    output = args.output or "/Volumes/Expansion/safedrones_marllib_vec/phase4_results.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2))
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
