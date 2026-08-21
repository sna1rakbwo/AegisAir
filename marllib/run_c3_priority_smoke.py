#!/usr/bin/env python3
"""Smoke gate for the C3 priority-constrained reassignment experiment.

``priority_reassign`` makes the rule planner's "nearest healthy drone" heuristic
diverge from the mission's semantic constraint: drone 1 is nearest to the orphan
but is "critical" and must keep its own mission, so the correct recovery
reassigns the normal-priority drone 2.

Primary metric: ``priority_violation`` (was the critical drone diverted?).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    LLMRecoveryClient,
    LLMRecoveryResult,
    MlxLmClient,
    RuleMissionPlanner,
)


def make_ra() -> RuntimeAssurance:
    return RuntimeAssurance(
        params=RuntimeAssuranceParams(tau_ctrl=0.2),
        v_max=1.5,
        sampled_data=True,
        gamma=0.1,
        tau_px4=0.2,
        tau_px4_min=0.2,
        tau_px4_max=0.2,
    )


class FixedDecisionClient(LLMRecoveryClient):
    name = "fixed-decision"

    def __init__(self, decision: dict) -> None:
        self.decision = decision

    def generate(self, context) -> LLMRecoveryResult:
        return LLMRecoveryResult(
            plan=None,
            raw=self.decision,
            latency_s=0.0,
            timeout=False,
            valid=True,
            errors=[],
            backend=self.name,
        )


def run(client, seed: int, real_time: bool) -> dict:
    spec = _scenario("priority_reassign")
    return run_sim_episode(
        spec=spec,
        env=MultiUAVEnv(spec["scenario"]),
        seed=seed,
        ra=make_ra(),
        mode="ASYNC",
        llm_client=client,
        llm_fallback=RuleMissionPlanner(),
        max_steps=200,
        real_time=real_time,
        execution_tau_s=0.2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 priority reassign smoke")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--real-qwen", action="store_true")
    parser.add_argument(
        "--qwen-model",
        default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench",
    )
    args = parser.parse_args()

    qwen = None
    if args.real_qwen:
        qwen = MlxLmClient(model_id=args.qwen_model, max_tokens=48, load=True)

    print("== R1 (rule / greedy nearest) ==")
    for seed in range(1, args.seeds + 1):
        r = run(RuleMissionPlanner(), seed, real_time=False)
        print(
            f"seed={seed} priority_violation={r['priority_violation']} "
            f"collision={r['collision']} completed={r['completed']}",
            flush=True,
        )

    print("== R2 (mock correct: REASSIGN agent=2) ==")
    for seed in range(1, args.seeds + 1):
        r = run(FixedDecisionClient({"action": "REASSIGN", "agent": 2}), seed, real_time=False)
        print(
            f"seed={seed} priority_violation={r['priority_violation']} "
            f"collision={r['collision']} completed={r['completed']}",
            flush=True,
        )

    if qwen is not None:
        print("== R2 (real Qwen) ==")
        for seed in range(1, args.seeds + 1):
            r = run(qwen, seed, real_time=True)
            counters = r.get("counters") or {}
            print(
                f"seed={seed} priority_violation={r['priority_violation']} "
                f"collision={r['collision']} completed={r['completed']} "
                f"llm_committed={counters.get('llm_plans_committed', 0)} "
                f"fallback={counters.get('fallback_plans_committed', 0)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
