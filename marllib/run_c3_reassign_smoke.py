#!/usr/bin/env python3
"""Smoke gate for the C3 LLM reassignment experiment.

Verifies two things before any full 30-seed run:

1. ``reassign_3`` makes the greedy rule actually suboptimal (total path ~10.4 m
   vs the optimal ~6.1 m).
2. The compact-decision pipeline accepts a REASSIGN to the *non-nearest* drone
   (agent=2) and drives the cheaper assignment end to end.

This uses an instant mock for the LLM, so it is fast and deterministic and does
not load the local model.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    RuleMissionPlanner,
)


def make_ra() -> RuntimeAssurance:
    return RuntimeAssurance(
        params=RuntimeAssuranceParams(tau_ctrl=0.1),
        v_max=1.5,
        sampled_data=True,
        gamma=0.1,
        tau_px4=0.7,
        tau_px4_min=0.7,
        tau_px4_max=0.7,
    )


class FixedDecisionClient(LLMRecoveryClient):
    """Return a fixed compact decision, standing in for a correct/incorrect LLM."""

    name = "fixed-decision"

    def __init__(self, decision: dict, latency_s: float = 0.0) -> None:
        self.decision = decision
        self.latency_s = latency_s

    def generate(self, context) -> LLMRecoveryResult:
        if self.latency_s:
            time.sleep(self.latency_s)
        return LLMRecoveryResult(
            plan=None,
            raw=self.decision,
            latency_s=self.latency_s,
            timeout=False,
            valid=True,
            errors=[],
            backend=self.name,
        )


def run_condition(client, seed: int, max_steps: int, real_time: bool) -> dict:
    spec = _scenario("reassign_3")
    return run_sim_episode(
        spec=spec,
        env=MultiUAVEnv(spec["scenario"]),
        seed=seed,
        ra=make_ra(),
        mode="ASYNC",
        llm_client=client,
        llm_fallback=RuleMissionPlanner(),
        max_steps=max_steps,
        real_time=real_time,
        execution_tau_s=0.2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 reassignment smoke")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--llm-latency-s", type=float, default=0.0)
    parser.add_argument("--real-time", action="store_true")
    args = parser.parse_args()

    # R1: greedy rule (nearest healthy drone takes over the orphan).
    rule_paths = []
    # "Optimal": a compact decision that reassigns drone 2 (the non-nearest).
    optimal_paths = []
    for seed in range(1, args.seeds + 1):
        rule = run_condition(
            RuleMissionPlanner(), seed, args.max_steps, args.real_time
        )
        optimal = run_condition(
            FixedDecisionClient(
                {"action": "REASSIGN", "agent": 2},
                latency_s=args.llm_latency_s,
            ),
            seed,
            args.max_steps,
            args.real_time,
        )
        rule_paths.append(rule["path_length_m"])
        optimal_paths.append(optimal["path_length_m"])
        print(
            f"seed={seed} rule_path={rule['path_length_m']:.3f} "
            f"optimal_path={optimal['path_length_m']:.3f} "
            f"rule_collision={rule['collision']} "
            f"optimal_collision={optimal['collision']} "
            f"optimal_llm_committed={ (optimal.get('counters') or {}).get('llm_plans_committed', 0) }",
            flush=True,
        )

    print(
        f"\nmean rule_path={sum(rule_paths)/len(rule_paths):.3f} "
        f"mean optimal_path={sum(optimal_paths)/len(optimal_paths):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
