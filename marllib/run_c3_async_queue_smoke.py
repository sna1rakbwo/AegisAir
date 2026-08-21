#!/usr/bin/env python3
"""Minimal smoke for the C3 "async queue" (immediate fallback) hypothesis.

Question: does committing the deterministic rule fallback immediately on a
recovery trigger (instead of waiting for the slow local LLM) preserve the
safety margin in ``priority_conflict``?

This is an exploratory smoke, not a frozen experiment.  It uses a 2.8 s
deterministic stand-in for the local Qwen so it can be replayed without
loading the model, and it keeps the frozen C3 v1 result untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import DeterministicRecoveryClient, RuleMissionPlanner


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


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 async-queue smoke")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--llm-latency-s", type=float, default=2.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = _scenario("priority_conflict")
    rows = []
    for immediate in (False, True):
        for seed in range(1, args.seeds + 1):
            run = run_sim_episode(
                spec=spec,
                env=MultiUAVEnv(spec["scenario"]),
                seed=seed,
                ra=make_ra(),
                mode="ASYNC",
                llm_client=DeterministicRecoveryClient(
                    plan_latency_s=args.llm_latency_s
                ),
                llm_fallback=RuleMissionPlanner(),
                max_steps=args.max_steps,
                real_time=True,
                execution_tau_s=0.2,
                immediate_fallback=immediate,
            )
            counters = run.get("counters") or {}
            rows.append(
                {
                    "immediate_fallback": immediate,
                    "seed": seed,
                    "collision": run["collision"],
                    "completed": run["completed"],
                    "min_rho": run["min_rho"],
                    "recovery_step": run["recovery_step"],
                    "recovery_time_s": run["recovery_time_s"],
                    "high_reached_step": run.get("high_reached_step"),
                    "fallback_plans_committed": counters.get(
                        "fallback_plans_committed", 0
                    ),
                    "llm_stale_invalid": counters.get("llm_stale_invalid", 0),
                    "llm_timeouts": counters.get("llm_timeouts", 0),
                }
            )

    payload = {
        "protocol_id": "aegisair-c3-async-queue-smoke",
        "config": {
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "llm_latency_s": args.llm_latency_s,
            "execution_tau_s": 0.2,
        },
        "episodes": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            parser.error(f"refusing to overwrite {args.out}")
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for immediate in (False, True):
        group = [r for r in rows if r["immediate_fallback"] == immediate]
        rhos = [r["min_rho"] for r in group if r["min_rho"] is not None]
        recovery_steps = [r["recovery_step"] for r in group if r["recovery_step"] is not None]
        print(
            f"immediate_fallback={immediate}: "
            f"min_rho(min)={min(rhos) if rhos else None}, "
            f"recovery_step={recovery_steps}, "
            f"collisions={sum(r['collision'] for r in group)}/{len(group)}, "
            f"completed={sum(r['completed'] for r in group)}/{len(group)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
