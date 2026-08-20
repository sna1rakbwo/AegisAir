#!/usr/bin/env python3
"""Paired policy-quality smoke for nominal MAPPO versus S6 Runtime Assurance.

``pi*q`` is operationalized as deterministic action-amplitude scaling of the
same checkpoint actor.  It is a controlled nominal-authority degradation, not
a claim that differently trained policies have identical failure modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from marllib.policies.mappo import MappoPilot
from marllib.run_baseline_smoke import ObserveOnlyRA, make_controller


QUALITY_SCALES = (0.2, 0.5, 0.8, 1.0)
SAFETY_CONDITIONS = ("S0", "S6")


class ScaledPilot:
    def __init__(self, pilot: MappoPilot, scale: float) -> None:
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        self.pilot = pilot
        self.scale = scale

    def actions(self, **kwargs):
        return {agent: np.asarray(action, dtype=np.float64) * self.scale
                for agent, action in self.pilot.actions(**kwargs).items()}


def controller(condition: str, qp_max_iters: int):
    if condition == "S0":
        return ObserveOnlyRA(make_controller("S1", qp_max_iters=qp_max_iters))
    if condition == "S6":
        return make_controller("S6", qp_max_iters=qp_max_iters)
    raise ValueError(f"unsupported safety condition: {condition}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for training_seed in sorted({row["training_seed"] for row in rows}):
        for scale in QUALITY_SCALES:
            for condition in SAFETY_CONDITIONS:
                group = [row for row in rows if row["training_seed"] == training_seed
                         and row["quality_scale"] == scale and row["condition"] == condition]
                if not group:
                    continue
                summary.append({
                    "training_seed": training_seed,
                    "quality_scale": scale,
                    "condition": condition,
                    "episodes": len(group),
                    "collision_rate": float(np.mean([row["collision"] for row in group])),
                    "completion_rate": float(np.mean([row["completed"] for row in group])),
                    "mean_min_rho": float(np.mean([row["min_rho"] for row in group])),
                    "min_rho": float(np.min([row["min_rho"] for row in group])),
                    "mean_cbf_events": float(np.mean([row["cbf_events"] for row in group])),
                })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="MAPPO quality-scale S0/S6 smoke")
    parser.add_argument("--scenario", default="multi_uav")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--training-seeds", default="1,2,3,4,5")
    parser.add_argument("--eval-seed-start", type=int, default=101)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--qp-max-iters", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = _scenario(args.scenario)
    scenario = spec["scenario"]
    training_seeds = [int(value) for value in args.training_seeds.split(",") if value]
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    import hashlib

    for training_seed in training_seeds:
        checkpoint = args.checkpoint_root / f"seed{training_seed}" / "checkpoints" / "final.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes[str(training_seed)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        pilot = MappoPilot(
            checkpoint,
            obs_dim=4 + 5 * scenario.max_neighbors,
            num_agents=scenario.num_agents,
            speed_limit=scenario.speed_limit,
            max_neighbors=scenario.max_neighbors,
        )
        for scale in QUALITY_SCALES:
            scaled_pilot = ScaledPilot(pilot, scale)
            for eval_seed in eval_seeds:
                for condition in SAFETY_CONDITIONS:
                    run = run_sim_episode(
                        spec=spec,
                        env=MultiUAVEnv(scenario),
                        seed=eval_seed,
                        ra=controller(condition, args.qp_max_iters),
                        mode="CBF_ONLY",
                        llm_client=None,
                        llm_fallback=None,
                        max_steps=args.max_steps,
                        real_time=False,
                        pilot=scaled_pilot,
                    )
                    rows.append({
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "quality_scale": scale,
                        "condition": condition,
                        **run,
                    })

    output = {
        "config": {
            "protocol": "aegisair-policy-quality-smoke-v1",
            "scenario": args.scenario,
            "checkpoint_root": str(args.checkpoint_root),
            "training_seeds": training_seeds,
            "eval_seeds": eval_seeds,
            "quality_scales": list(QUALITY_SCALES),
            "conditions": list(SAFETY_CONDITIONS),
            "max_steps": args.max_steps,
            "qp_max_iters": args.qp_max_iters,
        },
        "checkpoint_sha256": checkpoint_hashes,
        "summary": summarize(rows),
        "episodes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
