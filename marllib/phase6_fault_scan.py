#!/usr/bin/env python3
"""Deterministic Phase 6 mission-level fault scan (no PX4/Gazebo).

This is the least-cost gate that complements ``px4_adapter/p5``.  It freezes the
adapter-level fail-closed behavior separately; here we inject the mission-level
faults the adapter scan cannot see:

- perception noise (perturbs the RA-observed shared state)
- stale telemetry (grows the RA AoI boundary and trips the adapter staleness gate)
- command latency (trips the adapter command-TTL gate)
- LLM timeout (AsyncMissionReplanner falls back)
- invalid LLM command (validator rejects and falls back)

Claim boundary: this validates deterministic RA + adapter *decisions* and the
recovery-layer fallback path.  It does not close the loop through the adapter's
fail-closed vehicle action; that requires the live PX4 gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.envs.multi_uav import MultiUAVEnv
from marllib.phase5_runner import _scenario, run_sim_episode
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    DeterministicRecoveryClient,
    LLMRecoveryClient,
    LLMRecoveryResult,
    RuleMissionPlanner,
)


PROTOCOL_ID = "safedrones-aegisair-phase6-local-v1"
SEEDS = list(range(1, 11))
MAX_STEPS = 200
DT = 0.05


class _FaultedClient(LLMRecoveryClient):
    """Deterministic backend that returns a fixed, invalid compact decision."""

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


FAULTS: list[dict[str, Any]] = [
    {
        "id": "NONE",
        "scenario": "head_on",
        "mode": "CBF_ONLY",
        "fault": {},
    },
    {
        "id": "PERCEPTION_NOISE",
        "scenario": "head_on",
        "mode": "CBF_ONLY",
        "fault": {
            "perception_noise_pos_m": 0.2,
            "perception_noise_vel_mps": 0.1,
        },
    },
    {
        "id": "STALE_TELEMETRY",
        "scenario": "head_on",
        "mode": "CBF_ONLY",
        "fault": {"telemetry_stale_ms": 3000},
    },
    {
        "id": "COMMAND_LATENCY",
        "scenario": "head_on",
        "mode": "CBF_ONLY",
        "fault": {"command_latency_ms": 1100},
    },
    {
        "id": "LLM_TIMEOUT",
        "scenario": "priority_conflict",
        "mode": "ASYNC",
        "llm_kind": "timeout",
        "replan_timeout_s": 0.05,
        "fault": {},
    },
    {
        "id": "INVALID_LLM_COMMAND",
        "scenario": "priority_conflict",
        "mode": "ASYNC",
        "llm_kind": "invalid",
        "fault": {},
    },
]


def _make_ra() -> RuntimeAssurance:
    return RuntimeAssurance(
        params=RuntimeAssuranceParams(tau_ctrl=0.0),
        v_max=1.5,
        sampled_data=True,
        gamma=0.1,
    )


def _clients_for(fault: dict[str, Any]) -> tuple[Any, Any]:
    kind = fault.get("llm_kind")
    if kind == "timeout":
        return (
            DeterministicRecoveryClient(plan_latency_s=0.3),
            RuleMissionPlanner(),
        )
    if kind == "invalid":
        return _FaultedClient({"action": "HOVER"}), RuleMissionPlanner()
    return None, None


def _row(
    fault: dict[str, Any],
    seed: int,
    run: dict[str, Any],
) -> dict[str, Any]:
    counters = run.get("counters") or {}
    return {
        "protocol_id": PROTOCOL_ID,
        "fault_id": fault["id"],
        "seed": seed,
        "collision": run["collision"],
        "completed": run["completed"],
        "completion_steps": run["completion_steps"],
        "min_rho": run["min_rho"],
        "cbf_events": run["cbf_events"],
        "emitted_commands": run["emitted_commands"],
        "rejected_commands": run["rejected_commands"],
        "rejected_reasons": run["rejected_reasons"],
        "llm_timeouts": counters.get("llm_timeouts", 0),
        "llm_syntactic_invalid": counters.get("llm_syntactic_invalid", 0),
        "llm_schema_invalid": counters.get("llm_schema_invalid", 0),
        "llm_semantic_invalid": counters.get("llm_semantic_invalid", 0),
        "llm_execution_invalid": counters.get("llm_execution_invalid", 0),
        "fallback_plans_committed": counters.get("fallback_plans_committed", 0),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by(fault_id: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["fault_id"] == fault_id]

    faults: dict[str, dict[str, Any]] = {}
    for fault in FAULTS:
        sub = by(fault["id"])
        rho = [r["min_rho"] for r in sub if r["min_rho"] is not None]
        reasons: dict[str, int] = {}
        for r in sub:
            for reason, count in r["rejected_reasons"].items():
                reasons[reason] = reasons.get(reason, 0) + count
        faults[fault["id"]] = {
            "episodes": len(sub),
            "collisions": sum(1 for r in sub if r["collision"]),
            "completed": sum(1 for r in sub if r["completed"]),
            "min_rho": (
                {"min": round(min(rho), 6), "max": round(max(rho), 6)}
                if rho
                else None
            ),
            "rejected_reasons": reasons,
            "llm_timeouts": sum(r["llm_timeouts"] for r in sub),
            "llm_syntactic_invalid": sum(r["llm_syntactic_invalid"] for r in sub),
            "llm_schema_invalid": sum(r["llm_schema_invalid"] for r in sub),
            "llm_semantic_invalid": sum(r["llm_semantic_invalid"] for r in sub),
            "fallback_plans_committed": sum(
                r["fallback_plans_committed"] for r in sub
            ),
        }

    none = faults["NONE"]
    stale = faults["STALE_TELEMETRY"]
    latency = faults["COMMAND_LATENCY"]
    ltime = faults["LLM_TIMEOUT"]
    linv = faults["INVALID_LLM_COMMAND"]

    checks = {
        "no_collision": all(r["collision"] is False for r in rows),
        "baseline_min_rho_ge_0": (
            none["min_rho"] is not None and none["min_rho"]["min"] >= 0.0
        ),
        "stale_telemetry_fail_closed": stale["rejected_reasons"].get(
            "telemetry_stale", 0
        )
        > 0,
        "command_latency_fail_closed": latency["rejected_reasons"].get(
            "command_expired", 0
        )
        > 0,
        "llm_timeout_fallback": (
            ltime["llm_timeouts"] > 0 and ltime["fallback_plans_committed"] > 0
        ),
        "invalid_llm_fallback": (
            linv["llm_schema_invalid"] > 0
            and linv["fallback_plans_committed"] > 0
        ),
    }

    return {
        "protocol_id": PROTOCOL_ID,
        "episodes": len(rows),
        "checks": checks,
        "pass": all(checks.values()),
        "faults": faults,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    rows: list[dict[str, Any]] = []
    for fault in FAULTS:
        spec = _scenario(fault["scenario"])
        spec["scenario"] = replace(spec["scenario"], dt=DT)
        client, fallback = _clients_for(fault)
        for seed in SEEDS:
            env = MultiUAVEnv(spec["scenario"])
            run = run_sim_episode(
                spec=spec,
                env=env,
                seed=seed,
                ra=_make_ra(),
                mode=fault["mode"],
                llm_client=client,
                llm_fallback=fallback,
                max_steps=MAX_STEPS,
                real_time=False,
                fault=fault["fault"],
                replan_timeout_s=fault.get("replan_timeout_s"),
            )
            rows.append(_row(fault, seed, run))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows),
        encoding="utf-8",
    )
    summary = _aggregate(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
