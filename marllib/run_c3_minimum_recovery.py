#!/usr/bin/env python3
"""Run the frozen minimum C3 recovery and fail-closed matrices."""

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
from swarm.interfaces import RecoveryPlan
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.recovery import (
    DeterministicRecoveryClient,
    LLMRecoveryClient,
    LLMRecoveryResult,
    MlxLmClient,
    RuleMissionPlanner,
)


PROTOCOL_ID = "aegisair-minimum-c3-v1"
SCENARIOS = ("drone_failure", "corridor_blocked", "priority_conflict")
CONDITIONS = ("R0", "R1", "R2")
FAULTS = ("timeout", "malformed", "schema", "semantic", "stale")


class FaultedClient(LLMRecoveryClient):
    name = "c3-faulted"

    def __init__(self, raw: dict[str, Any] | None) -> None:
        self.raw = raw

    def generate(self, context) -> LLMRecoveryResult:
        return LLMRecoveryResult(
            plan=None,
            raw=self.raw,
            latency_s=0.0,
            timeout=False,
            valid=self.raw is not None,
            errors=[],
            backend=self.name,
        )


class ExpiredPlanClient(LLMRecoveryClient):
    name = "c3-expired-plan"

    def generate(self, context) -> LLMRecoveryResult:
        plan = RuleMissionPlanner().generate(context).plan
        assert plan is not None
        expired = plan.model_copy(update={"timestamp_ms": 1})
        return LLMRecoveryResult(
            plan=None,
            raw=expired.model_dump(mode="json"),
            latency_s=0.0,
            timeout=False,
            valid=True,
            errors=[],
            backend=self.name,
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


def recovery_clients(condition: str, qwen: MlxLmClient | None) -> tuple[Any, Any]:
    if condition == "R0":
        return None, None
    if condition == "R1":
        return RuleMissionPlanner(), None
    if qwen is None:
        raise RuntimeError("R2 requires a loaded local Qwen client")
    return qwen, RuleMissionPlanner()


def row(scenario: str, condition: str, seed: int, run: dict[str, Any]) -> dict[str, Any]:
    counters = run.get("counters") or {}
    return {
        "protocol_id": PROTOCOL_ID,
        "scenario": scenario,
        "condition": condition,
        "seed": seed,
        "collision": run["collision"],
        "completed": run["completed"],
        "critical_reached": run["critical_reached"],
        "high_reached_step": run["high_reached_step"],
        "zone_crossed": run["zone_crossed"],
        "completion_steps": run["completion_steps"],
        "recovery_time_s": run["recovery_time_s"],
        "path_length_m": run["path_length_m"],
        "max_waiting_duration_s": run["max_waiting_duration_s"],
        "max_repeated_cbf_duration_s": run["max_repeated_cbf_duration_s"],
        "recovery_ra_veto_steps": run["recovery_ra_veto_steps"],
        "safety_bypass_count": run["safety_bypass_count"],
        "min_rho": run["min_rho"],
        "cbf_events": run["cbf_events"],
        "rejected_commands": run["rejected_commands"],
        "rejected_reasons": run["rejected_reasons"],
        **{key: counters.get(key, 0) for key in (
            "triggers", "plans_committed", "llm_plans_committed",
            "fallback_plans_committed", "mission_changes", "llm_timeouts",
            "llm_syntactic_invalid", "llm_schema_invalid", "llm_semantic_invalid",
            "llm_execution_invalid", "llm_stale_invalid",
        )},
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for scenario in SCENARIOS:
        for condition in CONDITIONS:
            group = [r for r in rows if r["scenario"] == scenario and r["condition"] == condition]
            if not group:
                continue
            recovery_times = [r["recovery_time_s"] for r in group if r["recovery_time_s"] is not None]
            high_steps = [r["high_reached_step"] for r in group if r["high_reached_step"] is not None]
            margins = [r["min_rho"] for r in group if r["min_rho"] is not None]
            summary.append({
                "scenario": scenario,
                "condition": condition,
                "episodes": len(group),
                "collision_rate": float(np.mean([r["collision"] for r in group])),
                "completion_rate": float(np.mean([r["completed"] for r in group])),
                "critical_task_completion_rate": float(np.mean([r["critical_reached"] for r in group])),
                "zone_cross_rate": float(np.mean([r["zone_crossed"] for r in group])),
                "high_priority_completion_rate": float(np.mean([r["high_reached_step"] is not None for r in group])),
                "mean_high_priority_reached_step": float(np.mean(high_steps)) if high_steps else None,
                "mean_recovery_time_s": float(np.mean(recovery_times)) if recovery_times else None,
                "mean_completion_steps": float(np.mean([r["completion_steps"] for r in group])),
                "mean_path_length_m": float(np.mean([r["path_length_m"] for r in group])),
                "max_waiting_duration_s": float(max(r["max_waiting_duration_s"] for r in group)),
                "max_repeated_cbf_duration_s": float(max(r["max_repeated_cbf_duration_s"] for r in group)),
                "min_rho": float(min(margins)) if margins else None,
                "recovery_ra_veto_steps": int(sum(r["recovery_ra_veto_steps"] for r in group)),
                "safety_bypass_count": int(sum(r["safety_bypass_count"] for r in group)),
                "fallback_plans_committed": int(sum(r["fallback_plans_committed"] for r in group)),
            })
    return summary


def fault_client(fault: str) -> tuple[LLMRecoveryClient, float | None]:
    if fault == "timeout":
        return DeterministicRecoveryClient(plan_latency_s=0.3), 0.05
    if fault == "malformed":
        return FaultedClient(None), None
    if fault == "schema":
        return FaultedClient({"action": "HOVER"}), None
    if fault == "semantic":
        return FaultedClient({"action": "REROUTE", "agent": 99}), None
    if fault == "stale":
        return ExpiredPlanClient(), None
    raise ValueError(f"unknown fault {fault}")


def fault_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for fault in FAULTS:
        group = [r for r in rows if r["fault"] == fault]
        grouped[fault] = {
            "episodes": len(group),
            "collisions": int(sum(r["collision"] for r in group)),
            "fallback_plans_committed": int(sum(r["fallback_plans_committed"] for r in group)),
            "validator_rejections": int(sum(
                r["llm_syntactic_invalid"] + r["llm_schema_invalid"]
                + r["llm_semantic_invalid"] + r["llm_execution_invalid"]
                + r["llm_stale_invalid"] for r in group
            )),
            "safety_bypass_count": int(sum(r["safety_bypass_count"] for r in group)),
        }
    checks = {
        "all_faults_rejected_or_timed_out": all(
            grouped[fault]["fallback_plans_committed"] == grouped[fault]["episodes"]
            for fault in FAULTS
        ),
        "safety_bypass_zero": all(grouped[fault]["safety_bypass_count"] == 0 for fault in FAULTS),
        "collision_zero": all(grouped[fault]["collisions"] == 0 for fault in FAULTS),
    }
    return {"by_fault": grouped, "checks": checks, "pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimum C3 recovery experiment")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fault-seeds", type=int, default=30)
    parser.add_argument("--qwen-model", default="/Users/lijiajun/.cache/aegisair-qwen3-4b-4bit-bench")
    parser.add_argument("--qwen-max-tokens", type=int, default=48)
    parser.add_argument("--skip-r2", action="store_true")
    parser.add_argument(
        "--conditions",
        default=None,
        help="comma-separated subset of R0,R1,R2; defaults to the full matrix",
    )
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or (not args.skip_faults and args.fault_seeds < 1):
        parser.error("seed counts must be positive")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    if args.conditions is not None:
        active_conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
        if not active_conditions or any(item not in CONDITIONS for item in active_conditions):
            parser.error("--conditions must be a non-empty subset of R0,R1,R2")
        if args.skip_r2 and "R2" in active_conditions:
            parser.error("--skip-r2 conflicts with --conditions containing R2")
    else:
        active_conditions = ("R0", "R1") if args.skip_r2 else CONDITIONS

    qwen = None if "R2" not in active_conditions else MlxLmClient(
        model_id=args.qwen_model, max_tokens=args.qwen_max_tokens, load=True
    )
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        spec = _scenario(scenario)
        for condition in active_conditions:
            client, fallback = recovery_clients(condition, qwen)
            for seed in range(1, args.seeds + 1):
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_ra(),
                    mode="CBF_ONLY" if condition == "R0" else "ASYNC",
                    llm_client=client,
                    llm_fallback=fallback,
                    max_steps=args.max_steps,
                    # The actual local LLM must be given wall-clock time to
                    # return; accelerated simulation would manufacture a
                    # timeout before its asynchronous future can be polled.
                    real_time=(condition == "R2"),
                    execution_tau_s=0.2,
                )
                rows.append(row(scenario, condition, seed, run))

    fault_rows: list[dict[str, Any]] = []
    if not args.skip_faults:
        spec = _scenario("priority_conflict")
        for fault in FAULTS:
            client, timeout = fault_client(fault)
            for seed in range(1, args.fault_seeds + 1):
                run = run_sim_episode(
                    spec=spec,
                    env=MultiUAVEnv(spec["scenario"]),
                    seed=seed,
                    ra=make_ra(),
                    mode="ASYNC",
                    llm_client=client,
                    llm_fallback=RuleMissionPlanner(),
                    max_steps=args.max_steps,
                    real_time=False,
                    execution_tau_s=0.2,
                    replan_timeout_s=timeout,
                )
                item = row("priority_conflict", "FAULT", seed, run)
                item["fault"] = fault
                fault_rows.append(item)

    result_fault_summary = fault_summary(fault_rows) if fault_rows else {"skipped": True, "pass": True}

    payload = {
        "protocol_id": PROTOCOL_ID,
        "config": {
            "seeds": args.seeds,
            "fault_seeds": args.fault_seeds,
            "max_steps": args.max_steps,
            "execution_tau_s": 0.2,
            "tau_ctrl_s": 0.1,
            "tau_px4_s": 0.7,
            "gamma": 0.1,
            "conditions": list(active_conditions),
        },
        "summary": summarize(rows),
        "episodes": rows,
        "fault_summary": result_fault_summary,
        "fault_episodes": fault_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "fault_summary": payload["fault_summary"]}, ensure_ascii=False, indent=2))
    return 0 if payload["fault_summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
