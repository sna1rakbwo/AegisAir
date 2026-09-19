#!/usr/bin/env python3
"""运行 failure-aware recoverability-admission 单个 PX4/Gazebo 条件。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.phase5_runner import run_mqtt_loop
from swarm.recovery import (
    RAOnlySafeHoldCoordinator,
    RecoverabilityAdmissionConfig,
    RecoverabilityAdmissionCoordinator,
    RuleMissionPlanner,
)


PROTOCOL_IDS = {
    "aegisair-c-recoverability-admission-calibration-v2",
    "aegisair-c-recoverability-admission-qualification-v2",
}
IMPLEMENTATION_VERSION = "dynamic_admission_v2"
CONDITIONS = {
    "IMMEDIATE_COMMIT_RA",
    "RECOVERABILITY_ADMISSION_RA",
    "RA_ONLY_HOLD",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ra_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_hocbf": True,
        "hocbf_k1": float(config["k1"]),
        "hocbf_k2": float(config["k2"]),
        "tau_ctrl": float(config["tau_ctrl_s"]),
        "tau_px4": float(config["execution_tau_s"]),
        "tau_px4_min": float(config["execution_tau_s"]),
        "tau_px4_max": float(config["execution_tau_s"]),
        "execution_model": "exact_zoh",
        "sampled_data": False,
        "hocbf_boundary_guard": float(config["boundary_guard"]),
        "hocbf_boundary_buffer_m": float(config["boundary_buffer_m"]),
        "hocbf_infeasible_fallback": "max_brake",
        "hocbf_pb_recovery": True,
        "hocbf_predictive_recovery": True,
        "hocbf_prediction_execution_fraction": float(
            config["prediction_execution_fraction"]
        ),
        "hocbf_prediction_steps": int(config["prediction_steps"]),
        "hocbf_recovery_reserve_threshold": float(
            config["recovery_reserve_threshold"]
        ),
        "hocbf_recovery_alpha": float(config["recovery_alpha"]),
        "hocbf_recovery_braking_accel": float(
            config["recovery_braking_accel_mps2"]
        ),
        "hocbf_recovery_boundary_buffer_m": float(
            config["recovery_boundary_buffer_m"]
        ),
        "hocbf_recovery_clear_steps": int(config["recovery_clear_steps"]),
        "ra_command_feedforward_tau_s": float(config["execution_tau_s"]),
    }


def _admission_config(
    config: dict[str, Any],
    *,
    rate_hz: float,
    execution_tau_s: float,
    command_feedforward_tau_s: float,
) -> RecoverabilityAdmissionConfig:
    return RecoverabilityAdmissionConfig(
        clearance_m=float(config["clearance_m"]),
        braking_accel_mps2=float(config["braking_accel_mps2"]),
        braking_accel_uncertainty_mps2=float(
            config.get("braking_accel_uncertainty_mps2", 0.25)
        ),
        ring_extra_m=tuple(float(value) for value in config["ring_extra_m"]),
        ring_samples=int(config["ring_samples"]),
        arena=tuple(float(value) for value in config["arena"]),
        waypoint_epsilon_m=float(config["waypoint_epsilon_m"]),
        max_route_length_m=float(config["max_route_length_m"]),
        dt_s=float(config.get("dt_s", 1.0 / rate_hz)),
        rollout_horizon_s=float(config.get("rollout_horizon_s", 16.0)),
        reaction_delay_s=float(config.get("reaction_delay_s", 1.0 / rate_hz)),
        goal_gain_s_inv=float(config.get("goal_gain_s_inv", 1.5)),
        velocity_gain_s_inv=float(config.get("velocity_gain_s_inv", 2.0)),
        healthy_velocity_limit_mps=float(
            config.get("healthy_velocity_limit_mps", 1.5)
        ),
        healthy_acceleration_limit_mps2=float(
            config.get("healthy_acceleration_limit_mps2", 2.0)
        ),
        execution_tau_s=float(config.get("execution_tau_s", execution_tau_s)),
        command_feedforward_tau_s=float(
            config.get(
                "command_feedforward_tau_s",
                command_feedforward_tau_s,
            )
        ),
        terminal_speed_mps=float(config.get("terminal_speed_mps", 0.15)),
        tracking_error_buffer_m=float(
            config.get("tracking_error_buffer_m", 0.20)
        ),
    )


def _trajectory_audit(
    path: Path,
    *,
    failed_drone: int,
    healthy_drone: int,
    change_step: int,
    critical_goal: tuple[float, float],
    goal_epsilon: float,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    after = [row for row in rows if int(row["step"]) >= change_step]
    failed_rows = [row["drones"][str(failed_drone)] for row in after]
    failed_commands = [row.get("published_velocity") for row in failed_rows]
    admission_rows = [
        row.get("recoverability_admission") for row in after
        if row.get("recoverability_admission") is not None
    ]
    hold_goals = [
        tuple(decision["hold_goal"])
        for decision in admission_rows
        if decision.get("hold_goal") is not None
        and decision.get("state") in {"rejected_hold", "ra_only_hold"}
    ]
    return {
        "steps": len(rows),
        "post_failure_steps": len(after),
        "ra_bypass_count": sum(
            int(drone.get("ra_bypass", True))
            for row in rows
            for drone in row["drones"].values()
        ),
        "failed_authority_revoked_all_steps": bool(failed_rows)
        and all(row.get("command_authority") == "failed_zero" for row in failed_rows),
        "failed_horizontal_command_zero_all_steps": bool(failed_rows)
        and all(
            command is not None
            and np.linalg.norm(np.asarray(command[:2], dtype=np.float64)) <= 1e-9
            for command in failed_commands
        ),
        "selected_qp_infeasible_steps": sum(
            any(drone.get("feasible") is False for drone in row["drones"].values())
            for row in rows
            if row.get("input_freshness", {}).get("fresh", False)
        ),
        "hold_goal_frozen": not hold_goals
        or all(goal == hold_goals[0] for goal in hold_goals),
        "healthy_post_failure_max_command_mps": max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(
                            row["drones"][str(healthy_drone)][
                                "published_velocity"
                            ][:2]
                        )
                    )
                )
                for row in after
                if row["drones"][str(healthy_drone)].get("published_velocity")
                is not None
            ),
            default=0.0,
        ),
        "post_failure_critical_reached_by_healthy": any(
            float(
                np.linalg.norm(
                    np.asarray(row["drones"][str(healthy_drone)]["pos"][:2])
                    - np.asarray(critical_goal)
                )
            ) < goal_epsilon
            for row in after
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") not in PROTOCOL_IDS:
        parser.error(f"protocol_id 必须属于 {sorted(PROTOCOL_IDS)}")
    if manifest.get("implementation_version") != IMPLEMENTATION_VERSION:
        parser.error(
            "implementation_version 必须为 "
            f"{IMPLEMENTATION_VERSION!r}，v1 manifest 仅保留为历史记录"
        )
    if args.out_dir.exists():
        parser.error(f"拒绝覆盖输出目录：{args.out_dir}")
    trial = next(
        (item for item in manifest["trials"] if item["trial_id"] == args.trial_id),
        None,
    )
    if trial is None:
        parser.error(f"未知 trial_id：{args.trial_id}")
    if args.condition not in trial["condition_order"]:
        parser.error("condition 不属于该 trial")
    geometry = manifest["geometries"][trial["geometry_id"]]
    starts = {int(key): tuple(value) for key, value in geometry["reset_starts"].items()}
    goals = {int(key): tuple(value) for key, value in geometry["base_goals"].items()}
    failed_drone = int(manifest["failed_drone"])
    healthy_drone = next(drone for drone in goals if drone != failed_drone)

    mode = "CBF_ONLY"
    client = None
    coordinator = None
    if args.condition == "IMMEDIATE_COMMIT_RA":
        mode = "ASYNC"
        client = RuleMissionPlanner()
    elif args.condition == "RECOVERABILITY_ADMISSION_RA":
        coordinator = RecoverabilityAdmissionCoordinator(
            _admission_config(
                manifest["recoverability_admission"],
                rate_hz=float(manifest["rate_hz"]),
                execution_tau_s=float(manifest["ra_config"]["execution_tau_s"]),
                command_feedforward_tau_s=float(manifest["tau_command_s"]),
            )
        )
    else:
        coordinator = RAOnlySafeHoldCoordinator()

    args.out_dir.mkdir(parents=True)
    trajectory = args.out_dir / "trajectory.jsonl"
    run = run_mqtt_loop(
        drone_ids=[int(value) for value in manifest["drone_ids"]],
        base_goals=goals,
        reset_starts=starts,
        mode=mode,
        llm_client=client,
        llm_fallback=None,
        host="127.0.0.1",
        port=1883,
        max_steps=int(manifest["max_steps"]),
        rate_hz=float(manifest["rate_hz"]),
        trajectory=trajectory,
        mission_change=manifest["mission_change"],
        change_step=int(manifest["change_step"]),
        failed_drone=failed_drone,
        critical_goal=tuple(geometry["critical_goal"]),
        goal_epsilon=float(manifest["goal_epsilon"]),
        velocity_command_mode=manifest["velocity_command_mode"],
        tau_command_s=float(manifest["tau_command_s"]),
        recoverability_admission_coordinator=coordinator,
        **_ra_kwargs(manifest["ra_config"]),
    )
    audit = _trajectory_audit(
        trajectory,
        failed_drone=failed_drone,
        healthy_drone=healthy_drone,
        change_step=int(manifest["change_step"]),
        critical_goal=tuple(geometry["critical_goal"]),
        goal_epsilon=float(manifest["goal_epsilon"]),
    )
    row = {
        "trial_id": trial["trial_id"],
        "geometry_id": trial["geometry_id"],
        "seed": trial["seed"],
        "condition": args.condition,
        "expected_admission": geometry["expected_admission"],
        "collision": run["collision"],
        "min_rho": run["min_rho"],
        "min_distance_m": run["min_distance_m"],
        "critical_reached": run["critical_reached"],
        "post_failure_critical_reached": audit[
            "post_failure_critical_reached_by_healthy"
        ],
        "recovery_step": run["recovery_step"],
        "path_length_m": run["path_length_m"],
        "counters": run.get("counters"),
        "recoverability_admission": run.get("recoverability_admission"),
        "safety_bypass_count": run["safety_bypass_count"],
        "infrastructure_valid": run["infrastructure_valid"],
        "infrastructure_invalid_reasons": run["infrastructure_invalid_reasons"],
        "freshness_gate": run["freshness_gate"],
        "published_command_mismatch_count": run[
            "published_command_mismatch_count"
        ],
        "published_command_constraint_unknown_count": run[
            "published_command_constraint_unknown_count"
        ],
        "published_command_constraint_failure_count": run[
            "published_command_constraint_failure_count"
        ],
        "ra_solve_latency_summary_ms": run["ra_solve_latency_summary_ms"],
        "trajectory_audit": audit,
        "trajectory": trajectory.name,
        "trajectory_sha256": _sha256(trajectory),
    }
    payload = {
        "protocol_id": manifest["protocol_id"],
        "phase": manifest["phase"],
        "manifest_sha256": _sha256(args.manifest),
        "trial": trial,
        "geometry": geometry,
        "trials": [row],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "manifest.json").write_bytes(args.manifest.read_bytes())
    (args.out_dir / "integrity.sha256").write_text(
        "".join(
            f"{_sha256(args.out_dir / name)}  {name}\n"
            for name in ("manifest.json", "trajectory.jsonl", "summary.json")
        ),
        encoding="utf-8",
    )
    (args.out_dir / "COMPLETE").touch()
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
