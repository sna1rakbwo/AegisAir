#!/usr/bin/env python3
"""在 PX4/Gazebo 上运行 C1 强 CBF 基线对比。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.phase5_runner import OBSERVATION_SHARED_CURRENT, run_mqtt_loop
from swarm.ra.margins import RuntimeAssuranceParams


PROTOCOL_IDS = {
    "aegisair-c1-sota-cbf-px4-smoke-v1",
    "aegisair-c1-sota-cbf-px4-validation-v1",
    "aegisair-c1-hocbf-v3-px4-smoke-v1",
    "aegisair-c1-hocbf-v3-px4-validation-v1",
    "aegisair-c1-sota-only-px4-smoke-v1",
    "aegisair-c1-hocbf-v3-vs-pb-px4-validation-v1",
    "aegisair-c1-hocbf-v4-px4-calibration-v1",
    "aegisair-c1-hocbf-v4-px4-calibration-v2",
    "aegisair-c1-hocbf-v4-px4-smoke-v1",
    "aegisair-c1-hocbf-v4-px4-validation-v1",
    "aegisair-c1-hocbf-v4-px4-postfreeze-ood-v1",
    "aegisair-c1-hocbf-v4-certificate-calibration-v1",
    "aegisair-c2-v4-execution-bridge-v1",
    "aegisair-c1-hocbf-v4-ablation-sealed-v1",
    "aegisair-c1-pcbf-huang-ecc2025-calibration-v1",
    "aegisair-c1-pcbf-huang-ecc2025-development-v2",
    "aegisair-c1-pcbf-huang-ecc2025-development-v3",
    "aegisair-c1-pcbf-huang-ecc2025-calibration-v3",
    "aegisair-c1-external-pcbf-sealed-v1",
    "aegisair-c1-external-pcbf-buffer-matched-fresh-v1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method_kwargs(method: str, config: dict, tau_command: float) -> dict:
    common = dict(
        a_max=2.0,
        kv=2.0,
        observation_mode=OBSERVATION_SHARED_CURRENT,
        ra_params=RuntimeAssuranceParams(degradation_dt=0.05),
        constraint_boundary=config["constraint_boundary"],
    )
    if method == "VELOCITY_CBF":
        return {**common, "velocity_command_mode": "safe_action"}
    feedforward = dict(
        velocity_command_mode="feedforward_tau",
        tau_command_s=tau_command,
        ra_command_feedforward_tau_s=tau_command,
    )
    if method == "ZOCBF":
        tau = float(config["barrier_model_tau_s"])
        return {
            **common,
            **feedforward,
            "sampled_data": True,
            "sampled_data_method": "zocbf",
            "tau_px4": tau,
            "tau_px4_min": tau,
            "tau_px4_max": tau,
            "gamma": float(config["gamma"]),
            "zocbf_delta": float(config["delta_m"]),
            "sampled_data_boundary_buffer_m": float(config.get("boundary_buffer_m", 0.0)),
            "sampled_data_infeasible_fallback": str(config.get("infeasible_fallback", "velocity_cancel")),
        }
    if method == "PB_CBF":
        return {
            **common,
            **feedforward,
            "sampled_data": True,
            "sampled_data_method": "pb_cbf",
            "tau_px4": tau_command,
            "tau_px4_min": tau_command,
            "tau_px4_max": tau_command,
            "pb_alpha": float(config["alpha"]),
            "pb_braking_accel": float(config["braking_accel_mps2"]),
            "sampled_data_boundary_buffer_m": float(config.get("boundary_buffer_m", 0.0)),
            "sampled_data_infeasible_fallback": str(config.get("infeasible_fallback", "velocity_cancel")),
        }
    if method == "PCBF_HUANG_ECC2025":
        return {
            **common,
            **feedforward,
            "sampled_data": True,
            "sampled_data_method": "pcbf",
            "tau_px4": tau_command,
            "tau_px4_min": tau_command,
            "tau_px4_max": tau_command,
            "sampled_data_boundary_buffer_m": float(config.get("boundary_buffer_m", 0.0)),
            "pcbf_horizon": int(config["horizon"]),
            "pcbf_terminal_buffer_m": float(config["terminal_buffer_m"]),
            "pcbf_terminal_velocity_tolerance_mps": float(config.get("terminal_velocity_tolerance_mps", 0.0)),
            "pcbf_position_bound_m": float(config.get("position_bound_m", 20.0)),
            "pcbf_velocity_bound_mps": float(config.get("velocity_bound_mps", 5.0)),
            "pcbf_max_iterations": int(config.get("max_iterations", 300)),
            "pcbf_multistart_count": int(config.get("multistart_count", 3)),
            "pcbf_tolerance": float(config.get("tolerance", 1e-7)),
            "pcbf_acceptable_tolerance": float(config.get("acceptable_tolerance", 1e-5)),
            "pcbf_lexicographic_tolerance": float(config.get("lexicographic_tolerance", 1e-7)),
        }
    if method == "AEGIS_HOCBF_V2":
        return {
            **common,
            **feedforward,
            "use_hocbf": True,
            "hocbf_k1": float(config["k1"]),
            "hocbf_k2": float(config["k2"]),
            "hocbf_boundary_guard": float(config["boundary_guard"]),
            "hocbf_boundary_buffer_m": float(config["boundary_buffer_m"]),
            "tau_px4": float(config["barrier_model_tau_s"]),
        }
    if method == "AEGIS_HOCBF_V3":
        return {
            **common,
            **feedforward,
            "use_hocbf": True,
            "hocbf_k1": float(config["k1"]),
            "hocbf_k2": float(config["k2"]),
            "hocbf_boundary_guard": float(config["boundary_guard"]),
            "hocbf_boundary_buffer_m": float(config["boundary_buffer_m"]),
            "hocbf_infeasible_fallback": "max_brake",
            "tau_px4": float(config["barrier_model_tau_s"]),
        }
    if method.startswith("AEGIS_HOCBF_V4"):
        return {
            **common,
            **feedforward,
            "use_hocbf": True,
            "hocbf_k1": float(config["k1"]),
            "hocbf_k2": float(config["k2"]),
            "hocbf_boundary_guard": float(config["boundary_guard"]),
            "hocbf_boundary_buffer_m": float(config["boundary_buffer_m"]),
            "hocbf_infeasible_fallback": "max_brake",
            "hocbf_pb_recovery": True,
            "hocbf_predictive_recovery": bool(
                config.get("predictive_recovery", True)
            ),
            "hocbf_prediction_execution_fraction": float(
                config.get("prediction_execution_fraction", 1.0)
            ),
            "hocbf_prediction_steps": int(config.get("prediction_steps", 1)),
            "hocbf_recovery_reserve_threshold": (
                float(config["recovery_reserve_threshold"])
                if config.get("recovery_reserve_threshold") is not None
                else None
            ),
            "hocbf_recovery_alpha": float(config["recovery_alpha"]),
            "hocbf_recovery_braking_accel": float(
                config["recovery_braking_accel_mps2"]
            ),
            "hocbf_recovery_boundary_buffer_m": float(
                config["recovery_boundary_buffer_m"]
            ),
            "hocbf_recovery_clear_steps": int(config["recovery_clear_steps"]),
            "tau_px4": float(config["barrier_model_tau_s"]),
        }
    raise ValueError(f"未知方法：{method}")


def _trajectory_metrics(path: Path) -> tuple[float, int]:
    effort = []
    infeasible_steps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("input_freshness", {}).get("fresh", False):
            continue
        step_infeasible = False
        for drone in row["drones"].values():
            published_velocity = drone.get("published_velocity")
            if published_velocity is None:
                continue
            effort.append(
                float(
                    np.linalg.norm(
                        np.asarray(published_velocity[:2])
                        - np.asarray(drone["v_nom"][:2])
                    )
                )
            )
            if drone["feasible"] is False:
                step_infeasible = True
        infeasible_steps += int(step_infeasible)
    return float(np.mean(effort)) if effort else float("nan"), infeasible_steps


def _trial_geometry(manifest: dict, trial: dict) -> tuple[str, dict, dict]:
    """Return a predeclared scenario geometry without changing controller knobs."""
    scenario_id = str(trial.get("scenario_id", "nominal"))
    scenario = manifest.get("scenarios", {}).get(scenario_id, {})
    base_goals = scenario.get("base_goals", manifest["base_goals"])
    reset_starts = scenario.get("reset_starts", manifest["reset_starts"])
    return (
        scenario_id,
        {int(i): tuple(v) for i, v in base_goals.items()},
        {int(i): tuple(v) for i, v in reset_starts.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trial-id", action="append")
    parser.add_argument(
        "--method",
        action="append",
        choices=(
            "VELOCITY_CBF",
            "ZOCBF",
            "PB_CBF",
            "PCBF_HUANG_ECC2025",
            "AEGIS_HOCBF_V2",
            "AEGIS_HOCBF_V3",
            "AEGIS_HOCBF_V4",
            "AEGIS_HOCBF_V4_REACTIVE",
            "AEGIS_HOCBF_V4_RESERVE_ONLY",
            "AEGIS_HOCBF_V4_BAD_TAU",
        ),
        help="只运行指定方法；正式 PX4 编排用它保证每个方法使用 fresh SITL。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") not in PROTOCOL_IDS:
        parser.error("不支持的 protocol_id")
    if args.out_dir.exists():
        parser.error(f"拒绝覆盖已有目录：{args.out_dir}")
    args.out_dir.mkdir(parents=True)
    wanted = set(args.trial_id or [])
    wanted_methods = set(args.method or [])
    trials = [t for t in manifest["trials"] if not wanted or t["trial_id"] in wanted]
    rows = []
    selected = [
        (trial, order_index, method)
        for trial in trials
        for order_index, method in enumerate(trial["condition_order"])
        if not wanted_methods or method in wanted_methods
    ]
    if not selected:
        parser.error("trial/method 过滤后没有待运行条件")
    total = len(selected)
    episode = 0
    for trial, order_index, method in selected:
        scenario_id, base_goals, reset_starts = _trial_geometry(manifest, trial)
        jitter = manifest.get("command_hold_jitter", {}).get(scenario_id, {})
        trajectory = args.out_dir / f"{trial['trial_id']}_{order_index:02d}_{method}.jsonl"
        run = run_mqtt_loop(
            drone_ids=[int(i) for i in manifest["drone_ids"]],
            base_goals=base_goals,
            mode="CBF_ONLY",
            llm_client=None,
            llm_fallback=None,
            host=args.host,
            port=args.port,
            max_steps=int(manifest["max_steps"]),
            trajectory=trajectory,
            reset_starts=reset_starts,
            rate_hz=float(manifest["rate_hz"]),
            goal_epsilon=float(manifest["goal_epsilon"]),
            command_hold_jitter_probability=float(jitter.get("probability", 0.0)),
            command_hold_jitter_seed=(
                int(jitter.get("seed_offset", 0)) + int(trial["seed"])
            ),
            land_at_end=episode == total - 1,
            **_method_kwargs(
                method,
                manifest["methods"][method],
                float(manifest["execution_tau_s"]),
            ),
        )
        effort, infeasible = _trajectory_metrics(trajectory)
        complete = all(
            np.linalg.norm(
                np.asarray(run["final_positions"][i][:2])
                - np.asarray(base_goals[i][:2])
            )
            < float(manifest["goal_epsilon"])
            for i in [int(v) for v in manifest["drone_ids"]]
        )
        row = {
            "trial_id": trial["trial_id"],
            "seed": trial["seed"],
            "method": method,
            "scenario_id": scenario_id,
            "order_index": order_index,
            "min_rho": run["min_rho"],
            "min_distance_m": run["min_distance_m"],
            "collision": run["collision"],
            "mission_complete": complete,
            "cbf_events": run["cbf_events"],
            "qp_infeasible_steps": infeasible,
            "hocbf_primary_infeasible_steps": run.get(
                "hocbf_primary_infeasible_steps", 0
            ),
            "hocbf_predictive_infeasible_steps": run.get(
                "hocbf_predictive_infeasible_steps", 0
            ),
            "hocbf_recovery_infeasible_steps": run.get(
                "hocbf_recovery_infeasible_steps", 0
            ),
            "hocbf_recovery_steps": run.get("hocbf_recovery_steps", 0),
            "hocbf_recovery_entries": run.get("hocbf_recovery_entries", 0),
            "mean_control_effort": effort,
            "path_length_m": run["path_length_m"],
            "command_hold_jitter_probability": run["command_hold_jitter_probability"],
            "command_hold_jitter_count": run["command_hold_jitter_count"],
            "ra_solve_latency_summary_ms": run["ra_solve_latency_summary_ms"],
            "safety_bypass_count": run["safety_bypass_count"],
            "infrastructure_valid": run["infrastructure_valid"],
            "infrastructure_invalid_reasons": run[
                "infrastructure_invalid_reasons"
            ],
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
            "trajectory": trajectory.name,
            "trajectory_sha256": _sha256(trajectory),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        episode += 1
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol_id": manifest["protocol_id"],
                "manifest_sha256": _sha256(args.manifest),
                "trials": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
