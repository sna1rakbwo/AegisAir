#!/usr/bin/env python3
"""Run frozen C3 closed-loop fail-closed recovery trials on PX4/Gazebo.

This is the closed-loop counterpart of ``run_c3_minimum_recovery.py``.  The full
PX4/Gazebo/MQTT stack must already be running (broker, GCS heartbeat, PX4 SITL,
and the ``aegisair-adapters`` bridge).  Each physical trial should use a fresh
SITL instance to avoid EKF/offboard state leaking across trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.phase5_runner import run_mqtt_loop
from swarm.recovery import (
    RuleMissionPlanner,
)


PROTOCOL_IDS = {
    "aegisair-c3-closed-loop-v1",
    "aegisair-c3-closed-loop-v2-current-head",
    "aegisair-c3-closed-loop-v3-hocbf-v4-smoke-v1",
    "aegisair-c3-closed-loop-v3-hocbf-v4-validation-v1",
}
CONDITIONS = ("R0", "R1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _recovery_clients(condition: str) -> tuple[str, Any, Any]:
    """Return ``(mode, llm_client, llm_fallback)`` for a frozen condition."""
    if condition == "R0":
        return "CBF_ONLY", None, None
    if condition == "R1":
        return "ASYNC", RuleMissionPlanner(), None
    raise ValueError(f"unknown C3 condition: {condition}")


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 closed-loop PX4/Gazebo runner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--velocity-command-mode",
        choices=["safe_action", "feedforward_tau", "nominal"],
        default="safe_action",
    )
    parser.add_argument("--tau-command-s", type=float, default=0.7)
    parser.add_argument(
        "--trial-id",
        action="append",
        help="Run only named physical trial(s); use fresh SITL per invocation.",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    if manifest.get("protocol_id") not in PROTOCOL_IDS:
        parser.error(f"manifest protocol_id must be one of {sorted(PROTOCOL_IDS)}")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite existing output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    drone_ids = [int(v) for v in manifest["drone_ids"]]
    base_goals = {
        int(k): tuple(float(x) for x in v) for k, v in manifest["base_goals"].items()
    }
    reset_starts = {
        int(k): tuple(float(x) for x in v) for k, v in manifest["reset_starts"].items()
    }
    mission_change = manifest.get("mission_change")
    change_step = manifest.get("change_step")
    failed_drone = manifest.get("failed_drone")
    critical_goal = (
        tuple(float(x) for x in manifest["critical_goal"])
        if manifest.get("critical_goal") is not None
        else None
    )
    blocked_zone = (
        tuple(float(x) for x in manifest["blocked_zone"])
        if manifest.get("blocked_zone") is not None
        else None
    )
    goal_epsilon = float(manifest.get("goal_epsilon", 0.5))
    rate_hz = float(manifest["rate_hz"])
    max_steps = int(manifest["max_steps"])
    ra_config = manifest.get("ra_config")
    if ra_config is not None and manifest["protocol_id"] not in {
        "aegisair-c3-closed-loop-v3-hocbf-v4-smoke-v1",
        "aegisair-c3-closed-loop-v3-hocbf-v4-validation-v1",
    }:
        parser.error("仅 C3-v4 protocol 可以提供 ra_config")

    def c3_ra_kwargs() -> dict[str, Any]:
        if ra_config is None:
            return {
                "tau_ctrl": 0.2, "tau_px4": 0.2, "execution_model": "exact_zoh",
                "sampled_data": True, "gamma": 0.1,
            }
        return {
            "use_hocbf": True,
            "hocbf_k1": float(ra_config["k1"]),
            "hocbf_k2": float(ra_config["k2"]),
            "tau_ctrl": float(ra_config["tau_ctrl_s"]),
            "tau_px4": float(ra_config["execution_tau_s"]),
            "tau_px4_min": float(ra_config["execution_tau_s"]),
            "tau_px4_max": float(ra_config["execution_tau_s"]),
            "execution_model": "exact_zoh",
            "sampled_data": False,
            "hocbf_boundary_guard": float(ra_config["boundary_guard"]),
            "hocbf_boundary_buffer_m": float(ra_config["boundary_buffer_m"]),
            "hocbf_infeasible_fallback": "max_brake",
            "hocbf_pb_recovery": True,
            "hocbf_predictive_recovery": True,
            "hocbf_prediction_execution_fraction": float(ra_config["prediction_execution_fraction"]),
            "hocbf_prediction_steps": int(ra_config["prediction_steps"]),
            "hocbf_recovery_reserve_threshold": float(ra_config["recovery_reserve_threshold"]),
            "hocbf_recovery_alpha": float(ra_config["recovery_alpha"]),
            "hocbf_recovery_braking_accel": float(ra_config["recovery_braking_accel_mps2"]),
            "hocbf_recovery_boundary_buffer_m": float(ra_config["recovery_boundary_buffer_m"]),
            "hocbf_recovery_clear_steps": int(ra_config["recovery_clear_steps"]),
            "ra_command_feedforward_tau_s": float(ra_config["execution_tau_s"]),
        }

    all_trials = list(manifest["trials"])
    selected_ids = set(args.trial_id or [])
    selected_trials = (
        [t for t in all_trials if str(t["trial_id"]) in selected_ids]
        if selected_ids
        else all_trials
    )
    missing_ids = selected_ids.difference(
        str(t["trial_id"]) for t in selected_trials
    )
    if missing_ids:
        parser.error(f"unknown trial_id(s): {sorted(missing_ids)}")

    rows: list[dict[str, Any]] = []
    total_episodes = sum(len(t["condition_order"]) for t in selected_trials)
    episode_index = 0
    for trial in selected_trials:
        trial_id = str(trial["trial_id"])
        for order_index, condition in enumerate(trial["condition_order"]):
            if condition not in CONDITIONS:
                parser.error(f"unknown condition {condition} in trial {trial_id}")
            mode, llm_client, llm_fallback = _recovery_clients(condition)
            trajectory = args.out_dir / f"{trial_id}_{order_index:02d}_{condition}.jsonl"
            run = run_mqtt_loop(
                drone_ids=drone_ids,
                base_goals=base_goals,
                mode=mode,
                llm_client=llm_client,
                llm_fallback=llm_fallback,
                host=args.host,
                port=args.port,
                max_steps=max_steps,
                trajectory=trajectory,
                reset_starts=reset_starts,
                rate_hz=rate_hz,
                **c3_ra_kwargs(),
                mission_change=mission_change,
                change_step=change_step,
                failed_drone=failed_drone,
                blocked_zone=blocked_zone,
                critical_goal=critical_goal,
                goal_epsilon=goal_epsilon,
                velocity_command_mode=args.velocity_command_mode,
                tau_command_s=args.tau_command_s,
                land_at_end=(episode_index == total_episodes - 1),
            )
            rows.append(
                {
                    "trial_id": trial_id,
                    "seed": trial.get("seed"),
                    "condition": condition,
                    "order_index": order_index,
                    "min_rho": run["min_rho"],
                    "min_distance_m": run["min_distance_m"],
                    "cbf_events": run["cbf_events"],
                    "collision": run["collision"],
                    "critical_reached": run["critical_reached"],
                    "zone_crossed": run["zone_crossed"],
                    "recovery_step": run["recovery_step"],
                    "recovery_time_s": run["recovery_time_s"],
                    "path_length_m": run["path_length_m"],
                    "counters": run.get("counters"),
                    "trajectory": trajectory.name,
                    "trajectory_sha256": _sha256(trajectory),
                }
            )
            episode_index += 1
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    payload = {
        "protocol_id": manifest["protocol_id"],
        "phase": manifest["phase"],
        "scenario": manifest.get("scenario"),
        "selected_trial_ids": [str(t["trial_id"]) for t in selected_trials],
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "velocity_command_mode": args.velocity_command_mode,
        "tau_command_s": args.tau_command_s,
        "trials": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
