#!/usr/bin/env python3
"""四机中等密度 C3-GroupSlot + RA calibration smoke。"""

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
from marllib.run_c1_sota_cbf_gazebo import _method_kwargs, _trajectory_metrics
from swarm.recovery import C3GroupSlotCoordinator, GroupSlotConfig


PROTOCOL_ID = "aegisair-c3-group-slot-4uav-calibration-smoke-v2"
QUALIFICATION_PROTOCOL_ID = "aegisair-c3-group-slot-4uav-qualification-v2"
SEALED_PROTOCOL_ID = "aegisair-c3-group-slot-4uav-sealed-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coordinator(
    manifest: dict[str, Any],
    drone_ids: list[int],
    goals: dict[int, tuple[float, float, float]],
) -> C3GroupSlotCoordinator:
    config = manifest["c3_group_slot"]
    return C3GroupSlotCoordinator(
        GroupSlotConfig(
            conflict_zone_id=str(config["conflict_zone_id"]),
            conflict_zone_center=tuple(config["conflict_zone_center"]),
            group_order=tuple(tuple(group) for group in config["group_order"]),
            position_epsilon_m=float(config["position_epsilon_m"]),
            speed_epsilon_mps=float(config["speed_epsilon_mps"]),
            settle_steps=int(config["settle_steps"]),
            transit_scale=float(config["transit_scale"]),
        ),
        drone_ids,
        goals,
    )


def _audit(path: Path, expected_groups: list[list[int]]) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    decisions = [row.get("coordination_admission") for row in rows]
    decisions = [decision for decision in decisions if decision is not None]
    modes = {decision["coordination_mode"] for decision in decisions}
    frozen_versions: dict[str, set[tuple[float, float, float]]] = {}
    authorizations = []
    for decision in decisions:
        for drone, goal in decision["frozen_hold_goals"].items():
            frozen_versions.setdefault(str(drone), set()).add(tuple(goal))
        authorized = decision["authorized_drone_ids"]
        if authorized:
            authorizations.append(list(authorized))
    expected = [list(group) for group in expected_groups]
    return {
        "steps": len(rows),
        "selected_qp_infeasible_steps": sum(
            any(drone.get("feasible") is False for drone in row["drones"].values())
            for row in rows
            if row.get("input_freshness", {}).get("fresh", False)
        ),
        "ra_bypass_count": sum(
            int(drone.get("ra_bypass", True))
            for row in rows
            for drone in row["drones"].values()
        ),
        "modes_seen": sorted(modes),
        "required_modes_seen": all(
            mode in modes for mode in ["admission_pending", "hold", "pass", "release"]
        ),
        "frozen_hold_goals_stable": bool(frozen_versions)
        and all(len(values) == 1 for values in frozen_versions.values()),
        "authorized_groups_valid": all(group in expected for group in authorizations),
        "authorized_groups_seen": [group for group in expected if group in authorizations],
        "ra_vetoed_steps": sum(
            int(decision["ra_vetoed"]) for decision in decisions
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol_id = manifest.get("protocol_id")
    if protocol_id not in (PROTOCOL_ID, QUALIFICATION_PROTOCOL_ID, SEALED_PROTOCOL_ID):
        parser.error("protocol_id 不是冻结的 GroupSlot calibration/qualification")
    if protocol_id in (QUALIFICATION_PROTOCOL_ID, SEALED_PROTOCOL_ID):
        seed_key = "qualification_seeds" if protocol_id == QUALIFICATION_PROTOCOL_ID else "sealed_seeds"
        if args.seed not in manifest.get(seed_key, []):
            parser.error("seed 未冻结在当前 manifest 中")
    elif args.seed is not None:
        parser.error("calibration 不接受 qualification seed")
    if args.output.exists():
        parser.error(f"拒绝覆盖已有输出目录：{args.output}")
    args.output.mkdir(parents=True)
    drone_ids = [int(value) for value in manifest["drone_ids"]]
    starts = {int(key): tuple(value) for key, value in manifest["reset_starts"].items()}
    goals = {int(key): tuple(value) for key, value in manifest["base_goals"].items()}
    coordinator = _coordinator(manifest, drone_ids, goals)
    trajectory = args.output / "M4_C3_GROUP_SLOT.jsonl"
    run = run_mqtt_loop(
        drone_ids=drone_ids,
        base_goals=goals,
        reset_starts=starts,
        mode="CBF_ONLY",
        llm_client=None,
        llm_fallback=None,
        host="127.0.0.1",
        port=1883,
        max_steps=int(manifest["max_steps"]),
        rate_hz=float(manifest["rate_hz"]),
        goal_epsilon=float(manifest["goal_epsilon"]),
        trajectory=trajectory,
        group_slot_coordinator=coordinator,
        land_at_end=True,
        **_method_kwargs(
            manifest["method"], manifest["method_config"], float(manifest["execution_tau_s"])
        ),
    )
    effort, _ = _trajectory_metrics(trajectory)
    mission_complete = all(
        np.linalg.norm(
            np.asarray(run["final_positions"][drone][:2]) - np.asarray(goals[drone][:2])
        ) < float(manifest["goal_epsilon"])
        for drone in drone_ids
    )
    audit = _audit(trajectory, manifest["c3_group_slot"]["group_order"])
    state = run["c3_group_slot"] or {}
    latency = run["ra_solve_latency_summary_ms"]
    go = bool(
        mission_complete
        and run["infrastructure_valid"]
        and not run["collision"]
        and run["min_rho"] is not None
        and run["min_rho"] > 0.0
        and audit["selected_qp_infeasible_steps"] == 0
        and run["safety_bypass_count"] == 0
        and audit["ra_bypass_count"] == 0
        and audit["required_modes_seen"]
        and audit["frozen_hold_goals_stable"]
        and audit["authorized_groups_valid"]
        and audit["authorized_groups_seen"] == manifest["c3_group_slot"]["group_order"]
        and state.get("complete") is True
        and state.get("completed") == drone_ids
        and latency["p99"] < 50.0
        and latency["deadline_misses"] / int(manifest["max_steps"]) < 0.01
    )
    summary = {
        "protocol_id": protocol_id,
        "phase": manifest["phase"],
        "decision": "GO" if go else "NO_GO",
        "manifest_sha256": _sha256(args.manifest),
        "gazebo_seed": args.seed if args.seed is not None else manifest["gazebo_seed"],
        "seed": args.seed,
        "m4_go": go,
        "trial": {
            "condition": "M4_C3_GROUP_SLOT",
            "mission_complete": mission_complete,
            "collision": run["collision"],
            "min_rho": run["min_rho"],
            "min_distance_m": run["min_distance_m"],
            "path_length_m": run["path_length_m"],
            "mean_control_effort": effort,
            "ra_solve_latency_summary_ms": latency,
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
            "c3_group_slot_summary": state,
            "audit": audit,
            "trajectory": trajectory.name,
            "trajectory_sha256": _sha256(trajectory),
            "final_positions": run["final_positions"],
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "manifest.json").write_bytes(args.manifest.read_bytes())
    names = ["manifest.json", trajectory.name, "summary.json"]
    (args.output / "integrity.sha256").write_text(
        "".join(f"{_sha256(args.output / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    (args.output / "COMPLETE").touch()
    print(json.dumps({"decision": summary["decision"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
