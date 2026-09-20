#!/usr/bin/env python3
"""汇总 recoverability-admission calibration 并执行冻结 Go/No-Go 门。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _reconstruct_solver_feasible_publication_failures(
    trajectory_rows: list[dict[str, Any]],
) -> int:
    count = 0
    for record in trajectory_rows:
        drones = list(record.get("drones", {}).values())
        if (
            drones
            and all(
                drone.get("published_command_constraint_ok") is False
                for drone in drones
            )
            and all(drone.get("solver_feasible") is True for drone in drones)
        ):
            count += len(drones)
    return count


def _common_integrity(row: dict[str, Any]) -> bool:
    audit = row["trajectory_audit"]
    latency = row["ra_solve_latency_summary_ms"]
    return bool(
        row.get("infrastructure_valid", False)
        and row["safety_bypass_count"] == 0
        and audit["ra_bypass_count"] == 0
        and audit["failed_authority_revoked_all_steps"]
        and audit["failed_horizontal_command_zero_all_steps"]
        and row.get("published_command_mismatch_count") == 0
        and row.get("published_command_constraint_unknown_count") == 0
        and row.get(
            "published_constraint_failure_while_solver_feasible_count",
            row.get("published_command_constraint_failure_count"),
        ) == 0
        and latency["p99"] < 50.0
        and latency["deadline_misses"] == 0
    )


def _safe_gate(row: dict[str, Any]) -> bool:
    return bool(
        _common_integrity(row)
        and row.get("published_command_constraint_failure_count") == 0
        and not row["collision"]
        and row["min_rho"] is not None
        and row["min_rho"] > 0.0
        and row["trajectory_audit"]["selected_qp_infeasible_steps"] == 0
    )


def _post_failure_critical_reached(row: dict[str, Any]) -> bool:
    if "post_failure_critical_reached" in row:
        return bool(row["post_failure_critical_reached"])
    return bool(row["critical_reached"])


def analyze(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(manifest["trials"]) * len(manifest["conditions"])
    unique = {(row["trial_id"], row["condition"]) for row in rows}
    complete = len(rows) == expected and len(unique) == expected
    admission_rows = [
        row for row in rows if row["condition"] == "RECOVERABILITY_ADMISSION_RA"
    ]
    hold_rows = [row for row in rows if row["condition"] == "RA_ONLY_HOLD"]
    immediate_rows = [
        row for row in rows if row["condition"] == "IMMEDIATE_COMMIT_RA"
    ]
    admission_checks = []
    for row in admission_rows:
        summary = row["recoverability_admission"] or {}
        if row["expected_admission"] == "admit":
            passed = bool(
                _safe_gate(row)
                and summary.get("admission_count") == 1
                and summary.get("rejection_count") == 0
                and summary.get("plans_committed") == 1
                and summary.get("decision_latency_ms", float("inf")) < 50.0
                and _post_failure_critical_reached(row)
                and summary.get("completed")
            )
        else:
            passed = bool(
                _safe_gate(row)
                and summary.get("admission_count") == 0
                and summary.get("rejection_count") == 1
                and summary.get("plans_committed") == 0
                and summary.get("decision_latency_ms", float("inf")) < 50.0
                and not _post_failure_critical_reached(row)
                and row["trajectory_audit"]["hold_goal_frozen"]
            )
        admission_checks.append(
            {
                "trial_id": row["trial_id"],
                "expected": row["expected_admission"],
                "passed": passed,
            }
        )
    hold_checks = [
        {
            "trial_id": row["trial_id"],
            "passed": bool(
                _safe_gate(row)
                and not _post_failure_critical_reached(row)
                and row["trajectory_audit"]["hold_goal_frozen"]
                and (row["recoverability_admission"] or {}).get("plans_committed") == 0
            ),
        }
        for row in hold_rows
    ]
    immediate_integrity = [
        {"trial_id": row["trial_id"], "passed": _common_integrity(row)}
        for row in immediate_rows
    ]
    anti_vacuity = bool(
        any(row["expected_admission"] == "admit" for row in admission_rows)
        and any(row["expected_admission"] == "reject" for row in admission_rows)
        and any(
            (row["recoverability_admission"] or {}).get("admission_count") == 1
            for row in admission_rows
        )
        and any(
            (row["recoverability_admission"] or {}).get("rejection_count") == 1
            for row in admission_rows
        )
    )
    go = bool(
        complete
        and len(admission_rows) == len(manifest["trials"])
        and len(hold_rows) == len(manifest["trials"])
        and len(immediate_rows) == len(manifest["trials"])
        and all(item["passed"] for item in admission_checks)
        and all(item["passed"] for item in hold_checks)
        and anti_vacuity
    )
    return {
        "protocol_id": manifest["protocol_id"],
        "decision": "GO" if go else "NO_GO",
        "complete": complete,
        "expected_conditions": expected,
        "found_conditions": len(rows),
        "anti_vacuity": anti_vacuity,
        "admission_checks": admission_checks,
        "hold_checks": hold_checks,
        "immediate_integrity_checks": immediate_integrity,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    geometry_by_trial = {
        trial["trial_id"]: manifest["geometries"][trial["geometry_id"]]
        for trial in manifest["trials"]
    }
    for path in sorted(args.root.glob("*/summary.json")):
        loaded = json.loads(path.read_text(encoding="utf-8"))["trials"]
        for row in loaded:
            trajectory = path.parent / row["trajectory"]
            trajectory_rows = [
                json.loads(line)
                for line in trajectory.read_text(encoding="utf-8").splitlines()
            ]
            row.setdefault(
                "published_constraint_failure_while_solver_feasible_count",
                _reconstruct_solver_feasible_publication_failures(trajectory_rows),
            )
            geometry = geometry_by_trial[row["trial_id"]]
            critical_goal = geometry["critical_goal"]
            healthy_drone = next(
                drone
                for drone in manifest["drone_ids"]
                if int(drone) != int(manifest["failed_drone"])
            )
            row["post_failure_critical_reached"] = any(
                float(
                    sum(
                        (
                            record["drones"][str(healthy_drone)]["pos"][axis]
                            - critical_goal[axis]
                        ) ** 2
                        for axis in (0, 1)
                    )
                    ** 0.5
                ) < float(manifest["goal_epsilon"])
                for record in trajectory_rows
                if int(record["step"]) >= int(manifest["change_step"])
            )
            rows.append(row)
    result = analyze(manifest, rows)
    (args.root / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["complete"]:
        (args.root / "COMPLETE").touch()
    print(json.dumps({key: result[key] for key in (
        "decision", "complete", "expected_conditions", "found_conditions", "anti_vacuity"
    )}, ensure_ascii=False))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
