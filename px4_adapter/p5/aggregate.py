#!/usr/bin/env python3
"""Aggregate P5 local fault-scan episodes and apply the frozen stop rules."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("FAIL: no episodes")

    faults: dict[str, dict] = {}
    for fault_id in sorted({row["fault_id"] for row in rows}):
        sub = [row for row in rows if row["fault_id"] == fault_id]
        reasons: Counter[str] = Counter()
        latencies = []
        for row in sub:
            reasons.update(row["reasons"])
            if row["first_fail_closed_ms"] is not None and row["fault_onset_ms"] is not None:
                latencies.append(row["first_fail_closed_ms"] - row["fault_onset_ms"])
        faults[fault_id] = {
            "episodes": len(sub),
            "accepted": sum(row["accepted"] for row in sub),
            "rejected": sum(row["rejected"] for row in sub),
            "dropped": sum(row["dropped"] for row in sub),
            "bypass": sum(row["bypass"] for row in sub),
            "reasons": dict(reasons),
            "first_fail_closed_latency_ms": (
                {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": round(statistics.mean(latencies), 3),
                }
                if latencies
                else None
            ),
        }

    bypass_total = sum(fault["bypass"] for fault in faults.values())
    checks = {
        "bypass_total_zero": bypass_total == 0,
        "command_loss_monotonic": (
            faults["COMMAND_LOSS_30"]["accepted"]
            < faults["COMMAND_LOSS_20"]["accepted"]
            < faults["COMMAND_LOSS_10"]["accepted"]
            < faults["NONE"]["accepted"]
        ),
        "command_latency_1100_expired": (
            faults["COMMAND_LATENCY_1100MS"]["accepted"] == 0
            and faults["COMMAND_LATENCY_1100MS"]["reasons"].get("command_expired", 0) > 0
        ),
        "telemetry_stale_detected": (
            faults["TELEMETRY_STALE"]["reasons"].get("telemetry_stale", 0) > 0
            and faults["OFFBOARD_LOSS"]["reasons"].get("telemetry_stale", 0) > 0
        ),
        "gcs_lost_detected": (
            faults["GCS_CONNECTION_LOST"]["reasons"].get("gcs_connection_lost", 0) > 0
        ),
    }

    summary = {
        "protocol_id": rows[0].get("protocol_id"),
        "episodes": len(rows),
        "bypass_total": bypass_total,
        "checks": checks,
        "pass": all(checks.values()),
        "faults": faults,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
