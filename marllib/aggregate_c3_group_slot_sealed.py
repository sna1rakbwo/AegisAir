#!/usr/bin/env python3
"""汇总四机 GroupSlot sealed trial，并生成可审计统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half, center + half]


def _bootstrap_mean(values: list[float], seed: int = 20260825) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    invalid = 0
    for seed in manifest["sealed_seeds"]:
        seed_root = args.root / f"seed_{seed}"
        summaries = sorted(seed_root.glob("attempt_*/calibration/summary.json"))
        valid = [p for p in summaries if (p.parent / "COMPLETE").is_file()]
        if len(valid) != 1:
            continue
        rows.append(json.loads(valid[0].read_text(encoding="utf-8")))
        invalid += len(list(seed_root.glob("attempt_*/INVALID_INFRASTRUCTURE")))
    trials = [row["trial"] for row in rows]
    successes = sum(bool(row["m4_go"]) for row in rows)
    complete = len(rows) == len(manifest["sealed_seeds"])
    result = {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "decision": "GO" if complete and successes == len(rows) else ("NO_GO" if rows and successes < len(rows) else "INCOMPLETE"),
        "valid_trials": len(rows),
        "invalid_infrastructure_attempts": invalid,
        "joint_successes": successes,
        "joint_success_rate": successes / len(rows) if rows else None,
        "joint_success_wilson_95ci": _wilson(successes, len(rows)) if rows else None,
        "metrics": {},
        "trials": rows,
    }
    for name, getter in {
        "min_rho": lambda t: t["min_rho"],
        "min_distance_m": lambda t: t["min_distance_m"],
        "path_length_m": lambda t: t["path_length_m"],
        "mean_control_effort": lambda t: t["mean_control_effort"],
        "p99_latency_ms": lambda t: t["ra_solve_latency_summary_ms"]["p99"],
    }.items():
        values = [float(getter(trial)) for trial in trials]
        if values:
            result["metrics"][name] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "min": min(values), "max": max(values), "mean_bootstrap_95ci": _bootstrap_mean(values)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
