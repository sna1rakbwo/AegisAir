#!/usr/bin/env python3
"""审计冻结的 C1-v4 四条件 reserve/prediction 消融。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(root: Path) -> list[dict]:
    rows = []
    for summary in sorted(root.glob("abl*_*/*summary.json")):
        if summary.name.startswith("._"):
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if len(payload.get("trials", [])) != 1:
            raise ValueError(f"每个条件目录必须只有一个 trial：{summary}")
        rows.append({**payload["trials"][0], "condition_dir": summary.parent.name})
    return rows


def _bootstrap(delta: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = np.asarray([
        np.mean(rng.choice(delta, size=len(delta), replace=True))
        for _ in range(10_000)
    ])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _method_summary(rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        output.append({
            "method": method,
            "episodes": len(group),
            "completed": sum(bool(row["mission_complete"]) for row in group),
            "collisions": sum(bool(row["collision"]) for row in group),
            "minimum_rho": min(float(row["min_rho"]) for row in group),
            "mean_min_rho": float(np.mean([row["min_rho"] for row in group])),
            "mean_control_effort": float(np.mean([row["mean_control_effort"] for row in group])),
            "qp_infeasible_steps": sum(int(row["qp_infeasible_steps"]) for row in group),
            "mean_recovery_entries": float(np.mean([row["hocbf_recovery_entries"] for row in group])),
        })
    return output


def _paired(rows: list[dict], comparisons: list[list[str]]) -> list[dict]:
    by_trial: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_trial[row["trial_id"]][row["method"]] = row
    rng = np.random.default_rng(20260825)
    output = []
    for left, right, label in comparisons:
        pairs = [values for _, values in sorted(by_trial.items()) if left in values and right in values]
        metrics = {}
        for metric in ("qp_infeasible_steps", "mean_control_effort", "min_rho", "path_length_m"):
            delta = np.asarray([float(pair[left][metric]) - float(pair[right][metric]) for pair in pairs])
            metrics[metric] = {"mean_left_minus_right": float(np.mean(delta)), "bootstrap_95_ci": _bootstrap(delta, rng)}
        output.append({"label": label, "left": left, "right": right, "pairs": len(pairs), "metrics": metrics})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"拒绝覆盖审计文件：{args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = _rows(args.root)
    methods = set(manifest["methods"])
    expected = len(manifest["trials"]) * len(methods)
    payload = {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": _sha256(args.manifest),
        "conditions_expected": expected,
        "conditions_found": len(rows),
        "method_summary": _method_summary(rows) if rows else [],
        "paired": _paired(rows, manifest["comparisons"]) if rows else [],
        "go": bool(len(rows) == expected and all(
            row["method"] in methods and not row["collision"] and row["mission_complete"]
            and float(row["min_rho"]) > 0.0 for row in rows
        )),
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
