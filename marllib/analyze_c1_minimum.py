#!/usr/bin/env python3
"""Paired bootstrap summary for a completed frozen C1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_ci(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = {(r["scenario"], r["ablation"], r["seed"]): r for r in data["episodes"]}
    seeds = data["config"]["seeds"]
    comparisons = []
    for scenario in data["config"]["scenarios"]:
        for reference in ("fixed_distance", "no_perception_margin", "no_aoi_margin"):
            if reference == "no_perception_margin" and scenario != "perception_dropout":
                continue
            if reference == "no_aoi_margin" and scenario != "telemetry_delay":
                continue
            group = [(rows[scenario, "full_envelope", seed], rows[scenario, reference, seed]) for seed in seeds]
            metrics = {
                "collision": np.array([float(a["collision"]) - float(b["collision"]) for a, b in group]),
                "boundary_violation": np.array([float(a["min_rho"] < 0.0) - float(b["min_rho"] < 0.0) for a, b in group]),
                "completion": np.array([float(a["completed"]) - float(b["completed"]) for a, b in group]),
                "min_pairwise_distance_m": np.array([a["min_pairwise_distance_m"] - b["min_pairwise_distance_m"] for a, b in group]),
            }
            comparisons.append({
                "scenario": scenario,
                "treatment": "full_envelope",
                "reference": reference,
                "n_pairs": len(group),
                "metrics": {
                    name: paired_ci(values, draws=args.draws, seed=args.seed + index)
                    for index, (name, values) in enumerate(metrics.items())
                },
            })
    payload = {
        "protocol": "aegisair-minimum-c1-v2-paired-bootstrap",
        "input": str(args.input),
        "draws": args.draws,
        "seed": args.seed,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
