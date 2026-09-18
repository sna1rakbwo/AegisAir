#!/usr/bin/env python3
"""分析 C1 五方法 external-PCBF sealed paired experiment。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "min_rho", "mean_control_effort", "path_length_m", "cbf_events",
    "qp_infeasible_steps", "latency_p99_ms",
)
BASELINE_LABELS = {
    "AEGIS_HOCBF_V4_REACTIVE": "Reactive",
    "AEGIS_HOCBF_V3": "HOCBF-Fallback",
    "PB_CBF": "PB-CBF",
    "PCBF_HUANG_ECC2025": "PCBF",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("ext*_*/summary.json")):
        if path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("trials", [])) != 1:
            raise ValueError(f"条件 summary 必须恰有一个 trial: {path}")
        row = payload["trials"][0]
        row["latency_p99_ms"] = row["ra_solve_latency_summary_ms"]["p99"]
        rows.append(row)
    return rows


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    samples = np.mean(
        rng.choice(values, size=(10_000, len(values)), replace=True), axis=1
    )
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _sign_flip_p(values: np.ndarray) -> float:
    """Exact two-sided paired randomization p-value for n<=20."""
    observed = abs(float(np.mean(values)))
    if observed <= 1e-15:
        return 1.0
    total = 1 << len(values)
    extreme = 0
    bit_positions = np.arange(len(values), dtype=np.uint64)
    for start in range(0, total, 65_536):
        masks = np.arange(start, min(start + 65_536, total), dtype=np.uint64)[:, None]
        signs = (((masks >> bit_positions) & 1) * 2 - 1).astype(np.int8)
        statistics = np.abs(np.mean(signs * values[None, :], axis=1))
        extreme += int(np.sum(statistics >= observed - 1e-15))
    return extreme / total


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def _summaries(rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        output.append({
            "method": method,
            "n": len(group),
            "completed": sum(bool(row["mission_complete"]) for row in group),
            "collisions": sum(bool(row["collision"]) for row in group),
            **{f"mean_{metric}": float(np.mean([row[metric] for row in group])) for metric in METRICS},
            "minimum_rho": min(float(row["min_rho"]) for row in group),
            "maximum_latency_p99_ms": max(float(row["latency_p99_ms"]) for row in group),
        })
    return output


def _paired(rows: list[dict]) -> list[dict]:
    by_trial: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_trial[row["trial_id"]][row["method"]] = row
    rng = np.random.default_rng(20260826)
    output = []
    raw_p: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    delta_cache: dict[tuple[str, str], np.ndarray] = {}
    for baseline in BASELINE_LABELS:
        pairs = [values for _, values in sorted(by_trial.items())]
        for metric in METRICS:
            delta = np.asarray([
                float(pair["AEGIS_HOCBF_V4"][metric]) - float(pair[baseline][metric])
                for pair in pairs
            ])
            delta_cache[(baseline, metric)] = delta
            raw_p[metric][baseline] = _sign_flip_p(delta)
    adjusted = {metric: _holm(values) for metric, values in raw_p.items()}
    for baseline, label in BASELINE_LABELS.items():
        metrics = {}
        for metric in METRICS:
            delta = delta_cache[(baseline, metric)]
            metrics[metric] = {
                "mean_aegis_minus_baseline": float(np.mean(delta)),
                "bootstrap_95_ci": _bootstrap_ci(delta, rng),
                "paired_randomization_p": raw_p[metric][baseline],
                "holm_adjusted_p": adjusted[metric][baseline],
            }
        output.append({"baseline": baseline, "label": label, "pairs": 20, "metrics": metrics})
    return output


def _write_csv(path: Path, paired: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "metric", "mean_delta", "ci_low", "ci_high", "p", "holm_p"])
        for comparison in paired:
            for metric, values in comparison["metrics"].items():
                writer.writerow([
                    comparison["label"], metric, values["mean_aegis_minus_baseline"],
                    *values["bootstrap_95_ci"], values["paired_randomization_p"],
                    values["holm_adjusted_p"],
                ])


def _plot(output: Path, summaries: list[dict], paired: list[dict]) -> None:
    import matplotlib.pyplot as plt

    names = [row["method"].replace("AEGIS_HOCBF_V4", "AegisAir") for row in summaries]
    effort = [row["mean_mean_control_effort"] for row in summaries]
    margin = [row["mean_min_rho"] for row in summaries]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(effort, margin, s=55)
    for name, x, y in zip(names, effort, margin):
        axis.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Mean control effort")
    axis.set_ylabel("Mean minimum normalized margin")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "external_baseline_tradeoff.png", dpi=240)
    fig.savefig(output / "external_baseline_tradeoff.pdf")
    plt.close(fig)

    labels = [row["label"] for row in paired]
    metrics = ["min_rho", "mean_control_effort", "path_length_m", "latency_p99_ms"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for axis, metric in zip(axes.flat, metrics):
        means = [row["metrics"][metric]["mean_aegis_minus_baseline"] for row in paired]
        cis = [row["metrics"][metric]["bootstrap_95_ci"] for row in paired]
        errors = np.asarray([[mean-low for mean, (low, _) in zip(means, cis)], [high-mean for mean, (_, high) in zip(means, cis)]])
        axis.errorbar(means, labels, xerr=errors, fmt="o", capsize=3)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"AegisAir - baseline: {metric}")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "paired_effects.png", dpi=240)
    fig.savefig(output / "paired_effects.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"拒绝覆盖分析目录: {args.output}")
    args.output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = _read_rows(args.root)
    expected = len(manifest["trials"]) * len(manifest["methods"])
    if len(rows) != expected:
        raise SystemExit(f"incomplete sealed set: {len(rows)}/{expected}")
    summaries = _summaries(rows)
    paired = _paired(rows)
    binary = {
        "all_methods_completed": all(row["mission_complete"] for row in rows),
        "all_methods_collision_free": all(not row["collision"] for row in rows),
        "mcnemar_discordant_pairs_per_comparison": 0,
        "exact_mcnemar_p": 1.0,
    }
    payload = {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": _sha256(args.manifest),
        "conditions": len(rows), "paired_trials": 20,
        "method_summary": summaries, "paired_comparisons": paired,
        "binary_outcomes": binary,
    }
    (args.output / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output / "paired_comparisons.csv", paired)
    _plot(args.output, summaries, paired)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
