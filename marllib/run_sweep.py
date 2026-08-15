"""Batch driver for the Phase 1 MARL curriculum sweep.

Runs ``train.py`` for every (scenario, seed) combination, skips combinations
whose ``final.pt`` already exists, and appends a compact progress record to a
sweep log on the external drive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.config import default_curriculum


DEFAULT_OUTPUT = "/Volumes/Expansion/safedrones_marllib"


def finished(output_root: Path, scenario: str, seed: int, domain_randomize: bool) -> bool:
    suffix = "_dr" if domain_randomize else ""
    return (output_root / f"{scenario}{suffix}" / f"seed{seed}" / "checkpoints" / "final.pt").exists()


def run_one(output_root: Path, scenario: str, seed: int, total_steps: int, domain_randomize: bool) -> bool:
    cmd = [
            sys.executable,
            str(ROOT / "marllib" / "train.py"),
            "--scenario",
            scenario,
            "--seed",
            str(seed),
            "--total-steps",
            str(total_steps),
            "--output",
            str(output_root),
        ]
    if domain_randomize:
        cmd.append("--domain-randomize")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 1 MARL curriculum sweep")
    parser.add_argument("--scenarios", default=None, help="comma-separated; default full curriculum")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--domain-randomize", action="store_true")
    args = parser.parse_args()

    curriculum = default_curriculum(domain_randomize=args.domain_randomize)
    if args.scenarios:
        names = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in curriculum if s.name in names]
        missing = names - {s.name for s in scenarios}
        if missing:
            print(f"unknown scenarios ignored: {sorted(missing)}", flush=True)
    else:
        scenarios = curriculum

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_log = output_root / "sweep.jsonl"

    for scenario in scenarios:
        for seed in seeds:
            if finished(output_root, scenario.name, seed, args.domain_randomize):
                print(f"SKIP {scenario.name} seed {seed}", flush=True)
                continue
            print(f"RUN  {scenario.name} seed {seed}", flush=True)
            t0 = time.time()
            ok = run_one(output_root, scenario.name, seed, args.steps, args.domain_randomize)
            elapsed = time.time() - t0
            record = {
                "scenario": scenario.name,
                "seed": seed,
                "steps": args.steps,
                "domain_randomize": args.domain_randomize,
                "ok": ok,
                "elapsed_s": round(elapsed, 1),
                "finished_at_ms": time.time_ns() // 1_000_000,
            }
            with sweep_log.open("a") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(f"     {scenario.name} seed {seed} -> {'OK' if ok else 'FAIL'} ({elapsed:.0f}s)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
