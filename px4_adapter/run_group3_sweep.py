#!/usr/bin/env python3
"""Drive the Group 3 PX4 C3 seed sweep with skip + retry.

Runs ``run_group3_crossing.sh`` for the requested (mode, seed) matrix, skips
combinations already present in the cumulative sweep log, and retries each
missing combination once on a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SWEEP_LOG = Path(
    "/Volumes/Expansion/safety_margin_scheduler_gate0_runtime/group3_c3_sweep.jsonl"
)
SCRIPT = Path(__file__).resolve().parent / "run_group3_crossing.sh"


def existing() -> set[tuple[str, int]]:
    if not SWEEP_LOG.exists():
        return set()
    seen: set[tuple[str, int]] = set()
    for line in SWEEP_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            seen.add((row["mode"], int(row["seed"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return seen


def run_one(mode: str, seed: int) -> bool:
    env = {
        "SEED": str(seed),
        "EGO": "1" if mode in ("ego", "ego_stereo") else "0",
        "REAL_EGO": "1" if mode == "ego_stereo" else "0",
        "NO_GATE": "1" if mode == "nogate" else "0",
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="gt,ego,nogate")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=10)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    done = existing()
    failed: list[tuple[str, int]] = []

    for mode in modes:
        for seed in range(args.start_seed, args.end_seed + 1):
            if (mode, seed) in done:
                print(f"SKIP {mode} seed {seed} (already present)", flush=True)
                continue
            ok = run_one(mode, seed)
            for attempt in range(args.retries):
                if ok:
                    break
                print(f"RETRY {mode} seed {seed} (attempt {attempt + 1})", flush=True)
                time.sleep(3)
                ok = run_one(mode, seed)
            if not ok:
                print(f"FAIL {mode} seed {seed}", flush=True)
                failed.append((mode, seed))
            else:
                print(f"OK   {mode} seed {seed}", flush=True)

    if failed:
        print("FAILED:", ", ".join(f"{m} seed {s}" for m, s in failed), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
