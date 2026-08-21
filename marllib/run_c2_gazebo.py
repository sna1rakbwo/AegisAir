#!/usr/bin/env python3
"""C2 Gazebo execution-model comparison (E0 / E1 / E2) against live PX4 SITL.

Requires the full stack already up: amqtt broker, 2x PX4 SITL + Gazebo,
GCS heartbeat, and the ``aegisair-adapters`` container publishing telemetry.

E0 = instantaneous (tau_px4=0, exact_zoh)
E1 = legacy trapezoidal heuristic (tau_px4=0.2, legacy_trapezoidal)
E2 = exact-ZOH projected barrier (tau_px4=0.2, exact_zoh)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marllib.phase5_runner import run_mqtt_loop


MODELS = {
    "E0": {"tau_px4": 0.0, "execution_model": "exact_zoh"},
    "E1": {"tau_px4": 0.2, "execution_model": "legacy_trapezoidal"},
    "E2": {"tau_px4": 0.2, "execution_model": "exact_zoh"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="C2 Gazebo E0/E1/E2 sweep")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--out", type=Path, default=Path("/Volumes/Expansion/Aegis/results/c2_gazebo_e0_e1_e2.json"))
    parser.add_argument("--models", default="E0,E1,E2")
    parser.add_argument("--lateral", type=float, default=0.0, help="head-on lateral offset (0 = direct crossing)")
    args = parser.parse_args()

    active = [m for m in args.models.split(",") if m in MODELS]
    drone_ids = [2, 3]
    episodes = []
    # A seed is one physical trial.  Run all execution models back-to-back for
    # that trial so the reset gate makes the resulting E0/E1/E2 rows pairable;
    # grouping all E0 runs before E1/E2 would silently let long-run SITL drift
    # confound the purported paired comparison.
    episode_order = 0
    for seed in range(1, args.seeds + 1):
        for name in active:
            cfg = MODELS[name]
            lateral = args.lateral
            base_goals = {
                2: (3.0, lateral, 2.5),
                3: (-3.0, -lateral, 2.5),
            }
            reset_starts = {2: (-3.0, 0.0, 2.5), 3: (3.0, 0.0, 2.5)}
            run = run_mqtt_loop(
                drone_ids=drone_ids,
                base_goals=base_goals,
                mode="CBF_ONLY",
                llm_client=None,
                llm_fallback=None,
                host=args.host,
                port=args.port,
                max_steps=args.max_steps,
                reset_starts=reset_starts,
                rate_hz=args.rate_hz,
                tau_ctrl=0.2,
                tau_px4=cfg["tau_px4"],
                execution_model=cfg["execution_model"],
                sampled_data=True,
                gamma=0.1,
            )
            min_distance = run.get("min_distance_m")
            episodes.append(
                {
                    "model": name,
                    "seed": seed,
                    "trial_index": seed,
                    "episode_order": episode_order,
                    "min_rho": run.get("min_rho"),
                    "min_distance_m": min_distance,
                    "collision": bool(min_distance is not None and min_distance < 0.25),
                    "cbf_events": run.get("cbf_events"),
                    "steps": run.get("steps"),
                    "reset_elapsed_s": run.get("reset_elapsed_s"),
                }
            )
            episode_order += 1
            print(
                f"{name} seed={seed} min_rho={run.get('min_rho')} "
                f"min_distance={min_distance} cbf={run.get('cbf_events')}",
                flush=True,
            )

    summary = {}
    for name in active:
        group = [e for e in episodes if e["model"] == name]
        rhos = [e["min_rho"] for e in group if e["min_rho"] is not None]
        dists = [e["min_distance_m"] for e in group if e["min_distance_m"] is not None]
        summary[name] = {
            "seeds": len(group),
            "collisions": sum(e["collision"] for e in group),
            "min_rho_worst": min(rhos) if rhos else None,
            "min_rho_mean": round(sum(rhos) / len(rhos), 6) if rhos else None,
            "min_distance_worst": min(dists) if dists else None,
            "min_distance_mean": round(sum(dists) / len(dists), 4) if dists else None,
        }

    payload = {
        "protocol_id": "aegisair-c2-gazebo-e0-e1-e2",
        "config": {
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "rate_hz": args.rate_hz,
            "tau_ctrl_s": 0.2,
            "gamma": 0.1,
            "tau_px4_s": {m: MODELS[m]["tau_px4"] for m in active},
            "execution_model": {m: MODELS[m]["execution_model"] for m in active},
            "models": active,
            "collision_radius_m": 0.25,
        },
        "summary": summary,
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
