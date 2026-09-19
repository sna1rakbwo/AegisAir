#!/usr/bin/env bash
# Run one fresh recoverability-admission PX4/Gazebo condition.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?usage: $0 MANIFEST TRIAL_ID CONDITION OUT_DIR}"
TRIAL_ID="${2:?usage: $0 MANIFEST TRIAL_ID CONDITION OUT_DIR}"
CONDITION="${3:?usage: $0 MANIFEST TRIAL_ID CONDITION OUT_DIR}"
OUT_DIR="${4:?usage: $0 MANIFEST TRIAL_ID CONDITION OUT_DIR}"
PYTHON="${PYTHON:-python}"
PX4_DIR="${PX4_DIR:?Set PX4_DIR to a PX4-Autopilot v1.16 checkout}"

[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite $OUT_DIR" >&2; exit 2; }
mapfile -t trial_setup < <("$PYTHON" - "$MANIFEST" "$TRIAL_ID" "$CONDITION" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for trial in manifest["trials"]:
    if trial["trial_id"] != sys.argv[2]:
        continue
    if sys.argv[3] not in trial.get("condition_order", []):
        raise SystemExit("condition is not assigned to this trial")
    geometry = manifest["geometries"][trial["geometry_id"]]
    starts = geometry["reset_starts"]
    poses = []
    for drone in manifest["drone_ids"]:
        start = starts[str(drone)]
        if len(start) != 3 or not all(math.isfinite(float(value)) for value in start):
            raise SystemExit(f"invalid reset start for drone {drone}")
        poses.append(
            f"{int(drone)}={float(start[0]):.12g},{float(start[1]):.12g},0.5"
        )
    print(trial["seed"])
    print(";".join(poses))
    break
else:
    raise SystemExit("unknown trial id")
PY
)
seed="${trial_setup[0]:?missing trial seed}"
poses="${trial_setup[1]:?missing trial launch poses}"

mkdir -p "$ROOT/.runtime"
"$PYTHON" "$ROOT/scripts/gcs_heartbeat.py" \
  --port 18572 --duration-s 900 >"$ROOT/.runtime/gcs_2.log" 2>&1 &
heartbeat_2=$!
"$PYTHON" "$ROOT/scripts/gcs_heartbeat.py" \
  --port 18573 --duration-s 900 >"$ROOT/.runtime/gcs_3.log" 2>&1 &
heartbeat_3=$!
launch_log="$ROOT/.runtime/launch_${TRIAL_ID}_${CONDITION}.log"
PX4_DIR="$PX4_DIR" GAZEBO_SEED="$seed" POSES="$poses" \
  "$ROOT/scripts/start_paper_s1_sitl.sh" >"$launch_log" 2>&1 &
launcher=$!

cleanup() {
  [[ -n "${state_dir:-}" ]] \
    && "$ROOT/scripts/stop_paper_sitl.sh" "$state_dir" \
    || true
  kill "$launcher" "$heartbeat_2" "$heartbeat_3" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  state_dir="$(head -n 1 "$launch_log" 2>/dev/null || true)"
  [[ -f "${state_dir:-}/STATE" ]] && break
  sleep 1
done
[[ -f "${state_dir:-}/STATE" ]] || { cat "$launch_log" >&2; exit 1; }

"$PYTHON" "$ROOT/scripts/wait_for_px4_telemetry.py" \
  --drone-ids 2 3 --timeout-s 90
"$PYTHON" "$ROOT/marllib/run_c_recoverability_admission_gazebo.py" \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --trial-id "$TRIAL_ID" \
  --condition "$CONDITION"
