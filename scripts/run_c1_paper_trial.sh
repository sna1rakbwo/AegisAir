#!/usr/bin/env bash
# Run one fresh PX4/Gazebo condition from a frozen C1 manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?usage: $0 MANIFEST TRIAL_ID METHOD OUT_DIR}"
TRIAL_ID="${2:?usage: $0 MANIFEST TRIAL_ID METHOD OUT_DIR}"
METHOD="${3:?usage: $0 MANIFEST TRIAL_ID METHOD OUT_DIR}"
OUT_DIR="${4:?usage: $0 MANIFEST TRIAL_ID METHOD OUT_DIR}"
PYTHON="${PYTHON:-python}"
PX4_DIR="${PX4_DIR:?Set PX4_DIR to a PX4-Autopilot v1.16 checkout}"

[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite $OUT_DIR" >&2; exit 2; }
seed="$("$PYTHON" - "$MANIFEST" "$TRIAL_ID" "$METHOD" <<'PY'
import json, sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
for trial in m["trials"]:
    if trial["trial_id"] == sys.argv[2]:
        if sys.argv[3] not in trial.get("condition_order", []):
            raise SystemExit("method is not assigned to this trial")
        print(trial["seed"])
        break
else:
    raise SystemExit("unknown trial id")
PY
)"

mkdir -p "$ROOT/.runtime"
"$PYTHON" "$ROOT/scripts/gcs_heartbeat.py" --port 18572 --duration-s 900 >"$ROOT/.runtime/gcs_2.log" 2>&1 & hb2=$!
"$PYTHON" "$ROOT/scripts/gcs_heartbeat.py" --port 18573 --duration-s 900 >"$ROOT/.runtime/gcs_3.log" 2>&1 & hb3=$!
launch_log="$ROOT/.runtime/launch_${TRIAL_ID}_${METHOD}.log"
PX4_DIR="$PX4_DIR" GAZEBO_SEED="$seed" "$ROOT/scripts/start_paper_s1_sitl.sh" >"$launch_log" 2>&1 & launcher=$!
cleanup() {
  [[ -n "${state_dir:-}" ]] && "$ROOT/scripts/stop_paper_sitl.sh" "$state_dir" || true
  kill "$launcher" "$hb2" "$hb3" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  state_dir="$(head -n 1 "$launch_log" 2>/dev/null || true)"
  [[ -f "${state_dir:-}/STATE" ]] && break
  sleep 1
done
[[ -f "${state_dir:-}/STATE" ]] || { cat "$launch_log" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/wait_for_px4_telemetry.py" --drone-ids 2 3 --timeout-s 90
"$PYTHON" "$ROOT/marllib/run_c1_sota_cbf_gazebo.py" \
  --manifest "$MANIFEST" --out-dir "$OUT_DIR" --trial-id "$TRIAL_ID" --method "$METHOD"
