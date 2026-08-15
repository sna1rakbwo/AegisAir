#!/usr/bin/env bash
set -euo pipefail

# Prove two PX4 vehicles can separate to explicit -3 / +3 targets using the
# validated Gate 0 offboard position loop. This deliberately replaces the
# px4_adapter flight-control path; the adapter is not started here.

INSTANCES="${INSTANCES:-2,3}"
TARGET_NEG_X="${TARGET_NEG_X:--3.0}"
TARGET_POS_X="${TARGET_POS_X:-3.0}"
TARGET_NEG_Y="${TARGET_NEG_Y:-0.0}"
TARGET_POS_Y="${TARGET_POS_Y:-0.0}"
ALTITUDE_M="${ALTITUDE_M:-1.0}"
CLIMB_S="${CLIMB_S:-4.0}"
DURATION_S="${DURATION_S:-40.0}"
SKIP_POSE="${SKIP_POSE:-0}"
POSES="${POSES:-2=0,-3,0.5;3=0,3,0.5}"

GATE0="/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$GATE0/scripts/launch_multi_sitl_pose.sh}"
RUNTIME_ROOT="/Volumes/Expansion/safety_margin_scheduler_gate0_runtime"
OUTDIR="$RUNTIME_ROOT/group3_separation_$(date +%Y%m%d_%H%M%S)"
CONTAINER="safety-margin-gate0-xrce"
STATE=""
HOST_PIDS=()

mkdir -p "$OUTDIR"

cleanup() {
  local pid
  for pid in "${HOST_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  if [[ -n "$STATE" && -f "$STATE/STATE" ]]; then
    "$GATE0/scripts/stop_multi_sitl.sh" "$STATE" >/dev/null 2>&1 || true
  fi
  for pid in $(pgrep -f 'gcs_heartbeat.py' || true); do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

python3 "$GATE0/scripts/gcs_heartbeat.py" --port 18572 --duration-s 1800 >"$OUTDIR/gcs18572.log" 2>&1 &
HOST_PIDS+=($!)
python3 "$GATE0/scripts/gcs_heartbeat.py" --port 18573 --duration-s 1800 >"$OUTDIR/gcs18573.log" 2>&1 &
HOST_PIDS+=($!)

POSES="$POSES" "$LAUNCH_SCRIPT" S1 "$INSTANCES" >"$OUTDIR/launch.log" 2>&1 &
HOST_PIDS+=($!)

STATE=""
for _ in $(seq 1 120); do
  STATE="$(ls -td "$RUNTIME_ROOT"/multi_state.* 2>/dev/null | head -1 || true)"
  [[ -n "$STATE" && -f "$STATE/STATE" ]] && break
  sleep 1
done
[[ -n "$STATE" && -f "$STATE/STATE" ]] || { echo "multi SITL did not become ready" >&2; exit 2; }

docker run -d --rm --name "$CONTAINER" \
  -e XRCE_UDP_PORT=8889 \
  -v "$GATE0/nodes:/work/nodes:ro" \
  -p 8889:8889/udp \
  safety-margin-gate0-ros2:humble >/dev/null

# Wait for the container to be up.
for _ in $(seq 1 30); do
  docker exec "$CONTAINER" bash -lc 'true' >/dev/null 2>&1 && break
  sleep 1
done

# The PX4 uXRCE-DDS client retries against the agent, but a fresh SITL boot can
# take tens of seconds before the first data writer appears. Wait for both
# instances to finish their session setup before starting the controllers.
for _ in $(seq 1 90); do
  px4_2_ok=0
  px4_3_ok=0
  grep -q "successfully created rt/px4_2/fmu/out/vehicle_local_position data writer" "$STATE/px4_2.log" 2>/dev/null && px4_2_ok=1
  grep -q "successfully created rt/px4_3/fmu/out/vehicle_local_position data writer" "$STATE/px4_3.log" 2>/dev/null && px4_3_ok=1
  [[ "$px4_2_ok" == 1 && "$px4_3_ok" == 1 ]] && break
  sleep 1
done

docker exec -d "$CONTAINER" bash -lc "
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
python3 /work/nodes/gate0_position_controller.py \
  --instance 2 --target-x $TARGET_NEG_X --target-y $TARGET_NEG_Y --altitude-m $ALTITUDE_M \
  --climb-s $CLIMB_S --duration-s $DURATION_S --out /tmp/sep2.jsonl \
  > /tmp/sep2.log 2>&1
"

docker exec -d "$CONTAINER" bash -lc "
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
python3 /work/nodes/gate0_position_controller.py \
  --instance 3 --target-x $TARGET_POS_X --target-y $TARGET_POS_Y --altitude-m $ALTITUDE_M \
  --climb-s $CLIMB_S --duration-s $DURATION_S --out /tmp/sep3.jsonl \
  > /tmp/sep3.log 2>&1
"

# Poll until both drones are armed and near their targets, or a deadline hits.
CONTAINER="$CONTAINER" DURATION_S="$DURATION_S" \
TARGET_NEG_X="$TARGET_NEG_X" TARGET_POS_X="$TARGET_POS_X" \
TARGET_NEG_Y="$TARGET_NEG_Y" TARGET_POS_Y="$TARGET_POS_Y" python3 - <<'PY'
import json
import os
import subprocess
import time

container = os.environ["CONTAINER"]
duration_s = float(os.environ["DURATION_S"])
targets = {
    2: (float(os.environ["TARGET_NEG_X"]), float(os.environ["TARGET_NEG_Y"])),
    3: (float(os.environ["TARGET_POS_X"]), float(os.environ["TARGET_POS_Y"])),
}
deadline = time.time() + duration_s + 120.0


def read_rows(path):
    try:
        out = subprocess.run(
            ["docker", "exec", container, "cat", path],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


while time.time() < deadline:
    r2 = read_rows("/tmp/sep2.jsonl")
    r3 = read_rows("/tmp/sep3.jsonl")
    armed2 = any(r.get("armed") for r in r2)
    armed3 = any(r.get("armed") for r in r3)
    reached2 = bool(r2) and abs(r2[-1].get("x", 0.0) - targets[2][0]) < 0.5
    reached3 = bool(r3) and abs(r3[-1].get("x", 0.0) - targets[3][0]) < 0.5
    print(
        f"poll instances=2:{len(r2)}/armed={armed2}/reached={reached2} "
        f"3:{len(r3)}/armed={armed3}/reached={reached3}",
        flush=True,
    )
    if armed2 and armed3 and reached2 and reached3:
        break
    time.sleep(2.0)
PY

docker cp "$CONTAINER:/tmp/sep2.jsonl" "$OUTDIR/instance2.jsonl" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/sep3.jsonl" "$OUTDIR/instance3.jsonl" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/sep2.log" "$OUTDIR/instance2.log" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/sep3.log" "$OUTDIR/instance3.log" >/dev/null 2>&1 || true

python3 - "$OUTDIR" "$TARGET_NEG_X" "$TARGET_POS_X" <<'PY'
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
targets = {2: float(sys.argv[2]), 3: float(sys.argv[3])}
summary = {}
for instance in (2, 3):
    path = outdir / f"instance{instance}.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        summary[instance] = {"records": 0, "reached": False}
        continue
    final = rows[-1]
    reached = abs(final["x"] - targets[instance]) < 0.5
    summary[instance] = {
        "records": len(rows),
        "reached": reached,
        "final": {k: final[k] for k in ("x", "y", "z", "armed", "nav_state", "t_s")},
        "min_x": min(r["x"] for r in rows),
        "max_x": max(r["x"] for r in rows),
    }

separated = bool(summary.get(2, {}).get("reached") and summary.get(3, {}).get("reached"))
result = {"separated": separated, "instances": summary}
(outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "OUTDIR=$OUTDIR"
