#!/usr/bin/env bash
set -euo pipefail

# Group 3 C3 head-on crossing against PX4 with the Safety Gate in the loop.
# px4_adapter runs read-only (telemetry -> MQTT), the Gate 0 position controller
# owns flight control and honors safety overrides from MQTT.

SEED="${SEED:-1}"
EGO="${EGO:-0}"
NO_GATE="${NO_GATE:-0}"
REAL_EGO="${REAL_EGO:-0}"
export YAW2="${YAW2:-0.0}"
export YAW3="${YAW3:-180.0}"
INSTANCES="${INSTANCES:-2,3}"
ALTITUDE_M="${ALTITUDE_M:-1.0}"
CLIMB_S="${CLIMB_S:-4.0}"
DURATION_S="${DURATION_S:-45.0}"
SAFE_DISTANCE="${SAFE_DISTANCE:-6.0}"
ESCAPE_DISTANCE="${ESCAPE_DISTANCE:-5.0}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.40}"

# Match the Stage 4 head-on lateral jitter: derive a per-seed lateral offset
# so each seed is a reproducible near-miss geometry.
eval "$(python3 - "$SEED" <<'PY'
import random, sys
seed = int(sys.argv[1])
lateral = random.Random(seed).uniform(-0.4, 0.4)
lateral = round(lateral, 4)
# PX4 NED y = -FLU y. Instance 2 mirrors Stage-4 drone1, instance 3 drone2.
print(f"LATERAL={lateral}")
print(f"POSE2={-lateral},-3,0.5")
print(f"POSE3={lateral},3,0.5")
print(f"OFF2=-3,{-lateral},0")
print(f"OFF3=3,{lateral},0")
print(f"TX2=6.0")
print(f"TY2={2*lateral}")
print(f"TX3=-6.0")
print(f"TY3={-2*lateral}")
PY
)"

MODE="gt"
[[ "$EGO" == "1" && "$REAL_EGO" == "1" ]] && MODE="ego_stereo"
[[ "$EGO" == "1" && "$REAL_EGO" != "1" ]] && MODE="ego"
[[ "$NO_GATE" == "1" ]] && MODE="nogate"
GATE_INPUT="gt"
[[ "$EGO" == "1" ]] && GATE_INPUT="ego"

GATE0="/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0"
SAFE="/Users/lijiajun/Documents/drone/SafeDrones"
RUNTIME_ROOT="/Volumes/Expansion/safety_margin_scheduler_gate0_runtime"
OUTDIR="$RUNTIME_ROOT/group3_c3_${MODE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
CONTAINER="safety-margin-gate0-xrce"
STATE=""
HOST_PIDS=()

mkdir -p "$OUTDIR"

# Make sure no stale PX4/Gazebo/heartbeat processes from a previous run are
# still alive before launch; their shutdown is asynchronous.
for _ in $(seq 1 15); do
  for pid in $(pgrep -f 'px4.*-i ' 2>/dev/null); do kill "$pid" 2>/dev/null; done
  for pid in $(pgrep -f 'gz sim.*safety_margin_gate0' 2>/dev/null); do kill "$pid" 2>/dev/null; done
  for pid in $(pgrep -f 'gcs_heartbeat.py' 2>/dev/null); do kill "$pid" 2>/dev/null; done
  sleep 1
done
for pid in $(pgrep -f 'px4.*-i ' 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
for pid in $(pgrep -f 'gz sim.*safety_margin_gate0' 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done

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

STATE=""
MARKER="$RUNTIME_ROOT/.group3_launch_marker_$$"
touch "$MARKER"
POSES="2=$POSE2;3=$POSE3" "$GATE0/scripts/launch_multi_sitl_pose.sh" S1 "$INSTANCES" >"$OUTDIR/launch.log" 2>&1 &
HOST_PIDS+=($!)

for _ in $(seq 1 120); do
  STATE="$(find "$RUNTIME_ROOT" -maxdepth 1 -type d -name 'multi_state.*' -newer "$MARKER" -print -quit 2>/dev/null || true)"
  [[ -n "$STATE" && -f "$STATE/STATE" ]] && break
  sleep 1
done
rm -f "$MARKER"
[[ -n "$STATE" && -f "$STATE/STATE" ]] || { echo "multi SITL did not become ready" >&2; exit 2; }

docker run -d --rm --name "$CONTAINER" \
  -e XRCE_UDP_PORT=8889 \
  -v "$GATE0/nodes:/work/nodes:ro" \
  -v "$SAFE/px4_adapter:/work/px4_adapter:ro" \
  -v "$SAFE:/work/safedrones:ro" \
  -p 8889:8889/udp -p 1883:1883 \
  safety-margin-gate0-ros2:humble >/dev/null

docker exec "$CONTAINER" bash -lc '
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --quiet
python3 -m pip install --no-cache-dir --quiet paho-mqtt amqtt pyyaml
printf "%s\n" \
  "listeners:" "  default:" "    type: tcp" "    bind: 0.0.0.0:1883" \
  "sys_interval: 10" "auth:" "  allow-anonymous: true" "topic-check:" "  enabled: false" \
  > /tmp/broker.yml
'

docker exec -d "$CONTAINER" amqtt -c /tmp/broker.yml

# Wait for the in-container broker to accept connections before any MQTT
# client starts; amqtt can take a moment to bind its TCP listener.
for _ in $(seq 1 30); do
  if docker exec "$CONTAINER" bash -lc 'python3 -c "import socket; s=socket.create_connection((\"127.0.0.1\",1883),timeout=1); s.close()"' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Wait for both PX4 instances to finish uXRCE-DDS session setup.
for _ in $(seq 1 90); do
  ok=1
  for id in 2 3; do
    grep -q "successfully created rt/px4_${id}/fmu/out/vehicle_local_position data writer" "$STATE/px4_${id}.log" 2>/dev/null || ok=0
  done
  [[ "$ok" == 1 ]] && break
  sleep 1
done

sleep 2

for id in 2 3; do
  if [[ "$id" == 2 ]]; then
    offset="$OFF2"; tx="$TX2"; ty="$TY2"
  else
    offset="$OFF3"; tx="$TX3"; ty="$TY3"
  fi
  docker exec -d "$CONTAINER" bash -lc "
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
export PYTHONPATH=/work:\$PYTHONPATH
python3 /work/px4_adapter/node.py --config /work/px4_adapter/config.yaml --instance $id \
  --origin-offset-ned='$offset' > /tmp/a$id.log 2>&1
"
  docker exec -d "$CONTAINER" bash -lc "
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
python3 /work/nodes/gate0_position_controller.py \
  --instance $id --target-x $tx --target-y $ty \
  --altitude-m $ALTITUDE_M --climb-s $CLIMB_S --duration-s $DURATION_S \
  --ramp-speed-mps 1.5 \
  --yaw-deg $([[ "$id" == 2 ]] && echo "$YAW2" || echo "$YAW3") \
  --mqtt-host 127.0.0.1 --mqtt-port 1883 --origin-offset-ned='$offset' \
  --out /tmp/cross$id.jsonl > /tmp/cross$id.log 2>&1
"
done

for id in 2 3; do
  ready=0
  for _ in $(seq 1 90); do
    if docker exec "$CONTAINER" grep -q "received first vehicle_local_position" "/tmp/a$id.log" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "adapter $id did not receive telemetry" >&2; docker exec "$CONTAINER" tail -20 "/tmp/a$id.log" 2>/dev/null || true; exit 2; }
done

if [[ "$EGO" == "1" ]]; then
  if [[ "$REAL_EGO" == "1" ]]; then
    (
      cd "$SAFE"
      PYTHONPATH=. /opt/anaconda3/envs/eai-swarm/bin/python px4_adapter/synthetic_stereo_publisher.py \
        --host 127.0.0.1 --port 1883 --camera-drones 2,3 --rate-hz 10 \
        --target-position 0,0,0
    ) >"$OUTDIR/synthetic_stereo.log" 2>&1 &
    HOST_PIDS+=($!)
    for camera in 2 3; do
      (
        cd "$SAFE"
        PYTHONPATH=. /opt/anaconda3/envs/eai-swarm/bin/python scripts/sim_cam_perception.py \
          --host 127.0.0.1 --port 1883 --camera-drone "$camera" --friend-drone-ids "$camera"
      ) >"$OUTDIR/simcam$camera.log" 2>&1 &
      HOST_PIDS+=($!)
    done
  else
    (
      cd "$SAFE"
      PYTHONPATH=. /opt/anaconda3/envs/eai-swarm/bin/python scripts/fake_ego_perception.py \
        --host 127.0.0.1 --port 1883 --noise-std-m 0.10 --drop-rate 0.0 \
        --delay-sec 0.10 --seed "$SEED"
    ) >"$OUTDIR/fake_ego.log" 2>&1 &
    HOST_PIDS+=($!)
  fi
fi

if [[ "$NO_GATE" != "1" ]]; then
  (
    cd "$SAFE"
    PYTHONPATH=. /opt/anaconda3/envs/eai-swarm/bin/python safety_gate.py \
      --host 127.0.0.1 --port 1883 --input-mode "$GATE_INPUT" \
      --ego-state-ttl-sec 0.6 \
      --safe-distance "$SAFE_DISTANCE" --escape-distance "$ESCAPE_DISTANCE" \
      --high-threshold "$HIGH_THRESHOLD" \
      --interval 0.1
  ) >"$OUTDIR/safety_gate.log" 2>&1 &
  HOST_PIDS+=($!)
fi

sleep 3

# Poll until both drones reach their phase-2 targets or the deadline hits.
CONTAINER="$CONTAINER" DURATION_S="$DURATION_S" python3 - <<'PY'
import json
import os
import subprocess
import time

container = os.environ["CONTAINER"]
duration_s = float(os.environ["DURATION_S"])
targets = {2: 6.0, 3: -6.0}
deadline = time.time() + duration_s + 150.0


def rows(path):
    try:
        out = subprocess.run(
            ["docker", "exec", container, "cat", path],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return []
    result = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


while time.time() < deadline:
    r2 = rows("/tmp/cross2.jsonl")
    r3 = rows("/tmp/cross3.jsonl")
    reached2 = bool(r2) and abs(r2[-1].get("x", 0.0) - targets[2]) < 0.5
    reached3 = bool(r3) and abs(r3[-1].get("x", 0.0) - targets[3]) < 0.5
    print(
        f"poll 2:{len(r2)}/reached={reached2} 3:{len(r3)}/reached={reached3}",
        flush=True,
    )
    if reached2 and reached3:
        break
    time.sleep(2.0)
PY

docker cp "$CONTAINER:/tmp/cross2.jsonl" "$OUTDIR/instance2.jsonl" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/cross3.jsonl" "$OUTDIR/instance3.jsonl" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/cross2.log" "$OUTDIR/instance2.log" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/cross3.log" "$OUTDIR/instance3.log" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/a2.log" "$OUTDIR/adapter2.log" >/dev/null 2>&1 || true
docker cp "$CONTAINER:/tmp/a3.log" "$OUTDIR/adapter3.log" >/dev/null 2>&1 || true

python3 - "$OUTDIR" "$OFF2" "$OFF3" "$SEED" "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
offsets = {
    2: tuple(float(v) for v in sys.argv[2].split(",")),
    3: tuple(float(v) for v in sys.argv[3].split(",")),
}
seed = int(sys.argv[4])
mode = sys.argv[5]


def load(instance):
    path = outdir / f"instance{instance}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


r2 = load(2)
r3 = load(3)
min_distance = None
for a in r2:
    b = min(r3, key=lambda r: abs(r["t_s"] - a["t_s"]))
    p2 = (a["x"] + offsets[2][0], a["y"] + offsets[2][1], a["z"])
    p3 = (b["x"] + offsets[3][0], b["y"] + offsets[3][1], b["z"])
    d = math.dist(p2, p3)
    min_distance = d if min_distance is None else min(min_distance, d)

override_count = 0
gate_log = outdir / "safety_gate.log"
if gate_log.exists():
    override_count = gate_log.read_text().count("override drone=")

collision = min_distance is not None and min_distance < 0.25
near_miss = min_distance is not None and min_distance < 0.8
result = {
    "seed": seed,
    "mode": mode,
    "completed": bool(r2 and r3),
    "min_distance_m": round(min_distance, 4) if min_distance is not None else None,
    "collision": collision,
    "near_miss": near_miss,
    "override_count": override_count,
    "records": {2: len(r2), 3: len(r3)},
    "final": {
        2: {k: r2[-1][k] for k in ("x", "y", "z")} if r2 else None,
        3: {k: r3[-1][k] for k in ("x", "y", "z")} if r3 else None,
    },
}
(outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

# Append one line to the cumulative sweep log for later aggregation.
sweep = Path("/Volumes/Expansion/safety_margin_scheduler_gate0_runtime/group3_c3_sweep.jsonl")
with sweep.open("a") as fh:
    fh.write(json.dumps(result, separators=(",", ":")) + "\n")
PY

echo "OUTDIR=$OUTDIR"
