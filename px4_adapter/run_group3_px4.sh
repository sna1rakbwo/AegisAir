#!/usr/bin/env bash
set -euo pipefail

# Run one PX4-based Group 3 C3 head-on trial using the migrated system.
SCENARIO="${SCENARIO:-head_on_crossing}"
SEED="${SEED:-1}"
TIMEOUT_S="${TIMEOUT_S:-180}"
OUT="${OUT:-/Volumes/Expansion/safety_margin_scheduler_gate0_runtime/group3_px4_20260814/result.json}"

GATE0="/Users/lijiajun/Documents/ChatGPT/无人机论文尝试/safety_margin_scheduler_gate0"
SAFE="/Users/lijiajun/Documents/drone/SafeDrones"
CONTAINER="safety-margin-gate0-xrce"
STATE=""
HOST_PIDS=()

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

python3 "$GATE0/scripts/gcs_heartbeat.py" --port 18572 --duration-s 1800 >/tmp/gcs18572.log 2>&1 &
HOST_PIDS+=($!)
python3 "$GATE0/scripts/gcs_heartbeat.py" --port 18573 --duration-s 1800 >/tmp/gcs18573.log 2>&1 &
HOST_PIDS+=($!)

SKIP_POSE=1 "$GATE0/scripts/launch_multi_sitl.sh" S1 2,3 >/tmp/group3-launch.log 2>&1 &
HOST_PIDS+=($!)

STATE=""
for _ in $(seq 1 120); do
  STATE="$(ls -td /Volumes/Expansion/safety_margin_scheduler_gate0_runtime/multi_state.* 2>/dev/null | head -1 || true)"
  [[ -n "$STATE" && -f "$STATE/STATE" ]] && break
  sleep 1
done
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

for instance in 2 3; do
  docker exec -d "$CONTAINER" bash -lc "
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
export PYTHONPATH=/work:\$PYTHONPATH
python3 /work/px4_adapter/node.py --config /work/px4_adapter/config.yaml --instance $instance --no-read-only > /tmp/a$instance.log 2>&1
"
done

for instance in 2 3; do
  ready=0
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" grep -q "received first vehicle_local_position" "/tmp/a$instance.log" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "adapter $instance did not receive telemetry" >&2; exit 2; }
done

(
  cd "$SAFE"
  conda run -n eai-swarm env PYTHONPATH=. python px4_adapter/synthetic_stereo_publisher.py --host 127.0.0.1 --port 1883 --camera-drones 2,3 --rate-hz 10 --target-position 0,0,0
) >/tmp/group3-stereo.log 2>&1 &
HOST_PIDS+=($!)

for camera in 2 3; do
  (
    cd "$SAFE"
    conda run -n eai-swarm env PYTHONPATH=. python scripts/sim_cam_perception.py --host 127.0.0.1 --port 1883 --camera-drone "$camera" --friend-drone-ids "$camera"
  ) >"/tmp/group3-simcam$camera.log" 2>&1 &
  HOST_PIDS+=($!)
done

(
  cd "$SAFE"
  conda run -n eai-swarm env PYTHONPATH=. python px4_adapter/group3_px4_runner.py \
    --phase prepare --scenario "$SCENARIO" --seed "$SEED" --host 127.0.0.1 --port 1883 \
    --timeout-s 60 --out /tmp/group3-prepare.json
)

(
  cd "$SAFE"
  conda run -n eai-swarm env PYTHONPATH=. python marl_pilot.py --host 127.0.0.1 --port 1883 --input-mode ego --ego-state-ttl-sec 0.6 --interval 0.2
) >/tmp/group3-marl.log 2>&1 &
HOST_PIDS+=($!)

(
  cd "$SAFE"
  conda run -n eai-swarm env PYTHONPATH=. python safety_gate.py --host 127.0.0.1 --port 1883 --input-mode ego --ego-state-ttl-sec 0.6 --safe-distance 2.55 --escape-distance 2.15
) >/tmp/group3-gate.log 2>&1 &
HOST_PIDS+=($!)

sleep 3

(
  cd "$SAFE"
  conda run -n eai-swarm env PYTHONPATH=. python px4_adapter/group3_px4_runner.py \
    --phase cross --scenario "$SCENARIO" --seed "$SEED" --host 127.0.0.1 --port 1883 \
    --timeout-s "$TIMEOUT_S" --out "$OUT"
)
