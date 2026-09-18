#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash

MicroXRCEAgent udp4 -p "${XRCE_UDP_PORT:-8889}" &
agent_pid=$!
cleanup() {
  kill "${agent_pid}" 2>/dev/null || true
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for instance in ${ADAPTER_INSTANCES:-2 3}; do
  offset_var="ORIGIN_OFFSET_${instance}"
  offset="${!offset_var:-}"
  args=(python3 /work/px4_adapter/node.py --config /work/px4_adapter/config.yaml --instance "${instance}" --no-read-only --mqtt-host "${MQTT_HOST:-mqtt}" --mqtt-port "${MQTT_PORT:-1883}" --control-rate-hz 20 --telemetry-rate-hz 20)
  if [[ -n "${offset}" ]]; then args+=(--origin-offset-ned="${offset}"); fi
  "${args[@]}" &
done
wait
