#!/usr/bin/env bash
# PX4-adapter packet-loss/latency demo.
#
# Starts a temporary amqtt broker and the px4_adapter/mqtt_lossy_proxy, then
# publishes a fixed number of raw commands and reports how many reached the
# adapter-native command topic.
set -euo pipefail

host="${MQTT_HOST:-127.0.0.1}"
port="${MQTT_PORT:-1883}"
instance="${INSTANCE:-2}"
loss_rate="${LOSS_RATE:-0.2}"
latency_ms="${LATENCY_MS:-0}"
count="${COUNT:-20}"
seed="${SEED:-1}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/px4-packet-loss.XXXXXX")"
broker_pid=""
proxy_pid=""

cleanup() {
  [[ -n "$proxy_pid" ]] && kill "$proxy_pid" 2>/dev/null || true
  [[ -n "$broker_pid" ]] && kill "$broker_pid" 2>/dev/null || true
  rm -rf "$workdir"
}
trap cleanup EXIT INT TERM

cat > "$workdir/broker.yml" <<'EOF'
listeners:
  default:
    type: tcp
    bind: 127.0.0.1:1883
sys_interval: 10
auth:
  allow-anonymous: true
topic-check:
  enabled: false
EOF

amqtt -c "$workdir/broker.yml" >"$workdir/broker.log" 2>&1 &
broker_pid=$!
sleep 1

PYTHONPATH="$root" python3 "$root/px4_adapter/mqtt_lossy_proxy.py" \
  --host "$host" --port "$port" --instance "$instance" \
  --loss-rate "$loss_rate" --latency-ms "$latency_ms" --seed "$seed" \
  --event-log "$workdir/proxy_events.jsonl" >"$workdir/proxy.log" 2>&1 &
proxy_pid=$!
sleep 1

PYTHONPATH="$root" python3 - "$instance" "$count" "$host" "$port" <<'PY'
import json
import sys
import time

import paho.mqtt.client as mqtt

instance = int(sys.argv[1])
count = int(sys.argv[2])
host = sys.argv[3]
port = int(sys.argv[4])
forwarded = []

def on_connect(client, userdata, flags, rc, props=None):
    client.subscribe(f"px4/{instance}/command")

def on_message(client, userdata, msg):
    forwarded.append(json.loads(msg.payload.decode()))

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(host, port, keepalive=30)
client.loop_start()
time.sleep(0.3)
for index in range(count):
    payload = {
        "drone": instance,
        "action": "hold",
        "command_id": f"demo-{index}",
        "timestamp_ms": time.time_ns() // 1_000_000,
        "source_frame": "PX4_NED",
    }
    client.publish(f"px4_raw/{instance}/command", json.dumps(payload, separators=(",", ":")))
    time.sleep(0.05)
time.sleep(1.0)
client.loop_stop()
client.disconnect()
print(json.dumps({"published": count, "forwarded": len(forwarded)}, indent=2))
PY

dropped="$(wc -l < "$workdir/proxy_events.jsonl" 2>/dev/null || echo 0)"
echo "dropped_events=$dropped"
