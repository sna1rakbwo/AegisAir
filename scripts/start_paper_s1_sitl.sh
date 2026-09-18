#!/usr/bin/env bash
# Start one public S1 PX4/Gazebo paper scenario on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:?Set PX4_DIR to a PX4-Autopilot v1.16 checkout}"
INSTANCES="${INSTANCES:-2,3}"
POSES="${POSES:-2=0,-3,0.5;3=0,3,0.5}"
SEED="${GAZEBO_SEED:-10001}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/.runtime}"
WORLD="$ROOT/worlds/s1_single_obstacle.sdf"
BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"

[[ -x "$BIN" ]] || { echo "PX4 SITL is not built: $BIN" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "GAZEBO_SEED must be an integer" >&2; exit 2; }
mkdir -p "$RUNTIME_ROOT"
STATE_DIR="$(mktemp -d "$RUNTIME_ROOT/s1.XXXXXX")"

source "$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh"
export PX4_GZ_WORLDS="$ROOT/worlds"
export GZ_SIM_RESOURCE_PATH="$PX4_GZ_MODELS:$PX4_GZ_WORLDS:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_PARTITION="aegisair-paper-${SEED}"

gz sim -r -s --seed "$SEED" "$WORLD" >"$STATE_DIR/gazebo.log" 2>&1 &
GZ_PID=$!
sleep 3
kill -0 "$GZ_PID" 2>/dev/null || { cat "$STATE_DIR/gazebo.log" >&2; exit 1; }

IFS=',' read -r -a ids <<< "$INSTANCES"
PIDS=()
for id in "${ids[@]}"; do
  rootfs="$STATE_DIR/rootfs_$id"
  cp -a "$PX4_DIR/build/px4_sitl_default/rootfs" "$rootfs"
  pose="0,0,0.5"
  IFS=';' read -r -a entries <<< "$POSES"
  for entry in "${entries[@]}"; do [[ "$entry" == "$id="* ]] && pose="${entry#*=}"; done
  (
    cd "$rootfs"
    export PX4_GZ_STANDALONE=1 PX4_SIM_MODEL=gz_x500 PX4_UXRCE_DDS_PORT=8889 PX4_GZ_MODEL_POSE="$pose"
    exec "$BIN" -d -i "$id"
  ) >"$STATE_DIR/px4_$id.log" 2>&1 &
  PIDS+=("$!")
done

printf 'state_dir=%q\ngazebo_pid=%q\npx4_pids=%q\n' "$STATE_DIR" "$GZ_PID" "${PIDS[*]}" >"$STATE_DIR/STATE"
echo "$STATE_DIR"
wait "${PIDS[@]}"
