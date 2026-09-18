#!/usr/bin/env bash
set -euo pipefail
STATE_DIR="${1:?usage: $0 STATE_DIR}"
source "$STATE_DIR/STATE"
for pid in $px4_pids "$gazebo_pid"; do kill "$pid" 2>/dev/null || true; done
for _ in $(seq 1 20); do
  alive=0
  for pid in $px4_pids "$gazebo_pid"; do kill -0 "$pid" 2>/dev/null && alive=1; done
  [[ "$alive" == 0 ]] && exit 0
  sleep 0.25
done
for pid in $px4_pids "$gazebo_pid"; do kill -KILL "$pid" 2>/dev/null || true; done
