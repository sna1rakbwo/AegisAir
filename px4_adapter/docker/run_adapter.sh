#!/usr/bin/env bash
# Start the Micro-XRCE-DDS agent and one px4_adapter node in the same container.
#
# The agent bridges the host PX4 SITL instance (UDP) to the ROS 2 graph; the
# adapter then publishes PX4 state to MQTT and, when control is enabled,
# converts MQTT commands back to PX4 control messages.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
set -u

XRCE_UDP_PORT="${XRCE_UDP_PORT:-8889}"
ADAPTER_INSTANCES="${ADAPTER_INSTANCES:-2}"
ADAPTER_ARGS="${ADAPTER_ARGS:---read-only}"
export PYTHONPATH="/work:${PYTHONPATH:-}"

MicroXRCEAgent udp4 -p "$XRCE_UDP_PORT" &
agent_pid=$!

cleanup() {
    for pid in "${adapter_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    kill "$agent_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

adapter_pids=()
for instance in $ADAPTER_INSTANCES; do
    offset_var="ORIGIN_OFFSET_${instance}"
    offset="${!offset_var:-}"
    args=( --config /work/px4_adapter/config.yaml --instance "$instance" )
    # ADAPTER_ARGS is intentionally word-split here (it is a shared arg list).
    # shellcheck disable=SC2206
    args+=( ${ADAPTER_ARGS} )
    if [[ -n "$offset" ]]; then
        args+=( "--origin-offset-ned=$offset" )
    fi
    python3 /work/px4_adapter/node.py "${args[@]}" &
    adapter_pids+=("$!")
done

wait
