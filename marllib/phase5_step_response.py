"""Single-PX4 velocity step-response identification (Phase 5 P3).

Measures the real PX4 velocity-tracking dynamics so the CBF ``M_dyn`` margin is
parameterised from live SITL data instead of the lightweight point-mass sim.

Protocol (frozen):
    1. arm + takeoff to cruise altitude;
    2. hold for a short settle;
    3. step forward velocity to ``--vx``;
    4. after ``--run-s``, step forward velocity back to 0;
    5. land and write a JSONL telemetry log plus aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paho.mqtt.client as mqtt

from marllib.phase5_runner import (
    ALTITUDE_HOLD_KP,
    CRUISE_ALTITUDE_M,
    MAX_VERTICAL_SPEED_MPS,
    build_phase5_velocity_command,
    snapshot_from_telemetry,
)


def _vertical_velocity(position_z: float) -> float:
    return max(
        -MAX_VERTICAL_SPEED_MPS,
        min(MAX_VERTICAL_SPEED_MPS, ALTITUDE_HOLD_KP * (CRUISE_ALTITUDE_M - position_z)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--run-s", type=float, default=6.0)
    parser.add_argument("--stop-s", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    drone = args.drone
    telemetry = {"payload": None, "snap": None}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"swarm/drone/{drone}/telemetry")
        client.subscribe(f"px4/{drone}/command_ack")

    def on_message(client, userdata, msg):
        if not msg.topic.endswith("/telemetry"):
            return
        try:
            payload = json.loads(msg.payload.decode())
        except (ValueError, UnicodeDecodeError):
            return
        telemetry["payload"] = payload
        telemetry["snap"] = snapshot_from_telemetry(payload)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, client_id="phase5-step-response"
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    time.sleep(0.8)

    def publish(action: str, *, velocity=None, altitude_m=None, native=False) -> None:
        now_ms = int(time.time_ns() // 1_000_000)
        payload = {
            "drone": drone,
            "action": action,
            "ttl_sec": 0.5,
            "command_id": f"step-{action}-{drone}-{now_ms}",
            "timestamp_ms": now_ms,
            "source_frame": "PX4_NED" if native else "FLU",
            "priority": "normal",
        }
        if velocity is not None:
            payload["velocity"] = list(velocity)
        if altitude_m is not None:
            payload["altitude_m"] = altitude_m
        topic = f"px4/{drone}/command" if native else f"swarm/drone/{drone}/command"
        client.publish(topic, json.dumps(payload, separators=(",", ":")))

    def wait_until(predicate, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if telemetry["snap"] is not None and predicate(telemetry["snap"]):
                return True
            time.sleep(0.1)
        return False

    deadline = time.time() + 30.0
    while time.time() < deadline and telemetry["snap"] is None:
        time.sleep(0.1)
    if telemetry["snap"] is None:
        raise SystemExit("telemetry not ready")

    for _ in range(8):
        publish("arm", native=True)
        time.sleep(0.3)
    if not wait_until(lambda s: s.status == "armed", 30.0):
        raise SystemExit("arm failed")

    for _ in range(8):
        publish("takeoff", altitude_m=CRUISE_ALTITUDE_M, native=True)
        time.sleep(0.3)
    if not wait_until(lambda s: s.position[2] > CRUISE_ALTITUDE_M * 0.8, 30.0):
        raise SystemExit("takeoff failed")

    rows: list[dict] = []
    t0 = time.monotonic()

    def record(tag: str) -> None:
        snap = telemetry["snap"]
        if snap is None:
            return
        rows.append(
            {
                "t_s": round(time.monotonic() - t0, 4),
                "tag": tag,
                "vx": round(snap.velocity[0], 4) if snap.velocity else 0.0,
                "vy": round(snap.velocity[1], 4) if snap.velocity else 0.0,
                "vz": round(snap.velocity[2], 4) if snap.velocity else 0.0,
                "x": round(snap.position[0], 4),
                "y": round(snap.position[1], 4),
                "z": round(snap.position[2], 4),
            }
        )

    # Settle, then accelerate forward.
    settle_end = t0 + args.settle_s
    while time.monotonic() < settle_end:
        snap = telemetry["snap"]
        if snap is None:
            time.sleep(0.05)
            continue
        publish(
            "velocity",
            velocity=(0.0, 0.0, _vertical_velocity(snap.position[2])),
        )
        record("settle")
        time.sleep(0.1)

    run_end = time.monotonic() + args.run_s
    while time.monotonic() < run_end:
        snap = telemetry["snap"]
        if snap is None:
            time.sleep(0.05)
            continue
        publish(
            "velocity",
            velocity=(args.vx, 0.0, _vertical_velocity(snap.position[2])),
        )
        record("run")
        time.sleep(0.1)

    stop_t = time.monotonic()
    stop_end = stop_t + args.stop_s
    while time.monotonic() < stop_end:
        snap = telemetry["snap"]
        if snap is None:
            time.sleep(0.05)
            continue
        publish(
            "velocity",
            velocity=(0.0, 0.0, _vertical_velocity(snap.position[2])),
        )
        record("stop")
        time.sleep(0.1)

    publish("land", native=True)
    time.sleep(1.0)
    client.loop_stop()
    client.disconnect()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n",
        encoding="utf-8",
    )

    stop_rows = [r for r in rows if r["tag"] == "stop"]
    run_rows = [r for r in rows if r["tag"] == "run"]
    v0 = max((r["vx"] for r in run_rows), default=args.vx)

    tau_ctrl = None
    if stop_rows:
        stop_start_x = stop_rows[0]["x"]
        # First sample after the command where vx clearly starts to drop.
        drop = next((r for r in stop_rows if r["vx"] < 0.9 * v0), None)
        if drop is not None:
            tau_ctrl = round(drop["t_s"] - stop_rows[0]["t_s"], 4)
        stop_end_x = stop_rows[-1]["x"]
        d_brake = max(0.0, stop_end_x - stop_start_x)
    else:
        d_brake = 0.0

    a_eff = (v0 * v0) / (2.0 * d_brake) if d_brake > 1e-3 else None

    result = {
        "drone": drone,
        "v0_mps": round(v0, 4),
        "tau_ctrl_s": tau_ctrl,
        "d_brake_m": round(d_brake, 4),
        "a_eff_mps2": round(a_eff, 4) if a_eff is not None else None,
        "samples": len(rows),
        "out": str(args.out),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
