from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import random
import time
from typing import Any


LOGGER = logging.getLogger("fake_ego_perception")


def build_mqtt_client(client_id: str):
    import paho.mqtt.client as mqtt

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def noisy_position(position: list[float], noise_std_m: float, rng: random.Random) -> list[float]:
    return [float(value) + rng.gauss(0.0, noise_std_m) for value in position]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish delayed/noisy ego position estimates from telemetry.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--qos", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--noise-std-m", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if args.noise_std_m < 0 or not 0 <= args.drop_rate < 1 or args.delay_sec < 0:
        raise SystemExit("noise must be non-negative, drop-rate must be in [0, 1), delay must be non-negative")

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)
    pending: deque[tuple[float, dict[str, Any]]] = deque()
    client = build_mqtt_client("fake-ego-perception")

    def on_connect(client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        LOGGER.info("connected to MQTT broker %s:%s result=%s", args.host, args.port, reason_code)
        client.subscribe("swarm/drone/+/telemetry", qos=args.qos)

    def on_message(client: Any, userdata: Any, message: Any) -> None:
        try:
            telemetry = json.loads(message.payload.decode("utf-8"))
            drone_id = int(telemetry["drone"])
            position = [float(value) for value in telemetry["position"]]
            if len(position) != 3:
                raise ValueError("position must contain three values")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("ignored invalid telemetry: %s", exc)
            return
        if rng.random() < args.drop_rate:
            return
        pending.append(
            (
                time.monotonic() + args.delay_sec,
                {
                    "drone": drone_id,
                    "position": noisy_position(position, args.noise_std_m, rng),
                    "estimated_depth": max(0.0, abs(position[2]) + rng.gauss(0.0, args.noise_std_m)),
                    "source": "fake_ego",
                    "timestamp_ms": time.time_ns() // 1_000_000,
                },
            )
        )

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    try:
        while True:
            now = time.monotonic()
            while pending and pending[0][0] <= now:
                _, payload = pending.popleft()
                client.publish(
                    f"swarm/drone_seen/{payload['drone']}/position",
                    payload=json.dumps(payload, separators=(",", ":")),
                    qos=args.qos,
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        LOGGER.info("stopping fake ego perception")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
