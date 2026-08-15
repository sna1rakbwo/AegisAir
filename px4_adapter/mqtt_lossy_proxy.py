#!/usr/bin/env python3
"""PX4 command-channel loss/latency proxy.

This is the PX4-adapter equivalent of the older SafeDrones ``mqtt_lossy_proxy``.
It subscribes to an upstream command topic, applies deterministic loss and
latency, and republishes to the adapter-native command topic.

The upstream topic is intentionally distinct (``px4_raw/{id}/command``) so the
proxy sits between a high-level command source and the adapter instead of
letting the adapter subscribe to both raw and proxied topics.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

def main() -> None:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: paho-mqtt") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--instance", type=int, default=2)
    parser.add_argument("--upstream-topic", default=None)
    parser.add_argument("--downstream-topic", default=None)
    parser.add_argument("--loss-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--event-log", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.loss_rate <= 1.0:
        raise SystemExit("--loss-rate must be between 0 and 1")
    if args.latency_ms < 0:
        raise SystemExit("--latency-ms must be non-negative")

    upstream = args.upstream_topic or f"px4_raw/{args.instance}/command"
    downstream = args.downstream_topic or f"px4/{args.instance}/command"
    rng = random.Random(args.seed)
    event_handle = args.event_log.open("a", encoding="utf-8") if args.event_log else None

    def on_connect(client, userdata, flags, rc, props=None):
        client.subscribe(upstream)

    def on_message(client, userdata, msg):
        if rng.random() < args.loss_rate:
            if event_handle:
                event_handle.write(
                    json.dumps(
                        {
                            "event": "dropped",
                            "upstream_topic": upstream,
                            "loss_rate": args.loss_rate,
                            "latency_ms": args.latency_ms,
                            "timestamp_ms": int(time.time() * 1000),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                event_handle.flush()
            return

        if args.latency_ms > 0:
            time.sleep(args.latency_ms / 1000.0)
        client.publish(downstream, msg.payload)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"px4-lossy-proxy-{args.instance}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()
