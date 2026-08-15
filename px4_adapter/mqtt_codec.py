"""MQTT topic and payload codecs for the PX4 adapter.

This module is intentionally free of ROS 2 and paho-mqtt imports so the
mapping can be unit-tested on the host without a live runtime.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from px4_adapter.px4_codec import flu_to_px4_ned, px4_ned_to_flu


VALID_ACTIONS = {"arm", "takeoff", "move_to", "velocity", "land", "rtl", "hold"}


def _as_vector3(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of three numbers")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numbers") from exc


def _finite(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def base_topics(base_topic: str, instance_id: int) -> dict[str, str]:
    """Return the adapter's default MQTT topics for one PX4 instance."""
    prefix = f"{base_topic}/{instance_id}"
    return {
        "telemetry": f"{prefix}/telemetry",
        "state": f"{prefix}/state",
        "command": f"{prefix}/command",
        "ack": f"{prefix}/command_ack",
    }


@dataclass(frozen=True)
class TelemetryState:
    """A normalized, source-frame-preserving PX4 telemetry snapshot."""

    instance_id: int
    source_frame: str
    timestamp_ms: int
    source_timestamp_us: int
    armed: bool
    nav_state: int
    failsafe: bool
    connection_lost: bool
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    xy_valid: bool
    z_valid: bool
    v_xy_valid: bool
    v_z_valid: bool
    yaw: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        position_flu = px4_ned_to_flu(*self.position)
        velocity_flu = px4_ned_to_flu(*self.velocity)
        return {
            "instance_id": self.instance_id,
            "drone": self.instance_id,
            "source_frame": self.source_frame,
            "timestamp_ms": self.timestamp_ms,
            "source_timestamp_us": self.source_timestamp_us,
            "position_ned": list(self.position),
            "velocity_ned": list(self.velocity),
            "position_flu": list(position_flu),
            "velocity_flu": list(velocity_flu),
            "yaw_deg": math.degrees(self.yaw),
            "armed": self.armed,
            "nav_state": self.nav_state,
            "failsafe": self.failsafe,
            "connection_lost": self.connection_lost,
            "xy_valid": self.xy_valid,
            "z_valid": self.z_valid,
            "v_xy_valid": self.v_xy_valid,
            "v_z_valid": self.v_z_valid,
            "status": self._status(),
        }

    def _status(self) -> str:
        if self.failsafe:
            return "failsafe"
        if self.connection_lost:
            return "gcs_connection_lost"
        if self.armed:
            return "armed"
        return "disarmed"


def encode_telemetry(state: TelemetryState) -> str:
    return json.dumps(state.as_dict(), ensure_ascii=False, separators=(",", ":"))


def encode_safedrones_telemetry(state: TelemetryState) -> str:
    """Publish PX4 state in the SafeDrones `swarm/drone/{id}/telemetry` shape.

    SafeDrones internally uses z-up coordinates, so the FLU projection is used
    here while the adapter-native topic keeps the raw PX4 NED projection.
    """
    data = state.as_dict()
    position = data["position_flu"]
    velocity = data["velocity_flu"]
    return json.dumps(
        {
            "drone": state.instance_id,
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "position": position,
            "velocity": velocity,
            "yaw_deg": data["yaw_deg"],
            "status": data["status"],
            "armed": state.armed,
            "nav_state": state.nav_state,
            "failsafe": state.failsafe,
            "connection_lost": state.connection_lost,
            "source_frame": "FLU",
            "source_timestamp_us": state.source_timestamp_us,
            "timestamp_ms": state.timestamp_ms,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def encode_stale_state(
    instance_id: int,
    source_frame: str,
    source_timestamp_us: int | None,
) -> str:
    """Publish an explicit stale marker instead of re-aging cached PX4 state."""
    return json.dumps(
        {
            "instance_id": instance_id,
            "source_frame": source_frame,
            "source_timestamp_us": source_timestamp_us,
            "status": "stale",
            "timestamp_ms": _now_ms(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class Px4Command:
    """A validated high-level command decoded from an MQTT payload."""

    drone: int
    action: str
    target: tuple[float, float, float] | None
    velocity: tuple[float, float, float] | None
    altitude_m: float | None
    yaw: float
    ttl_sec: float
    command_id: str
    timestamp_ms: int
    source_frame: str
    priority: str


def normalize_command_to_ned(command: Px4Command) -> Px4Command:
    """Return an internal PX4-NED command from any declared source frame."""
    if command.source_frame == "PX4_NED":
        return command

    target = (
        flu_to_px4_ned(*command.target) if command.target is not None else None
    )
    velocity = (
        flu_to_px4_ned(*command.velocity) if command.velocity is not None else None
    )
    return replace(
        command,
        target=target,
        velocity=velocity,
        source_frame="PX4_NED",
    )


def decode_command(
    payload: dict[str, Any],
    default_source_frame: str = "PX4_NED",
) -> Px4Command:
    """Decode and validate a SafeDrones-style command payload.

    Raises ``ValueError`` for malformed, unknown, or out-of-range commands so
    the adapter can fail-closed before touching PX4.
    """
    if not isinstance(payload, dict):
        raise ValueError("command payload must be an object")

    try:
        drone = int(payload["drone"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("command payload is missing integer field 'drone'") from exc

    action = str(payload.get("action") or "hold").strip().lower()
    if action in {"hover", "brake"}:
        action = "hold"
    if action not in VALID_ACTIONS:
        raise ValueError(f"unsupported command action: {action!r}")

    target: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    if action == "move_to":
        raw_target = payload.get("target", payload.get("waypoint"))
        if raw_target is None:
            raise ValueError("move_to requires a 'target' or 'waypoint'")
        target = _as_vector3(raw_target, "target")
        if payload.get("velocity") is not None:
            velocity = _as_vector3(payload["velocity"], "velocity")
    elif action == "velocity":
        raw_velocity = payload.get("velocity")
        if raw_velocity is None:
            raise ValueError("velocity requires a 'velocity' vector")
        velocity = _as_vector3(raw_velocity, "velocity")
    elif action == "takeoff":
        raw_target = payload.get("target", payload.get("waypoint"))
        if raw_target is not None:
            target = _as_vector3(raw_target, "target")

    altitude_m: float | None = None
    if action == "takeoff" and payload.get("altitude_m") is not None:
        altitude_m = _finite(payload["altitude_m"], "altitude_m")

    try:
        ttl_sec = _finite(payload.get("ttl_sec", 1.0), "ttl_sec")
    except (TypeError, ValueError) as exc:
        raise ValueError("ttl_sec must be a finite number") from exc
    if ttl_sec < 0:
        raise ValueError("ttl_sec must be non-negative")

    yaw = _finite(payload.get("yaw", 0.0), "yaw")
    source_frame = str(payload.get("source_frame") or default_source_frame)
    priority = str(payload.get("priority") or "normal").lower()
    command_id = str(payload.get("command_id") or "")
    timestamp_ms = int(payload.get("timestamp_ms") or _now_ms())

    return Px4Command(
        drone=drone,
        action=action,
        target=target,
        velocity=velocity,
        altitude_m=altitude_m,
        yaw=yaw,
        ttl_sec=ttl_sec,
        command_id=command_id,
        timestamp_ms=timestamp_ms,
        source_frame=source_frame,
        priority=priority,
    )


def encode_command_ack(
    instance_id: int,
    command_id: str,
    accepted: bool,
    reason: str,
    state: str,
) -> str:
    return json.dumps(
        {
            "instance_id": instance_id,
            "command_id": command_id,
            "accepted": bool(accepted),
            "reason": reason,
            "state": state,
            "timestamp_ms": _now_ms(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
