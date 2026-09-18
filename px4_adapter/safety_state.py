"""Local adapter safety state machine.

The state machine is a fail-closed boundary between SafeDrones' task-level
intent and PX4.  It has no ROS 2 or MQTT dependencies and must never consult
Gazebo ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from px4_adapter.mqtt_codec import Px4Command, TelemetryState


@dataclass(frozen=True)
class SafetyLimits:
    command_ttl_sec: float
    telemetry_stale_sec: float
    max_horizontal_speed_mps: float
    max_vertical_speed_mps: float
    min_altitude_m: float
    max_altitude_m: float
    max_position_distance_m: float
    fail_closed_action: str = "LAND"


@dataclass(frozen=True)
class SafetyDecision:
    state: str
    allowed: bool
    reason: str
    command_id: str | None


def _finite(value: Any, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _reject(state: str, reason: str, command_id: str | None) -> SafetyDecision:
    return SafetyDecision(state=state, allowed=False, reason=reason, command_id=command_id)


class LocalSafetyState:
    """Deterministic local safety gate for one PX4 instance."""

    def __init__(self, limits: SafetyLimits):
        if limits.command_ttl_sec < 0:
            raise ValueError("command_ttl_sec must be non-negative")
        if limits.telemetry_stale_sec <= 0:
            raise ValueError("telemetry_stale_sec must be positive")
        if limits.min_altitude_m > limits.max_altitude_m:
            raise ValueError("min_altitude_m must be <= max_altitude_m")
        if limits.max_position_distance_m < 0:
            raise ValueError("max_position_distance_m must be non-negative")
        self.limits = limits

    def evaluate(
        self,
        now_ms: int,
        telemetry: TelemetryState | None,
        command: Px4Command | None,
    ) -> SafetyDecision:
        command_id = command.command_id if command is not None else None

        if telemetry is None:
            return _reject(self.limits.fail_closed_action, "telemetry_missing", command_id)

        if now_ms - telemetry.timestamp_ms > self.limits.telemetry_stale_sec * 1000.0:
            return _reject(self.limits.fail_closed_action, "telemetry_stale", command_id)

        if telemetry.failsafe:
            return _reject("RTL", "px4_failsafe", command_id)

        if telemetry.connection_lost:
            return _reject("LAND", "gcs_connection_lost", command_id)

        if command is None:
            return SafetyDecision("NORMAL", False, "no_command", None)

        if now_ms - command.timestamp_ms > command.ttl_sec * 1000.0:
            return _reject("LOCAL_HOLD", "command_expired", command_id)

        if command.drone != telemetry.instance_id:
            return _reject("LOCAL_HOLD", "command_drone_mismatch", command_id)

        if command.action == "move_to":
            if command.target is None:
                return _reject("LOCAL_HOLD", "missing_target", command_id)

            horizontal = math.hypot(command.target[0], command.target[1])
            altitude = -command.target[2]
            if not (
                self.limits.min_altitude_m <= altitude <= self.limits.max_altitude_m
            ):
                return _reject("LOCAL_HOLD", "altitude_out_of_limits", command_id)
            if horizontal > self.limits.max_position_distance_m:
                return _reject("LOCAL_HOLD", "horizontal_distance_out_of_limits", command_id)

            if command.velocity is not None:
                horizontal_speed = math.hypot(command.velocity[0], command.velocity[1])
                vertical_speed = abs(command.velocity[2])
                if horizontal_speed > self.limits.max_horizontal_speed_mps:
                    return _reject("LOCAL_HOLD", "horizontal_speed_out_of_limits", command_id)
                if vertical_speed > self.limits.max_vertical_speed_mps:
                    return _reject("LOCAL_HOLD", "vertical_speed_out_of_limits", command_id)

        if command.action == "velocity":
            if command.velocity is None:
                return _reject("LOCAL_HOLD", "missing_velocity", command_id)
            horizontal_speed = math.hypot(command.velocity[0], command.velocity[1])
            vertical_speed = abs(command.velocity[2])
            if horizontal_speed > self.limits.max_horizontal_speed_mps:
                return _reject("LOCAL_HOLD", "horizontal_speed_out_of_limits", command_id)
            if vertical_speed > self.limits.max_vertical_speed_mps:
                return _reject("LOCAL_HOLD", "vertical_speed_out_of_limits", command_id)

        if command.action == "takeoff" and command.altitude_m is not None:
            if not (
                self.limits.min_altitude_m
                <= command.altitude_m
                <= self.limits.max_altitude_m
            ):
                return _reject("LOCAL_HOLD", "altitude_out_of_limits", command_id)

        return SafetyDecision("NORMAL", True, "ok", command_id)
