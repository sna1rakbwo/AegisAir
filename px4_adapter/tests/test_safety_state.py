from __future__ import annotations

import unittest

from px4_adapter.mqtt_codec import Px4Command, TelemetryState
from px4_adapter.safety_state import LocalSafetyState, SafetyLimits


def limits() -> SafetyLimits:
    return SafetyLimits(
        command_ttl_sec=1.0,
        telemetry_stale_sec=1.5,
        max_horizontal_speed_mps=3.0,
        max_vertical_speed_mps=2.0,
        min_altitude_m=0.0,
        max_altitude_m=10.0,
        max_position_distance_m=30.0,
        fail_closed_action="LAND",
    )


def telemetry(*, timestamp_ms: int = 0, **overrides: object) -> TelemetryState:
    values = {
        "instance_id": 2,
        "source_frame": "PX4_NED",
        "timestamp_ms": timestamp_ms,
        "source_timestamp_us": timestamp_ms * 1000,
        "armed": False,
        "nav_state": 4,
        "failsafe": False,
        "connection_lost": False,
        "position": (0.0, 0.0, 0.0),
        "velocity": (0.0, 0.0, 0.0),
        "xy_valid": True,
        "z_valid": True,
        "v_xy_valid": True,
        "v_z_valid": True,
    }
    values.update(overrides)
    return TelemetryState(**values)  # type: ignore[arg-type]


def command(action: str = "hold", *, timestamp_ms: int = 0, **overrides: object) -> Px4Command:
    values = {
        "drone": 2,
        "action": action,
        "target": None,
        "velocity": None,
        "altitude_m": None,
        "yaw": 0.0,
        "ttl_sec": 1.0,
        "command_id": "c1",
        "timestamp_ms": timestamp_ms,
        "source_frame": "PX4_NED",
        "priority": "normal",
    }
    values.update(overrides)
    return Px4Command(**values)  # type: ignore[arg-type]


class SafetyStateTest(unittest.TestCase):
    def test_missing_telemetry_fails_closed(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(0, None, None)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "LAND")
        self.assertEqual(decision.reason, "telemetry_missing")

    def test_stale_telemetry_fails_closed(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            2000,
            telemetry(timestamp_ms=0),
            None,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "telemetry_stale")

    def test_px4_failsafe_is_rtl(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            0,
            telemetry(timestamp_ms=0, failsafe=True),
            None,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "RTL")

    def test_gcs_connection_lost_is_land(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            0,
            telemetry(timestamp_ms=0, connection_lost=True),
            None,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.state, "LAND")

    def test_command_expired_is_rejected(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            1500,
            telemetry(timestamp_ms=0),
            command("arm", timestamp_ms=0, ttl_sec=1.0),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "command_expired")

    def test_valid_move_to_is_allowed(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            100,
            telemetry(timestamp_ms=0),
            command("move_to", timestamp_ms=0, target=(5.0, 0.0, -2.0)),
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.state, "NORMAL")

    def test_altitude_out_of_limits_is_rejected(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            100,
            telemetry(timestamp_ms=0),
            command("move_to", timestamp_ms=0, target=(0.0, 0.0, -20.0)),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "altitude_out_of_limits")

    def test_horizontal_speed_out_of_limits_is_rejected(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            100,
            telemetry(timestamp_ms=0),
            command(
                "move_to",
                timestamp_ms=0,
                target=(1.0, 0.0, -1.0),
                velocity=(5.0, 0.0, 0.0),
            ),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "horizontal_speed_out_of_limits")

    def test_takeoff_altitude_out_of_limits_is_rejected(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            100,
            telemetry(timestamp_ms=0),
            command("takeoff", timestamp_ms=0, altitude_m=20.0),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "altitude_out_of_limits")

    def test_command_drone_mismatch_is_rejected(self) -> None:
        decision = LocalSafetyState(limits()).evaluate(
            100,
            telemetry(timestamp_ms=0),
            command("hold", timestamp_ms=0, drone=3),
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "command_drone_mismatch")


if __name__ == "__main__":
    unittest.main()
