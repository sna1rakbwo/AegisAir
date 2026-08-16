"""Frozen tests for the Phase 5 command path (no PX4 runtime required)."""

from __future__ import annotations

import unittest

from marllib.phase5_runner import (
    CRUISE_ALTITUDE_M,
    _priority_order,
    build_phase5_command,
    build_phase5_velocity_command,
    flu_snapshot_to_telemetry_state,
    snapshot_from_telemetry,
    validate_command_path,
)
from px4_adapter.mqtt_codec import decode_command, normalize_command_to_ned


class Phase5CommandEncodingTest(unittest.TestCase):
    def test_priority_order_puts_urgent_first(self) -> None:
        self.assertEqual(_priority_order([2, 3, 4, 5], urgent_drone=4), [4, 2, 3, 5])
        self.assertEqual(_priority_order([2, 3, 4, 5]), [2, 3, 4, 5])

    def test_build_phase5_command_integrates_safe_velocity(self) -> None:
        command = build_phase5_command(
            drone=2,
            safe_velocity=(1.0, -0.5),
            position_flu=(1.0, 2.0, 0.0),
            timestamp_ms=123,
        )
        self.assertEqual(command["action"], "move_to")
        self.assertEqual(command["source_frame"], "FLU")
        # target = position + velocity * 0.5, altitude clamped to cruise alt.
        self.assertEqual(command["target"], [1.5, 1.75, CRUISE_ALTITUDE_M])
        self.assertEqual(command["drone"], 2)

    def test_command_decodes_and_normalizes_to_ned(self) -> None:
        command = build_phase5_command(
            drone=3,
            safe_velocity=(0.0, 1.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=1,
        )
        decoded = normalize_command_to_ned(
            decode_command(command, default_source_frame="PX4_NED")
        )
        # FLU target y = 0 + 1.0*0.5 = +0.5 maps to NED y=-0.5;
        # FLU altitude +2.5 maps to NED z=-2.5.
        self.assertEqual(decoded.source_frame, "PX4_NED")
        self.assertAlmostEqual(decoded.target[1], -0.5)
        self.assertAlmostEqual(decoded.target[2], -CRUISE_ALTITUDE_M)

    def test_build_phase5_velocity_command(self) -> None:
        command = build_phase5_velocity_command(
            drone=2,
            safe_velocity=(1.0, -0.5),
            vertical_velocity=0.2,
            timestamp_ms=123,
        )
        self.assertEqual(command["action"], "velocity")
        self.assertEqual(command["source_frame"], "FLU")
        self.assertEqual(command["velocity"], [1.0, -0.5, 0.2])

    def test_velocity_command_passes_local_safety(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_velocity_command(
            drone=2,
            safe_velocity=(1.0, 0.0),
            vertical_velocity=0.1,
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertTrue(allowed, reason)
        self.assertEqual(reason, "ok")

    def test_normal_command_passes_local_safety(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_command(
            drone=2,
            safe_velocity=(1.0, 0.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertTrue(allowed, reason)
        self.assertEqual(reason, "ok")

    def test_out_of_limits_command_is_rejected(self) -> None:
        timestamp_ms = 1000
        command = build_phase5_command(
            drone=2,
            safe_velocity=(100.0, 0.0),
            position_flu=(0.0, 0.0, CRUISE_ALTITUDE_M),
            timestamp_ms=timestamp_ms,
        )
        state = flu_snapshot_to_telemetry_state(
            2,
            (0.0, 0.0, CRUISE_ALTITUDE_M),
            (0.0, 0.0, 0.0),
            timestamp_ms=timestamp_ms,
        )
        allowed, reason = validate_command_path(
            command, state, now_ms=timestamp_ms
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "horizontal_distance_out_of_limits")

    def test_telemetry_payload_parses_flu_position(self) -> None:
        payload = {
            "drone": 2,
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.2, 0.3],
            "timestamp_ms": 7,
            "status": "armed",
        }
        snapshot = snapshot_from_telemetry(payload)
        self.assertEqual(snapshot.drone_id, 2)
        self.assertEqual(snapshot.position, (1.0, 2.0, 3.0))
        self.assertEqual(snapshot.velocity, (0.1, 0.2, 0.3))


if __name__ == "__main__":
    unittest.main()
