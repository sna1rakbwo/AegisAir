from __future__ import annotations

import unittest

from px4_adapter.mqtt_codec import (
    TelemetryState,
    base_topics,
    decode_command,
    encode_command_ack,
    encode_safedrones_telemetry,
    encode_stale_state,
    encode_telemetry,
    normalize_command_to_ned,
)


class MqttCodecTest(unittest.TestCase):
    def test_base_topics_are_instance_scoped(self) -> None:
        self.assertEqual(
            base_topics("px4", 2),
            {
                "telemetry": "px4/2/telemetry",
                "state": "px4/2/state",
                "command": "px4/2/command",
                "ack": "px4/2/command_ack",
            },
        )

    def test_decode_move_to_uses_target(self) -> None:
        command = decode_command(
            {
                "drone": 2,
                "action": "move_to",
                "target": [1.0, 2.0, -3.0],
                "velocity": [0.5, 0.0, 0.0],
                "ttl_sec": 1.0,
                "command_id": "c1",
                "timestamp_ms": 1000,
            }
        )
        self.assertEqual(command.action, "move_to")
        self.assertEqual(command.target, (1.0, 2.0, -3.0))
        self.assertEqual(command.velocity, (0.5, 0.0, 0.0))
        self.assertEqual(command.drone, 2)

    def test_decode_move_to_accepts_waypoint_fallback(self) -> None:
        command = decode_command({"drone": 2, "action": "move_to", "waypoint": [0.0, 0.0, -1.0]})
        self.assertEqual(command.target, (0.0, 0.0, -1.0))

    def test_decode_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            decode_command({"drone": 2, "action": "barrel_roll"})

    def test_decode_hover_maps_to_hold(self) -> None:
        command = decode_command({"drone": 2, "action": "hover"})
        self.assertEqual(command.action, "hold")

    def test_decode_default_source_frame(self) -> None:
        command = decode_command(
            {"drone": 2, "action": "move_to", "target": [1.0, 2.0, 3.0]},
            default_source_frame="FLU",
        )
        self.assertEqual(command.source_frame, "FLU")

    def test_normalize_flu_command_to_ned(self) -> None:
        command = decode_command(
            {"drone": 2, "action": "move_to", "target": [1.0, 2.0, 3.0]},
            default_source_frame="FLU",
        )
        normalized = normalize_command_to_ned(command)
        self.assertEqual(normalized.source_frame, "PX4_NED")
        self.assertEqual(normalized.target, (1.0, -2.0, -3.0))

    def test_decode_rejects_missing_move_to_target(self) -> None:
        with self.assertRaises(ValueError):
            decode_command({"drone": 2, "action": "move_to"})

    def test_decode_velocity_action(self) -> None:
        command = decode_command(
            {"drone": 2, "action": "velocity", "velocity": [1.0, -2.0, -0.5]}
        )
        self.assertEqual(command.action, "velocity")
        self.assertEqual(command.velocity, (1.0, -2.0, -0.5))
        self.assertIsNone(command.target)

    def test_decode_rejects_missing_velocity(self) -> None:
        with self.assertRaises(ValueError):
            decode_command({"drone": 2, "action": "velocity"})

    def test_normalize_flu_velocity_to_ned(self) -> None:
        command = decode_command(
            {"drone": 2, "action": "velocity", "velocity": [1.0, 2.0, 3.0]},
            default_source_frame="FLU",
        )
        normalized = normalize_command_to_ned(command)
        self.assertEqual(normalized.velocity, (1.0, -2.0, -3.0))

    def test_decode_takeoff_altitude(self) -> None:
        command = decode_command(
            {"drone": 2, "action": "takeoff", "altitude_m": 3.0, "ttl_sec": 0.5}
        )
        self.assertEqual(command.action, "takeoff")
        self.assertAlmostEqual(command.altitude_m, 3.0)

    def test_encode_telemetry_preserves_source_frame_and_flu(self) -> None:
        state = TelemetryState(
            instance_id=2,
            source_frame="PX4_NED",
            timestamp_ms=1234,
            source_timestamp_us=9876,
            armed=True,
            nav_state=14,
            failsafe=False,
            connection_lost=False,
            position=(1.0, 2.0, -3.0),
            velocity=(0.1, 0.2, -0.3),
            xy_valid=True,
            z_valid=True,
            v_xy_valid=True,
            v_z_valid=True,
        )
        payload = __import__("json").loads(encode_telemetry(state))
        self.assertEqual(payload["source_frame"], "PX4_NED")
        self.assertEqual(payload["position_ned"], [1.0, 2.0, -3.0])
        self.assertEqual(payload["position_flu"], [1.0, -2.0, 3.0])
        self.assertEqual(payload["velocity_flu"], [0.1, -0.2, 0.3])

    def test_encode_command_ack(self) -> None:
        import json

        payload = json.loads(
            encode_command_ack(2, "c1", False, "telemetry_stale", "LAND")
        )
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["state"], "LAND")
        self.assertEqual(payload["command_id"], "c1")

    def test_encode_stale_state(self) -> None:
        import json

        payload = json.loads(encode_stale_state(2, "PX4_NED", 1234))
        self.assertEqual(payload["status"], "stale")
        self.assertEqual(payload["source_timestamp_us"], 1234)
        self.assertEqual(payload["instance_id"], 2)

    def test_encode_safedrones_telemetry_uses_flu(self) -> None:
        import json

        state = TelemetryState(
            instance_id=2,
            source_frame="PX4_NED",
            timestamp_ms=1,
            source_timestamp_us=2,
            armed=False,
            nav_state=4,
            failsafe=False,
            connection_lost=False,
            position=(1.0, 2.0, -3.0),
            velocity=(0.1, 0.2, -0.3),
            xy_valid=True,
            z_valid=True,
            v_xy_valid=True,
            v_z_valid=True,
        )
        payload = json.loads(encode_safedrones_telemetry(state))
        self.assertEqual(payload["source_frame"], "FLU")
        self.assertEqual(payload["position"], [1.0, -2.0, 3.0])
        self.assertEqual(payload["velocity"], [0.1, -0.2, 0.3])
        self.assertEqual(payload["drone"], 2)


if __name__ == "__main__":
    unittest.main()
