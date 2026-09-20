"""Pure tests for adapter node watchdog helpers."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from px4_adapter.mqtt_codec import TelemetryState
from px4_adapter.node import MqttBridge, _local_hold_position, _measurement_timestamp_ms


class LocalHoldPositionTest(unittest.TestCase):
    def test_watchdog_holds_current_local_position(self) -> None:
        state = TelemetryState(
            instance_id=2,
            source_frame="PX4_NED",
            timestamp_ms=1000,
            source_timestamp_us=1_000_000,
            armed=True,
            nav_state=14,
            failsafe=False,
            connection_lost=False,
            position=(1.0, -2.0, -2.5),
            velocity=(0.0, 0.0, 0.0),
            xy_valid=True,
            z_valid=True,
            v_xy_valid=True,
            v_z_valid=True,
        )
        self.assertEqual(
            _local_hold_position(state, (9.0, 9.0, -2.5)),
            state.position,
        )

    def test_watchdog_falls_back_when_local_position_is_invalid(self) -> None:
        state = TelemetryState(
            instance_id=2,
            source_frame="PX4_NED",
            timestamp_ms=1000,
            source_timestamp_us=1_000_000,
            armed=True,
            nav_state=14,
            failsafe=False,
            connection_lost=False,
            position=(1.0, -2.0, -2.5),
            velocity=(0.0, 0.0, 0.0),
            xy_valid=False,
            z_valid=True,
            v_xy_valid=True,
            v_z_valid=True,
        )
        fallback = (9.0, 9.0, -2.5)
        self.assertEqual(_local_hold_position(state, fallback), fallback)


class MeasurementTimestampTest(unittest.TestCase):
    def test_position_arrival_time_is_preserved(self) -> None:
        self.assertEqual(
            _measurement_timestamp_ms(
                now_ms=1100,
                position_arrival_ms=1000,
                status_arrival_ms=1090,
                stale_after_s=1.5,
            ),
            1000,
        )

    def test_fresh_status_does_not_refresh_stale_position(self) -> None:
        self.assertIsNone(
            _measurement_timestamp_ms(
                now_ms=3000,
                position_arrival_ms=1000,
                status_arrival_ms=2990,
                stale_after_s=1.5,
            )
        )


class MqttBridgePublishTest(unittest.TestCase):
    def test_publish_enqueues_without_blocking_the_ros_executor(self) -> None:
        result = Mock(rc=0)
        client = Mock()
        client.publish.return_value = result
        bridge = object.__new__(MqttBridge)
        bridge.config = {"mqtt": {"qos": 0}}
        bridge._client = client

        bridge.publish("px4/2/state", "payload")

        client.publish.assert_called_once_with(
            "px4/2/state",
            payload="payload",
            qos=0,
            retain=False,
        )
        result.wait_for_publish.assert_not_called()

    def test_publish_rejects_an_enqueue_error(self) -> None:
        client = Mock()
        client.publish.return_value = Mock(rc=4)
        bridge = object.__new__(MqttBridge)
        bridge.config = {"mqtt": {"qos": 0}}
        bridge._client = client

        with self.assertRaisesRegex(RuntimeError, "rc=4"):
            bridge.publish("px4/2/state", "payload")


if __name__ == "__main__":
    unittest.main()
