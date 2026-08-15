import unittest
from math import dist

from swarm.safety import (
    DroneSnapshot,
    SafetyConfig,
    SafetyGate,
    build_hover_command,
    build_override_event,
    build_safety_command,
    build_status_event,
    pair_collision_risk,
)


class SafetyRiskTest(unittest.TestCase):
    def test_close_pair_has_high_risk(self) -> None:
        config = SafetyConfig(safe_distance_m=1.5)
        a = DroneSnapshot(1, (0, 0, 1), target=(5, 0, 1), speed_mps=1)
        b = DroneSnapshot(2, (0.5, 0, 1), target=(-5, 0, 1), speed_mps=1)

        risk = pair_collision_risk(a, b, config)

        self.assertGreaterEqual(risk.risk, 0.75)
        self.assertEqual(risk.target_drone, 2)

    def test_far_pair_has_low_risk(self) -> None:
        config = SafetyConfig(safe_distance_m=1.5)
        a = DroneSnapshot(1, (0, 0, 1), target=(1, 0, 1), speed_mps=1)
        b = DroneSnapshot(2, (10, 0, 1), target=(11, 0, 1), speed_mps=1)

        risk = pair_collision_risk(a, b, config)

        self.assertEqual(risk.risk, 0)

    def test_approaching_pair_uses_ttc_risk(self) -> None:
        config = SafetyConfig(safe_distance_m=1.0, ttc_safe_sec=3.0)
        a = DroneSnapshot(1, (0, 0, 1), target=(10, 0, 1), speed_mps=1)
        b = DroneSnapshot(2, (2, 0, 1), target=(-10, 0, 1), speed_mps=1)

        risk = pair_collision_risk(a, b, config)

        self.assertGreater(risk.risk, 0)
        self.assertLess(risk.ttc_sec, config.ttc_safe_sec)

    def test_telemetry_velocity_overrides_target_inference(self) -> None:
        config = SafetyConfig(safe_distance_m=1.0, ttc_safe_sec=3.0)
        a = DroneSnapshot(1, (0, 0, 1), target=(10, 0, 1), velocity=(0, 0, 0), speed_mps=1)
        b = DroneSnapshot(2, (2, 0, 1), target=(-10, 0, 1), velocity=(0, 0, 0), speed_mps=1)

        risk = pair_collision_risk(a, b, config)

        self.assertEqual(risk.approach_mps, 0)
        self.assertIsNone(risk.ttc_sec)
        self.assertEqual(risk.risk, 0)


class SafetyGateTest(unittest.TestCase):
    def test_default_gate_warns_before_override_margin(self) -> None:
        decisions = SafetyGate(SafetyConfig()).evaluate(
            [
                DroneSnapshot(1, (0, 0, 1), velocity=(0, 0, 0)),
                DroneSnapshot(2, (1.0, 0, 1), velocity=(0, 0, 0)),
            ],
            now=1.0,
        )

        self.assertEqual({decision.mode for decision in decisions}, {"warning"})

    def test_default_gate_overrides_static_pair_inside_tuned_margin(self) -> None:
        decisions = SafetyGate(SafetyConfig()).evaluate(
            [
                DroneSnapshot(1, (0, 0, 1), velocity=(0, 0, 0)),
                DroneSnapshot(2, (0.45, 0, 1), velocity=(0, 0, 0)),
            ],
            now=1.0,
        )

        self.assertEqual({decision.mode for decision in decisions}, {"override"})

    def test_gate_emits_override_decision(self) -> None:
        gate = SafetyGate(SafetyConfig(safe_distance_m=1.5, high_threshold=0.75))
        snapshots = [
            DroneSnapshot(1, (0, 0, 1), target=(5, 0, 1), speed_mps=1, last_command_id="cmd-1"),
            DroneSnapshot(2, (0.5, 0, 1), target=(-5, 0, 1), speed_mps=1, last_command_id="cmd-2"),
        ]

        decisions = gate.evaluate(snapshots, now=1.0)

        self.assertEqual({decision.mode for decision in decisions}, {"override"})
        event = build_override_event(decisions[0])
        self.assertEqual(event["event"], "safety_override")
        self.assertEqual(event["event_name"], "safety_override")
        self.assertEqual(event["reason"], "collision_risk_exceeded")
        self.assertEqual(event["target_drone"], 2)
        self.assertEqual(event["command_id"], "cmd-1")

    def test_gate_override_includes_safety_waypoint_that_increases_separation(self) -> None:
        gate = SafetyGate(SafetyConfig(safe_distance_m=1.5, high_threshold=0.75, escape_distance_m=1.2))
        peer_position = (0.5, 0, 1)
        decision = gate.evaluate(
            [
                DroneSnapshot(1, (0, 0, 1), target=(5, 0, 1), speed_mps=1, last_command_id="cmd-1"),
                DroneSnapshot(2, peer_position, target=(-5, 0, 1), speed_mps=1, last_command_id="cmd-2"),
            ],
            now=1.0,
        )[0]

        self.assertIsNotNone(decision.safety_waypoint)
        self.assertGreater(dist(decision.safety_waypoint, peer_position), dist(decision.position, peer_position))

    def test_hover_command_targets_drone(self) -> None:
        decision = SafetyGate(SafetyConfig()).evaluate(
            [DroneSnapshot(1, (0, 0, 1), target=(1, 0, 1), speed_mps=1)],
            now=1.0,
        )[0]

        command = build_hover_command(decision)

        self.assertEqual(command["drone"], 1)
        self.assertEqual(command["action"], "hover")
        self.assertEqual(command["priority"], "safety")

    def test_safety_command_uses_diversion_waypoint_when_available(self) -> None:
        decision = SafetyGate(SafetyConfig(safe_distance_m=1.5, high_threshold=0.75)).evaluate(
            [
                DroneSnapshot(1, (0, 0, 1), target=(5, 0, 1), speed_mps=1, last_command_id="cmd-1"),
                DroneSnapshot(2, (0.5, 0, 1), target=(-5, 0, 1), speed_mps=1, last_command_id="cmd-2"),
            ],
            now=1.0,
        )[0]

        command = build_safety_command(decision)

        self.assertEqual(command["drone"], 1)
        self.assertEqual(command["action"], "move_to")
        self.assertEqual(command["priority"], "safety")
        self.assertEqual(command["target"], list(decision.safety_waypoint))

    def test_status_event_exposes_visualization_mode(self) -> None:
        gate = SafetyGate(SafetyConfig(safe_distance_m=1.5, high_threshold=0.75))
        decision = gate.evaluate(
            [
                DroneSnapshot(1, (0, 0, 1), target=(5, 0, 1), speed_mps=1),
                DroneSnapshot(2, (0.5, 0, 1), target=(-5, 0, 1), speed_mps=1),
            ],
            now=1.0,
        )[0]

        event = build_status_event(decision)

        self.assertEqual(event["event"], "safety_status")
        self.assertEqual(event["event_name"], "safety_status")
        self.assertEqual(event["mode"], "override")
        self.assertEqual(event["status"], "safety_override")
        self.assertEqual(event["anomalies"], ["collision_risk"])


if __name__ == "__main__":
    unittest.main()
