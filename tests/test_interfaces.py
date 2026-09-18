"""Phase 0 frozen-interface tests.

These tests are the executable part of the interface freeze: every example
payload must validate, and breaking the declared fields must fail.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from swarm.interfaces import (
    CoordinationAdmissionDecision,
    CoordinationReservationDecision,
    CoordinationSpaceTimeReservationDecision,
    ExecutionAssuranceDecision,
    LogEvent,
    MarlAction,
    MissionDecision,
    Observation,
    RecoveryPlan,
    RiskEvent,
    SafetyDecision,
    Telemetry,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "interfaces"

MODELS = {
    "telemetry": Telemetry,
    "observation": Observation,
    "marl_action": MarlAction,
    "risk_event": RiskEvent,
    "recovery_plan": RecoveryPlan,
    "safety_decision": SafetyDecision,
    "execution_assurance": ExecutionAssuranceDecision,
    "coordination_admission": CoordinationAdmissionDecision,
    "coordination_reservation": CoordinationReservationDecision,
    "coordination_space_time_reservation": CoordinationSpaceTimeReservationDecision,
    "log_event": LogEvent,
}


class InterfaceValidationTest(unittest.TestCase):
    def test_example_payloads_validate(self) -> None:
        for name, model in MODELS.items():
            path = EXAMPLES / f"{name}.json"
            payload = json.loads(path.read_text())
            with self.subTest(name=name):
                instance = model.model_validate(payload)
                self.assertIsNotNone(instance)

    def test_unknown_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Telemetry.model_validate(
                {
                    "schema_version": 1,
                    "drone": 2,
                    "position": [0.0, 0.0, 1.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "status": "armed",
                    "armed": True,
                    "nav_state": 14,
                    "failsafe": False,
                    "connection_lost": False,
                    "source_timestamp_us": 1,
                    "timestamp_ms": 1,
                    "unexpected_field": 123,
                }
            )

    def test_marl_action_v1_is_2d(self) -> None:
        with self.assertRaises(ValidationError):
            MarlAction.model_validate(
                {
                    "schema_version": 1,
                    "agent_id": 2,
                    "velocity_cmd": [1.0, 0.0, 0.0],
                    "timestamp_ms": 1,
                }
            )

    def test_recovery_action_is_whitelisted(self) -> None:
        with self.assertRaises(ValidationError):
            RecoveryPlan.model_validate(
                {
                    "schema_version": 1,
                    "intent_text": "bad",
                    "commands": [
                        {
                            "drone": 1,
                            "action": "NOT_ALLOWED",
                            "ttl_sec": 1.0,
                            "command_id": "x",
                        }
                    ],
                    "timestamp_ms": 1,
                }
            )

    def test_risk_event_enum_is_restricted(self) -> None:
        with self.assertRaises(ValidationError):
            RiskEvent.model_validate(
                {
                    "schema_version": 1,
                    "event": "PREDICTED_CONFLICT",
                    "agent_i": 1,
                    "agent_j": 2,
                    "current_margin": 0.5,
                    "cause": "NOT_A_CAUSE",
                    "severity": "HIGH",
                    "timestamp_ms": 1,
                }
            )

    def test_repeated_route_conflict_event_validates(self) -> None:
        event = RiskEvent.model_validate(
            {
                "schema_version": 1,
                "event": "REPEATED_ROUTE_CONFLICT",
                "agent_i": 2,
                "agent_j": 4,
                "current_margin": 0.3,
                "intervention_count": 7,
                "mission_priority": {2: "high", 4: "normal"},
                "cause": "REPEATED_CONFLICT",
                "severity": "HIGH",
                "timestamp_ms": 1,
            }
        )
        self.assertEqual(event.event, "REPEATED_ROUTE_CONFLICT")
        self.assertEqual(event.cause, "REPEATED_CONFLICT")
        self.assertEqual(event.mission_priority, {2: "high", 4: "normal"})

    def test_mission_plan_invalidated_event_validates(self) -> None:
        event = RiskEvent.model_validate(
            {
                "schema_version": 1,
                "event": "MISSION_PLAN_INVALIDATED",
                "agent_i": 3,
                "agent_j": 3,
                "current_margin": 0.8,
                "cause": "ROUTE_DEVIATION",
                "severity": "MEDIUM",
                "timestamp_ms": 1,
            }
        )
        self.assertEqual(event.event, "MISSION_PLAN_INVALIDATED")
        self.assertEqual(event.cause, "ROUTE_DEVIATION")

    def test_compact_mission_decision_validates(self) -> None:
        decision = MissionDecision.model_validate({"action": "REASSIGN", "agent": 1})
        self.assertEqual(decision.action, "REASSIGN")
        self.assertEqual(decision.agent, 1)

    def test_execution_assurance_decision_validates(self) -> None:
        decision = ExecutionAssuranceDecision.model_validate(
            {
                "schema_version": 1,
                "active": True,
                "reasons": ["VELOCITY_RESIDUAL"],
                "residuals": {
                    2: {"dt_s": 0.05, "velocity_mps": 0.21, "position_m": 0.03}
                },
                "recoverability": [
                    {
                        "drones": [2, 3],
                        "separation_m": 4.0,
                        "closing_speed_mps": 3.0,
                        "required_distance_m": 4.75,
                        "margin_m": -0.75,
                    }
                ],
                "backup_drones": [3],
                "qp_feasible": True,
                "solve_elapsed_s": 0.004,
                "telemetry_ages_s": {2: 0.01},
                "consecutive_bad": 2,
                "consecutive_clean": 0,
                "timestamp_ms": 1,
            }
        )
        self.assertTrue(decision.active)

    def test_coordination_admission_decision_validates(self) -> None:
        decision = CoordinationAdmissionDecision.model_validate(
            {
                "schema_version": 1,
                "step": 12,
                "coordination_mode": "pass",
                "admission_triggers": ["conflict_zone", "predicted_rho_low"],
                "authorized_drone_ids": [2],
                "held_drone_ids": [3, 4, 5],
                "frozen_hold_goals": {
                    3: [4.2, -1.1, 2.5],
                    4: [-4.2, -1.1, 2.5],
                    5: [4.2, 1.1, 2.5],
                },
                "hold_enter_steps": {3: 10, 4: 10, 5: 10},
                "conflict_zone_id": "crossing_center",
                "ra_vetoed": False,
                "authority_source": "c3_rule",
                "timestamp_ms": 1,
            }
        )
        self.assertEqual(decision.authorized_drone_ids, [2])
        self.assertEqual(decision.coordination_mode, "pass")

    def test_coordination_admission_rejects_direct_command_field(self) -> None:
        with self.assertRaises(ValidationError):
            CoordinationAdmissionDecision.model_validate(
                {
                    "step": 1,
                    "coordination_mode": "normal",
                    "authority_source": "c3_rule",
                    "timestamp_ms": 1,
                    "velocity_cmd": [1.0, 0.0],
                }
            )

    def test_coordination_reservation_decision_validates(self) -> None:
        decision = CoordinationReservationDecision.model_validate(
            {
                "step": 80,
                "coordination_mode": "clear",
                "admission_triggers": ["conflict_zone", "predicted_rho_low"],
                "authorized_drone_ids": [2],
                "held_drone_ids": [3, 4, 5],
                "frozen_hold_goals": {3: [4.5, -1.1, 2.5]},
                "clearance_drone_id": 2,
                "clearance_goal": [3.0, -3.0, 2.5],
                "service_latched_drone_ids": [2],
                "cleared_drone_ids": [],
                "final_returned_drone_ids": [],
                "conflict_zone_id": "crossing_center",
                "ra_vetoed": False,
                "ra_veto_streak": 0,
                "authority_source": "c3_rule",
                "timestamp_ms": 1,
            }
        )
        self.assertEqual(decision.coordination_mode, "clear")
        self.assertEqual(decision.clearance_drone_id, 2)

    def test_coordination_reservation_rejects_direct_command_field(self) -> None:
        with self.assertRaises(ValidationError):
            CoordinationReservationDecision.model_validate(
                {
                    "step": 1,
                    "coordination_mode": "normal",
                    "authority_source": "c3_rule",
                    "timestamp_ms": 1,
                    "actuator_command": [1.0, 0.0],
                }
            )

    def test_coordination_space_time_reservation_validates(self) -> None:
        decision = CoordinationSpaceTimeReservationDecision.model_validate(
            {
                "step": 120,
                "coordination_mode": "pass",
                "admission_triggers": ["conflict_zone"],
                "authorized_drone_ids": [2],
                "held_drone_ids": [3, 4, 5],
                "frozen_hold_goals": {3: [4.5, -1.1, 2.5]},
                "committed_waypoints": {2: [[3.0, -0.8, 2.5], [3.0, -3.0, 2.5]]},
                "reservation_windows": [
                    {
                        "slot_id": "service-00-uav2",
                        "phase": "service",
                        "drone_ids": [2],
                        "planned_open_step": 100,
                        "planned_close_step": 300,
                        "actual_open_step": 120,
                        "status": "open",
                        "delay_steps": 0,
                    }
                ],
                "active_slot_id": "service-00-uav2",
                "service_latched_drone_ids": [],
                "cleared_drone_ids": [],
                "final_returned_drone_ids": [],
                "conflict_zone_id": "crossing_center",
                "ra_vetoed": False,
                "schedule_revision": 0,
                "total_delay_steps": 0,
                "authority_source": "c3_rule",
                "timestamp_ms": 1,
            }
        )
        self.assertEqual(decision.active_slot_id, "service-00-uav2")
        self.assertEqual(decision.reservation_windows[0].status, "open")

    def test_coordination_space_time_reservation_rejects_actuator_command(self) -> None:
        with self.assertRaises(ValidationError):
            CoordinationSpaceTimeReservationDecision.model_validate(
                {
                    "step": 1,
                    "coordination_mode": "normal",
                    "authority_source": "c3_rule",
                    "timestamp_ms": 1,
                    "velocity_cmd": [1.0, 0.0],
                }
            )


class JsonSchemaGenerationTest(unittest.TestCase):
    def test_all_models_emit_json_schema(self) -> None:
        for name, model in MODELS.items():
            with self.subTest(name=name):
                schema = model.model_json_schema()
                self.assertEqual(schema.get("type"), "object")
                self.assertIn("properties", schema)


if __name__ == "__main__":
    unittest.main()
