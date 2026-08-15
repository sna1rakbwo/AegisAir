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
    LogEvent,
    MarlAction,
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


class JsonSchemaGenerationTest(unittest.TestCase):
    def test_all_models_emit_json_schema(self) -> None:
        for name, model in MODELS.items():
            with self.subTest(name=name):
                schema = model.model_json_schema()
                self.assertEqual(schema.get("type"), "object")
                self.assertIn("properties", schema)


if __name__ == "__main__":
    unittest.main()
