"""Phase 4 semantic-recovery layer tests."""

from __future__ import annotations

import unittest

from swarm.interfaces import RecoveryCommand, RecoveryPlan
from swarm.ra.runtime_assurance import FilterResult
from swarm.recovery import (
    DeterministicRecoveryClient,
    MlxLmClient,
    RecoveryConfig,
    RecoveryValidator,
    SemanticRecovery,
)
from swarm.recovery.executor import ActiveRecovery, apply_plan, tick
from swarm.recovery.llm import RecoveryContext, RecoveryLLMError, TransformersQwenClient
from swarm.safety import DroneSnapshot


def _snapshot(drone_id: int, position, velocity) -> DroneSnapshot:
    return DroneSnapshot(drone_id=drone_id, position=position, velocity=velocity)


def _snapshots() -> dict[int, DroneSnapshot]:
    return {
        0: _snapshot(0, (-2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        1: _snapshot(1, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    }


def _result(
    drone: int,
    *,
    margin: float = 1.0,
    pred: float = 1.0,
    deg: float = 0.0,
    pair: int | None = None,
    intervened: bool = False,
    proactive: bool = False,
) -> FilterResult:
    return FilterResult(
        drone=drone,
        mode="warning" if margin < 0.2 else "normal",
        nominal_action=(1.0, 0.0),
        safe_action=(1.0, 0.0),
        safety_margin=margin,
        predicted_margin=pred,
        time_to_min_margin=0.0,
        time_to_safety_boundary=None,
        prediction_reliability_score=1.0,
        recovery_buffer=None,
        semantic_recovery_feasible=True,
        degradation=deg,
        worst_pair=pair,
        intervened=intervened,
        proactive=proactive,
    )


def _context() -> RecoveryContext:
    snapshots = _snapshots()
    return RecoveryContext(
        event="PREDICTED_CONFLICT",
        agent_i=0,
        agent_j=1,
        current_margin=0.3,
        predicted_min_margin=-0.2,
        margin_degradation=0.4,
        intervention_count=0,
        cause="TRAJECTORY_CONFLICT",
        severity="HIGH",
        snapshots=snapshots,
        current_goals={0: (4.0, 0.0, 0.0), 1: (-4.0, 0.0, 0.0)},
        base_goals={0: (4.0, 0.0, 0.0), 1: (-4.0, 0.0, 0.0)},
        priorities={0: "normal", 1: "normal"},
        timestamp_ms=1000,
    )


class ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = RecoveryValidator(known_drones={0, 1}, now_ms=1000)

    def test_rejects_non_dict(self) -> None:
        result = self.validator.validate("not json")
        self.assertFalse(result.valid)
        self.assertTrue(result.errors)

    def test_rejects_forbidden_field(self) -> None:
        payload = {
            "intent_text": "x",
            "commands": [],
            "d0": 0.1,
        }
        result = self.validator.validate(payload)
        self.assertFalse(result.valid)
        self.assertTrue(any("forbidden" in e for e in result.errors))

    def test_rejects_illegal_action(self) -> None:
        payload = {
            "intent_text": "x",
            "commands": [
                {
                    "drone": 0,
                    "action": "EXPLODE",
                    "ttl_sec": 1.0,
                    "command_id": "c1",
                }
            ],
        }
        result = self.validator.validate(payload)
        self.assertFalse(result.valid)

    def test_reroute_requires_waypoint(self) -> None:
        payload = {
            "intent_text": "x",
            "commands": [
                {
                    "drone": 0,
                    "action": "REROUTE",
                    "ttl_sec": 1.0,
                    "command_id": "c1",
                }
            ],
        }
        result = self.validator.validate(payload)
        self.assertFalse(result.valid)
        self.assertTrue(any("waypoint" in e for e in result.errors))

    def test_rejects_waypoint_outside_arena(self) -> None:
        payload = {
            "intent_text": "x",
            "commands": [
                {
                    "drone": 0,
                    "action": "REROUTE",
                    "waypoint": [100.0, 0.0, 0.0],
                    "ttl_sec": 1.0,
                    "command_id": "c1",
                }
            ],
        }
        result = self.validator.validate(payload)
        self.assertFalse(result.valid)

    def test_injects_timestamp_and_accepts_valid_plan(self) -> None:
        payload = {
            "intent_text": "recover",
            "commands": [
                {
                    "drone": 0,
                    "action": "REROUTE",
                    "waypoint": [1.0, 1.0, 0.0],
                    "ttl_sec": 1.0,
                    "command_id": "c1",
                }
            ],
        }
        result = self.validator.validate(payload)
        self.assertTrue(result.valid)
        self.assertEqual(result.plan.timestamp_ms, 1000)


class DeterministicClientTest(unittest.TestCase):
    def test_generates_valid_reroute_plan(self) -> None:
        client = DeterministicRecoveryClient()
        result = client.generate(_context())
        self.assertTrue(result.valid)
        self.assertEqual(result.plan.commands[0].action, "REROUTE")


class TransformersClientTest(unittest.TestCase):
    def test_4bit_fails_without_bitsandbytes(self) -> None:
        client = TransformersQwenClient(
            model_id="Qwen/Qwen3-4B-Instruct",
            load_in_4bit=True,
            load=False,
        )
        with self.assertRaises(RecoveryLLMError):
            client.load()


class MlxLmClientTest(unittest.TestCase):
    def test_constructs_without_loading_model(self) -> None:
        client = MlxLmClient(model_id="mlx-community/Qwen3-4B-4bit", load=False)
        self.assertEqual(client.name, "mlx-lm-qwen")
        self.assertEqual(client.max_tokens, 256)

    def test_reuses_json_extraction(self) -> None:
        client = MlxLmClient(model_id="x", load=False)
        self.assertEqual(
            client._extract_json('prefix {"a": 1} suffix'),
            {"a": 1},
        )


class ExecutorTest(unittest.TestCase):
    def test_reroute_and_expiry(self) -> None:
        active = {0: ActiveRecovery()}
        plan = RecoveryPlan(
            schema_version=1,
            intent_text="reroute",
            commands=[
                RecoveryCommand(
                    drone=0,
                    action="REROUTE",
                    waypoint=(0.0, 2.0, 0.0),
                    priority="normal",
                    ttl_sec=1.0,
                    command_id="c1",
                )
            ],
            rationale="",
            timestamp_ms=0,
        )
        apply_plan(active, plan, t=0.0, base_goals={0: (4.0, 0.0, 0.0)})
        self.assertEqual(active[0].goal_override, (0.0, 2.0, 0.0))
        tick(active, t=1.0)
        self.assertIsNone(active[0].goal_override)

    def test_hold_and_abort(self) -> None:
        active = {0: ActiveRecovery()}
        plan = RecoveryPlan(
            schema_version=1,
            intent_text="hold then abort",
            commands=[
                RecoveryCommand(
                    drone=0,
                    action="HOLD",
                    priority="normal",
                    ttl_sec=0.5,
                    command_id="c1",
                ),
                RecoveryCommand(
                    drone=0,
                    action="ABORT",
                    priority="normal",
                    ttl_sec=0.5,
                    command_id="c2",
                ),
            ],
            rationale="",
            timestamp_ms=0,
        )
        apply_plan(active, plan, t=0.0, base_goals={0: (4.0, 0.0, 0.0)})
        self.assertTrue(active[0].aborted)
        self.assertEqual(active[0].velocity_scale, 0.0)


class SemanticRecoveryTest(unittest.TestCase):
    def _step(
        self,
        recovery: SemanticRecovery,
        t: float,
        *,
        margin: float = 1.0,
        pred: float = 1.0,
        deg: float = 0.0,
        intervened: bool = False,
        proactive: bool = False,
    ) -> None:
        snapshots = _snapshots()
        results = {
            0: _result(0, margin=margin, pred=pred, deg=deg, pair=1, intervened=intervened, proactive=proactive),
            1: _result(1, margin=margin, pred=pred, deg=deg, pair=0, intervened=intervened, proactive=proactive),
        }
        recovery.step(
            t=t,
            snapshots=snapshots,
            results=results,
            current_goals={0: (4.0, 0.0, 0.0), 1: (-4.0, 0.0, 0.0)},
            base_goals={0: (4.0, 0.0, 0.0), 1: (-4.0, 0.0, 0.0)},
        )

    def test_r1_executes_on_prealert_and_counts_unnecessary(self) -> None:
        recovery = SemanticRecovery(mode="R1")
        self._step(recovery, 0.0, margin=0.5, pred=-0.2, proactive=True)
        self.assertEqual(recovery.counters.mission_changes, 1)
        self.assertEqual(recovery.counters.plans_executed, 1)
        # No runtime confirmation in the next step, so the R1 mission change is
        # flagged as unnecessary once the confirmation window expires.
        self._step(recovery, 2.0, margin=0.5, pred=1.0, proactive=False)
        self.assertEqual(recovery.counters.unnecessary_mission_changes, 1)

    def test_r0_ignores_prealert_and_executes_on_confirmation(self) -> None:
        recovery = SemanticRecovery(mode="R0")
        self._step(recovery, 0.0, margin=0.5, pred=-0.2, proactive=True)
        self.assertEqual(recovery.counters.mission_changes, 0)
        self._step(recovery, 0.1, margin=0.1, deg=0.5, proactive=False)
        self.assertEqual(recovery.counters.mission_changes, 1)

    def test_r2_speculates_then_confirms(self) -> None:
        recovery = SemanticRecovery(mode="R2")
        self._step(recovery, 0.0, margin=0.5, pred=-0.2, proactive=True)
        self.assertEqual(recovery.counters.speculative_plans_generated, 1)
        self.assertEqual(recovery.counters.mission_changes, 0)
        self._step(recovery, 0.1, margin=0.1, deg=0.5, proactive=False)
        self.assertEqual(recovery.counters.mission_changes, 1)
        self.assertEqual(recovery.counters.speculative_plans_discarded, 0)
        self.assertEqual(len(recovery.counters.candidate_lead_times_s), 1)

    def test_r2_discards_unconfirmed_candidate(self) -> None:
        recovery = SemanticRecovery(mode="R2")
        self._step(recovery, 0.0, margin=0.5, pred=-0.2, proactive=True)
        self._step(recovery, 3.0, margin=0.5, pred=1.0, proactive=False)
        self.assertEqual(recovery.counters.speculative_plans_generated, 1)
        self.assertEqual(recovery.counters.speculative_plans_discarded, 1)
        self.assertEqual(recovery.counters.mission_changes, 0)

    def test_timeout_falls_back_to_deterministic_plan(self) -> None:
        slow_client = DeterministicRecoveryClient(plan_latency_s=0.3)
        recovery = SemanticRecovery(
            mode="R0",
            client=slow_client,
            config=RecoveryConfig(latency_budget_s=0.05),
        )
        self._step(recovery, 0.0, margin=0.1, deg=0.5, proactive=False)
        self.assertEqual(recovery.counters.llm_timeouts, 1)
        self.assertEqual(recovery.counters.mission_changes, 1)
        recovery.shutdown()


if __name__ == "__main__":
    unittest.main()
