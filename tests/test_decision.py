"""Tests for compact decision -> RecoveryPlan expansion."""

from __future__ import annotations

import unittest

from swarm.interfaces import MissionDecision
from swarm.recovery import expand_decision, parse_decision
from swarm.recovery.decision import semantic_errors
from swarm.recovery.llm import RecoveryContext
from swarm.safety import DroneSnapshot


def _context(mission_change: dict) -> RecoveryContext:
    snapshots = {
        0: DroneSnapshot(0, (-4.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        1: DroneSnapshot(1, (-4.0, -2.0, 0.0), (0.0, 0.0, 0.0)),
    }
    return RecoveryContext(
        event="MISSION_PLAN_INVALIDATED",
        agent_i=0,
        agent_j=0,
        current_margin=1.0,
        predicted_min_margin=None,
        margin_degradation=None,
        intervention_count=0,
        cause="MISSION_CHANGE",
        severity="MEDIUM",
        snapshots=snapshots,
        current_goals={0: (4.0, 0.0, 0.0), 1: (4.0, -2.0, 0.0)},
        base_goals={0: (4.0, 0.0, 0.0), 1: (4.0, -2.0, 0.0)},
        priorities={0: "normal", 1: "normal"},
        timestamp_ms=1000,
        mission_change=mission_change,
    )


class DecisionExpansionTest(unittest.TestCase):
    def test_parse_rejects_non_dict(self) -> None:
        decision, errors = parse_decision("nope")
        self.assertIsNone(decision)
        self.assertTrue(errors)

    def test_fail_drone_expands_to_abort_and_reassign(self) -> None:
        decision = MissionDecision(action="REASSIGN", agent=1)
        plan = expand_decision(decision, _context({"kind": "fail_drone", "drone": 0}))
        actions = {c.drone: c.action for c in plan.commands}
        self.assertEqual(actions[0], "ABORT")
        self.assertEqual(actions[1], "REASSIGN")
        reassign = next(c for c in plan.commands if c.action == "REASSIGN")
        self.assertEqual(reassign.waypoint, (4.0, 0.0, 0.0))

    def test_block_corridor_expands_to_reroute(self) -> None:
        decision = MissionDecision(action="REROUTE", agent=0)
        plan = expand_decision(
            decision,
            _context(
                {
                    "kind": "block_corridor",
                    "drone": 0,
                    "zone": [-0.3, 0.3, -1.0, 1.0],
                }
            ),
        )
        self.assertEqual(plan.commands[0].action, "REROUTE")
        self.assertEqual(plan.commands[0].ttl_sec, 5.0)
        x, y, _ = plan.commands[0].waypoint
        self.assertFalse(-0.3 <= x <= 0.3 and -1.0 <= y <= 1.0)

    def test_priority_change_expands_to_priority_and_yield(self) -> None:
        decision = MissionDecision(action="CHANGE_PRIORITY", high=0, low=1)
        plan = expand_decision(
            decision, _context({"kind": "priority_change", "high": 0, "low": 1})
        )
        by_action = {c.action: c for c in plan.commands}
        self.assertIn("CHANGE_PRIORITY", by_action)
        self.assertIn("YIELD", by_action)
        self.assertEqual(by_action["CHANGE_PRIORITY"].priority, "safety")

    def test_semantic_errors_reject_unknown_agent(self) -> None:
        decision = MissionDecision(action="YIELD", agent=99)
        errors = semantic_errors(
            decision, _context({"kind": "priority_change", "high": 0, "low": 1})
        )
        self.assertTrue(any("not an active agent" in e for e in errors))

    def test_semantic_errors_reject_failed_self_reassign(self) -> None:
        decision = MissionDecision(action="REASSIGN", agent=0)
        errors = semantic_errors(
            decision, _context({"kind": "fail_drone", "drone": 0})
        )
        self.assertTrue(any("cannot reassign" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
