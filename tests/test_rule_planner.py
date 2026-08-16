"""Tests for the deterministic semantic mission planner."""

from __future__ import annotations

import unittest

from swarm.recovery import RuleMissionPlanner
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


class RuleMissionPlannerTest(unittest.TestCase):
    def test_fail_drone_reassigns_to_nearest_healthy(self) -> None:
        plan = RuleMissionPlanner().generate(
            _context({"kind": "fail_drone", "drone": 0})
        ).plan
        actions = {c.drone: c.action for c in plan.commands}
        self.assertEqual(actions[0], "ABORT")
        self.assertEqual(actions[1], "REASSIGN")
        reassign = next(c for c in plan.commands if c.action == "REASSIGN")
        self.assertEqual(reassign.waypoint, (4.0, 0.0, 0.0))

    def test_block_corridor_reroutes_around_zone(self) -> None:
        plan = RuleMissionPlanner().generate(
            _context(
                {
                    "kind": "block_corridor",
                    "drone": 0,
                    "zone": [-0.5, 0.5, -2.0, 2.0],
                }
            )
        ).plan
        command = plan.commands[0]
        self.assertEqual(command.action, "REROUTE")
        x, y, _ = command.waypoint
        in_zone = -0.5 <= x <= 0.5 and -2.0 <= y <= 2.0
        self.assertFalse(in_zone)

    def test_priority_change_yields_low_priority(self) -> None:
        plan = RuleMissionPlanner().generate(
            _context({"kind": "priority_change", "high": 0, "low": 1})
        ).plan
        by_action = {c.action: c for c in plan.commands}
        self.assertIn("CHANGE_PRIORITY", by_action)
        self.assertIn("YIELD", by_action)
        self.assertEqual(by_action["CHANGE_PRIORITY"].drone, 0)
        self.assertEqual(by_action["CHANGE_PRIORITY"].priority, "safety")
        self.assertEqual(by_action["YIELD"].drone, 1)

    def test_coordination_degradation_yields_and_reroutes(self) -> None:
        context = _context({"kind": "priority_change", "high": 0, "low": 1})
        context = RecoveryContext(
            event=context.event,
            agent_i=0,
            agent_j=0,
            current_margin=context.current_margin,
            predicted_min_margin=context.predicted_min_margin,
            margin_degradation=context.margin_degradation,
            intervention_count=context.intervention_count,
            cause="COORDINATION_DEGRADATION",
            severity=context.severity,
            snapshots=context.snapshots,
            current_goals=context.current_goals,
            base_goals=context.base_goals,
            priorities=context.priorities,
            timestamp_ms=context.timestamp_ms,
            mission_change=None,
        )
        plan = RuleMissionPlanner().generate(context).plan
        actions = {c.action for c in plan.commands}
        self.assertIn("YIELD", actions)
        self.assertIn("REROUTE", actions)


if __name__ == "__main__":
    unittest.main()
