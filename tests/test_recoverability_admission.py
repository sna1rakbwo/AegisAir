from __future__ import annotations

import unittest

from swarm.recovery.recoverability_admission import (
    RecoverabilityAdmissionConfig,
    RecoverabilityAdmissionCoordinator,
)
from swarm.safety import DroneSnapshot


def _snapshots(*, failed_position=(0.0, 0.0, 2.5)):
    return {
        2: DroneSnapshot(
            2,
            failed_position,
            velocity=(1.2, 0.0, 0.0),
        ),
        3: DroneSnapshot(
            3,
            (-3.0, -1.5, 2.5),
            velocity=(1.0, 0.0, 0.0),
        ),
    }


class RecoverabilityAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RecoverabilityAdmissionConfig(
            clearance_m=2.0,
            ring_extra_m=(0.5, 1.0),
            ring_samples=24,
            max_route_length_m=20.0,
        )

    def test_admits_safe_route_before_commit(self) -> None:
        coordinator = RecoverabilityAdmissionCoordinator(self.config)
        result = coordinator.step(
            step=30,
            snapshots=_snapshots(),
            base_goals={2: (4.0, 0.0, 2.5), 3: (4.0, -1.5, 2.5)},
            mission_change={"kind": "fail_drone", "drone": 2},
        )
        summary = coordinator.summary()
        self.assertEqual(summary["admission_count"], 1)
        self.assertEqual(summary["plans_committed"], 1)
        self.assertEqual(summary["unsafe_commit_count"], 0)
        self.assertGreaterEqual(
            summary["predicted_min_clearance_m"],
            summary["clearance_threshold_m"],
        )
        self.assertIn(3, result.goal_override)
        self.assertNotEqual(result.goal_override[3], (4.0, 0.0, 2.5))

    def test_rejects_occupied_orphan_goal_and_holds(self) -> None:
        coordinator = RecoverabilityAdmissionCoordinator(self.config)
        snapshots = _snapshots(failed_position=(2.0, 0.0, 2.5))
        result = coordinator.step(
            step=30,
            snapshots=snapshots,
            base_goals={2: (2.0, 0.0, 2.5), 3: (4.0, -1.5, 2.5)},
            mission_change={"kind": "fail_drone", "drone": 2},
        )
        summary = coordinator.summary()
        self.assertEqual(summary["rejection_count"], 1)
        self.assertEqual(summary["plans_committed"], 0)
        self.assertEqual(summary["rejection_reason"], "orphan_goal_inside_failed_vehicle_envelope")
        self.assertEqual(result.velocity_scale[3], 0.0)
        self.assertEqual(result.goal_override[3], snapshots[3].position)

    def test_hold_goal_is_frozen(self) -> None:
        coordinator = RecoverabilityAdmissionCoordinator(self.config)
        snapshots = _snapshots(failed_position=(2.0, 0.0, 2.5))
        first = coordinator.step(
            step=30,
            snapshots=snapshots,
            base_goals={2: (2.0, 0.0, 2.5), 3: (4.0, -1.5, 2.5)},
            mission_change={"kind": "fail_drone", "drone": 2},
        )
        moved = dict(snapshots)
        moved[3] = DroneSnapshot(3, (-2.8, -1.4, 2.5), velocity=(0.0, 0.0, 0.0))
        second = coordinator.step(
            step=31,
            snapshots=moved,
            base_goals={2: (2.0, 0.0, 2.5), 3: (4.0, -1.5, 2.5)},
            mission_change=None,
        )
        self.assertEqual(first.goal_override[3], second.goal_override[3])


if __name__ == "__main__":
    unittest.main()
