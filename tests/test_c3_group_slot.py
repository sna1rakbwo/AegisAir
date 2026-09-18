from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from swarm.recovery import C3GroupSlotCoordinator, GroupSlotConfig
from swarm.safety import DroneSnapshot


DRONES = [2, 3, 4, 5]
STARTS = {
    2: (-4.0, -1.5, 2.5),
    3: (4.0, 1.5, 2.5),
    4: (-1.5, 4.0, 2.5),
    5: (1.5, -4.0, 2.5),
}
GOALS = {
    2: (4.0, -1.5, 2.5),
    3: (-4.0, 1.5, 2.5),
    4: (-1.5, -4.0, 2.5),
    5: (1.5, 4.0, 2.5),
}


def snapshots(positions=None, velocities=None):
    positions = positions or STARTS
    velocities = velocities or {drone: (0.0, 0.0, 0.0) for drone in DRONES}
    return {
        drone: DroneSnapshot(
            drone_id=drone,
            position=positions[drone],
            target=GOALS[drone],
            velocity=velocities[drone],
        )
        for drone in DRONES
    }


class C3GroupSlotCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = C3GroupSlotCoordinator(
            GroupSlotConfig(
                conflict_zone_id="moderate_crossing",
                conflict_zone_center=(0.0, 0.0),
                group_order=((2, 3), (4, 5)),
                settle_steps=2,
            ),
            DRONES,
            GOALS,
        )
        self.step = 0

    def call(self, current):
        result = self.coordinator.directives(
            step=self.step,
            timestamp_ms=self.step + 1,
            snapshots=current,
            nominal={drone: np.zeros(2) for drone in DRONES},
            previous_results=None,
        )
        self.step += 1
        return result

    def test_freezes_holds_and_releases_compatible_groups(self) -> None:
        first = self.call(snapshots())
        self.assertEqual(first.decision.coordination_mode, "admission_pending")
        self.assertEqual(first.decision.frozen_hold_goals, STARTS)
        self.call(snapshots())
        first_pass = self.call(snapshots())
        self.assertEqual(first_pass.decision.authorized_drone_ids, [2, 3])
        self.assertEqual(set(first_pass.decision.held_drone_ids), {4, 5})

        positions = dict(STARTS)
        positions[2] = GOALS[2]
        positions[3] = GOALS[3]
        self.call(snapshots(positions))
        released = self.call(snapshots(positions))
        self.assertEqual(released.decision.coordination_mode, "release")
        second_pass = self.call(snapshots(positions))
        self.assertEqual(second_pass.decision.authorized_drone_ids, [4, 5])
        self.assertEqual(set(second_pass.decision.held_drone_ids), set())

    def test_complete_latches_without_retrigger(self) -> None:
        self.call(snapshots())
        self.call(snapshots())
        self.call(snapshots())
        positions = dict(STARTS)
        positions[2], positions[3] = GOALS[2], GOALS[3]
        self.call(snapshots(positions))
        self.call(snapshots(positions))
        self.call(snapshots(positions))
        positions[4], positions[5] = GOALS[4], GOALS[5]
        self.call(snapshots(positions))
        self.call(snapshots(positions))
        done = self.call(snapshots(positions))
        self.assertEqual(done.decision.coordination_mode, "normal")
        self.assertTrue(self.coordinator.summary()["complete"])
        later = self.call(snapshots(positions))
        self.assertEqual(later.decision.coordination_mode, "normal")
        self.assertEqual(later.decision.authorized_drone_ids, [])
        self.assertEqual(self.coordinator.summary()["trigger_count"], 1)

    def test_ra_veto_is_logged_but_cannot_bypass_ra(self) -> None:
        self.call(snapshots())
        self.call(snapshots())
        passed = self.call(snapshots())
        results = {
            drone: SimpleNamespace(feasible=(drone != 2)) for drone in DRONES
        }
        observed = self.coordinator.observe_ra(results)
        self.assertTrue(observed.ra_vetoed)
        self.assertEqual(self.coordinator.summary()["ra_veto_count"], 1)


if __name__ == "__main__":
    unittest.main()
