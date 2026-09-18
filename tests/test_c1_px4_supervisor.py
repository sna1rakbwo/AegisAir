"""Unit tests for the frozen C1 PX4 v2 safety supervisor."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from swarm.ra.c1_px4_supervisor import (
    C1Px4AdmissionSupervisor,
    C1Px4SupervisorConfig,
)
from swarm.ra.margins import RuntimeAssuranceParams
from swarm.safety import DroneSnapshot


class C1Px4AdmissionSupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = C1Px4SupervisorConfig()
        self.goals = {2: (4.0, 0.0, 2.5), 3: (-4.0, 0.0, 2.5)}
        self.starts = {2: (-4.0, 0.0, 2.5), 3: (4.0, 0.0, 2.5)}
        self.supervisor = C1Px4AdmissionSupervisor(
            config=self.config,
            drone_ids=[2, 3],
            base_goals=self.goals,
            reset_starts=self.starts,
            rate_hz=20.0,
            ra_params=RuntimeAssuranceParams(tau_ctrl=0.2),
            v_max=1.5,
        )

    @staticmethod
    def _snap(drone: int, position: tuple[float, float, float]) -> DroneSnapshot:
        return DroneSnapshot(
            drone_id=drone,
            position=position,
            velocity=(0.0, 0.0, 0.0),
        )

    def test_warmup_covers_delay_plus_two_cycles(self) -> None:
        self.assertEqual(self.supervisor.warmup_steps, 8)
        snapshots = {
            2: self._snap(2, self.starts[2]),
            3: self._snap(3, self.starts[3]),
        }
        _, scales = self.supervisor.directives(
            step=7, snapshots=snapshots, goal_epsilon=0.5
        )
        self.assertEqual(scales, {2: 0.0, 3: 0.0})
        goals, scales = self.supervisor.directives(
            step=8, snapshots=snapshots, goal_epsilon=0.5
        )
        self.assertEqual(self.supervisor.phase, "STAGE_YIELDER")
        self.assertEqual(scales, {2: 0.0, 3: 1.0})
        self.assertEqual(goals[3], self.supervisor.stage_goal)

    def test_clearance_uses_point_seven_execution_tau(self) -> None:
        self.assertAlmostEqual(
            self.supervisor.required_stage_clearance_m, 3.7975, places=4
        )
        summary = self.supervisor.summary()
        self.assertEqual(summary["barrier_tau_px4_s"], 0.2)
        self.assertEqual(summary["command_feedforward_tau_s"], 0.7)
        self.assertEqual(summary["admission_execution_tau_s"], 0.7)

    def test_serial_admission_reaches_complete(self) -> None:
        moving = {
            2: self._snap(2, self.starts[2]),
            3: self._snap(3, self.starts[3]),
        }
        self.supervisor.directives(step=8, snapshots=moving, goal_epsilon=0.5)
        moving[3] = self._snap(3, self.supervisor.stage_goal)
        self.supervisor.directives(step=9, snapshots=moving, goal_epsilon=0.5)
        self.assertEqual(self.supervisor.phase, "PASS_PRIMARY")
        moving[2] = self._snap(2, self.goals[2])
        self.supervisor.directives(step=10, snapshots=moving, goal_epsilon=0.5)
        self.assertEqual(self.supervisor.phase, "CLEAR_PRIMARY")
        moving[2] = self._snap(2, self.supervisor.primary_clear_goal)
        self.supervisor.directives(step=11, snapshots=moving, goal_epsilon=0.5)
        self.assertEqual(self.supervisor.phase, "PASS_YIELDER")
        moving[3] = self._snap(3, self.goals[3])
        self.supervisor.directives(step=12, snapshots=moving, goal_epsilon=0.5)
        self.assertEqual(self.supervisor.phase, "RETURN_PRIMARY")
        moving[2] = self._snap(2, self.goals[2])
        _, scales = self.supervisor.directives(
            step=13, snapshots=moving, goal_epsilon=0.5
        )
        self.assertTrue(self.supervisor.complete)
        self.assertEqual(scales, {2: 0.0, 3: 0.0})

    def test_brake_latch_has_hysteresis(self) -> None:
        bad = {2: SimpleNamespace(feasible=False, safety_margin=0.4)}
        clean = {2: SimpleNamespace(feasible=True, safety_margin=0.6)}
        self.supervisor.observe_ra(bad)
        self.assertTrue(self.supervisor.brake_latched)
        for _ in range(self.config.latch_release_cycles - 1):
            self.supervisor.observe_ra(clean)
            self.assertTrue(self.supervisor.brake_latched)
        self.supervisor.observe_ra(clean)
        self.assertFalse(self.supervisor.brake_latched)
        self.assertEqual(self.supervisor.latch_trip_count, 1)
        self.assertEqual(self.supervisor.latch_release_count, 1)


if __name__ == "__main__":
    unittest.main()
