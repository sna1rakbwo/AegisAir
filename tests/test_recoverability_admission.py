from __future__ import annotations

import unittest

import numpy as np

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

    def test_continuous_failed_trajectory_closes_sampling_gaps(self) -> None:
        coordinator = RecoverabilityAdmissionCoordinator(self.config)
        clearance = coordinator._route_clearance(
            [np.array([-1.0, 0.0]), np.array([1.0, 0.0])],
            [np.array([0.0, -1.0]), np.array([0.0, 1.0])],
        )
        self.assertAlmostEqual(clearance, 0.0)

    def test_admission_depends_on_healthy_velocity(self) -> None:
        config = RecoverabilityAdmissionConfig(
            clearance_m=0.5,
            tracking_error_buffer_m=0.0,
            ring_extra_m=(0.5,),
            ring_samples=8,
            max_route_length_m=4.0,
            rollout_horizon_s=2.0,
        )

        def decide(velocity: float) -> dict[str, object]:
            coordinator = RecoverabilityAdmissionCoordinator(config)
            coordinator.step(
                step=30,
                snapshots={
                    2: DroneSnapshot(
                        2,
                        (4.0, 4.0, 2.5),
                        velocity=(0.0, 0.0, 0.0),
                    ),
                    3: DroneSnapshot(
                        3,
                        (-1.5, 0.0, 2.5),
                        velocity=(velocity, 0.0, 0.0),
                    ),
                },
                base_goals={2: (0.0, 0.0, 2.5), 3: (3.0, 0.0, 2.5)},
                mission_change={"kind": "fail_drone", "drone": 2},
            )
            return coordinator.summary()

        toward = decide(1.0)
        away = decide(-1.0)
        self.assertEqual(toward["state"], "admitted")
        self.assertEqual(away["state"], "rejected_hold")
        self.assertEqual(
            away["rejection_reason"],
            "no_dynamically_recoverable_candidate",
        )

    def test_rejects_initial_velocity_above_horizontal_limit(self) -> None:
        coordinator = RecoverabilityAdmissionCoordinator(self.config)
        snapshots = _snapshots()
        snapshots[3] = DroneSnapshot(
            3,
            snapshots[3].position,
            velocity=(self.config.healthy_velocity_limit_mps + 0.1, 0.0, 0.0),
        )
        result = coordinator.step(
            step=30,
            snapshots=snapshots,
            base_goals={2: (4.0, 0.0, 2.5), 3: (4.0, -1.5, 2.5)},
            mission_change={"kind": "fail_drone", "drone": 2},
        )
        summary = coordinator.summary()
        self.assertEqual(summary["state"], "rejected_hold")
        self.assertEqual(
            summary["rejection_reason"],
            "initial_velocity_limit_violation",
        )
        self.assertEqual(summary["plans_committed"], 0)
        self.assertEqual(result.velocity_scale[3], 0.0)

    def test_rollout_rejects_execution_response_speed_overshoot(self) -> None:
        config = RecoverabilityAdmissionConfig(
            clearance_m=0.5,
            tracking_error_buffer_m=0.0,
            reaction_delay_s=0.0,
            command_feedforward_tau_s=40.0,
        )
        coordinator = RecoverabilityAdmissionCoordinator(config)
        rollout = coordinator._rollout_route(
            points=[np.array([0.0, 0.0]), np.array([-5.0, 0.0])],
            initial_velocity=np.array([config.healthy_velocity_limit_mps, 0.0]),
            failed_trajectory=[
                np.array([5.0, 5.0]),
                np.array([5.0, 5.0]),
            ],
        )
        self.assertFalse(rollout.feasible)
        self.assertEqual(rollout.rejection_reason, "velocity_limit_violation")
        self.assertGreater(
            abs(rollout.terminal_speed_mps),
            config.healthy_velocity_limit_mps,
        )

    def test_tracking_error_budget_is_part_of_admission_threshold(self) -> None:
        config = RecoverabilityAdmissionConfig(
            clearance_m=2.0,
            tracking_error_buffer_m=0.3,
        )
        self.assertAlmostEqual(config.effective_clearance_m, 2.3)

    def test_braking_uncertainty_uses_conservative_capability(self) -> None:
        config = RecoverabilityAdmissionConfig(
            braking_accel_mps2=2.0,
            braking_accel_uncertainty_mps2=0.4,
        )
        self.assertAlmostEqual(config.conservative_braking_accel_mps2, 1.6)


if __name__ == "__main__":
    unittest.main()
