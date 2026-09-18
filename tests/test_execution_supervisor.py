"""C2-prime execution-conformance supervisor tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from swarm.ra.execution_supervisor import (
    ExecutionConformanceSupervisor,
    ExecutionSupervisorConfig,
    deterministic_backup_velocity,
)
from swarm.safety import DroneSnapshot


def _snapshot(
    drone: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    timestamp_ms: int,
) -> DroneSnapshot:
    return DroneSnapshot(
        drone_id=drone,
        position=position,
        velocity=velocity,
        status="armed",
        timestamp_ms=timestamp_ms,
    )


class ExecutionConformanceSupervisorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExecutionSupervisorConfig(
            tau_s=0.2,
            velocity_residual_limit_mps=0.1,
            position_residual_limit_m=0.02,
            trip_samples=2,
            release_samples=3,
        )

    def test_exact_zoh_observation_has_zero_residual(self) -> None:
        supervisor = ExecutionConformanceSupervisor(self.config)
        first = {2: _snapshot(2, (0.0, 0.0, 2.5), (0.0, 0.0, 0.0), 1000)}
        supervisor.record_commands(first, {2: np.array([1.0, 0.0])}, now_t_s=0.0)
        dt = 0.05
        alpha = 1.0 - math.exp(-dt / 0.2)
        beta = dt - 0.2 * alpha
        second = {
            2: _snapshot(
                2,
                (beta, 0.0, 2.5),
                (alpha, 0.0, 0.0),
                1050,
            )
        }
        residual = supervisor.observe(second, now_t_s=dt)[2]
        self.assertAlmostEqual(residual.velocity_mps, 0.0)
        self.assertAlmostEqual(residual.position_m, 0.0)

    def test_residual_trip_and_hysteretic_release(self) -> None:
        supervisor = ExecutionConformanceSupervisor(self.config)
        bad = {
            2: type("Residual", (), {"velocity_mps": 0.2, "position_m": 0.0})()
        }
        decision = supervisor.assess(
            residuals=bad,
            qp_feasible=True,
            solve_elapsed_s=0.001,
            telemetry_ages_s={2: 0.01},
        )
        self.assertFalse(decision.active)
        decision = supervisor.assess(
            residuals=bad,
            qp_feasible=True,
            solve_elapsed_s=0.001,
            telemetry_ages_s={2: 0.01},
        )
        self.assertTrue(decision.active)
        self.assertTrue(decision.newly_tripped)

        for _ in range(2):
            decision = supervisor.assess(
                residuals={},
                qp_feasible=True,
                solve_elapsed_s=0.001,
                telemetry_ages_s={2: 0.01},
            )
            self.assertTrue(decision.active)
            self.assertIn("HYSTERESIS_HOLD", decision.reasons)
        decision = supervisor.assess(
            residuals={},
            qp_feasible=True,
            solve_elapsed_s=0.001,
            telemetry_ages_s={2: 0.01},
        )
        self.assertFalse(decision.active)
        self.assertTrue(decision.newly_released)

    def test_qp_infeasible_trips_when_residual_gate_disabled(self) -> None:
        config = ExecutionSupervisorConfig(
            tau_s=0.2,
            velocity_residual_limit_mps=0.1,
            position_residual_limit_m=0.02,
            trip_samples=1,
            enable_residual_gate=False,
        )
        supervisor = ExecutionConformanceSupervisor(config)
        decision = supervisor.assess(
            residuals={},
            qp_feasible=False,
            solve_elapsed_s=0.001,
            telemetry_ages_s={2: 0.01},
        )
        self.assertTrue(decision.active)
        self.assertIn("QP_INFEASIBLE", decision.reasons)

    def test_backup_is_acceleration_limited_and_points_outward(self) -> None:
        snapshots = {
            2: _snapshot(2, (-1.0, 0.0, 2.5), (1.0, 0.0, 0.0), 1000),
            3: _snapshot(3, (1.0, 0.0, 2.5), (-1.0, 0.0, 0.0), 1000),
        }
        command = deterministic_backup_velocity(
            drone=2,
            snapshots=snapshots,
            dt_s=0.05,
            a_max=3.0,
            v_max=1.5,
            retreat_speed_mps=0.6,
        )
        # Drone 2 is left of the centroid, so backup must reduce +x velocity.
        self.assertLess(command[0], 1.0)
        self.assertLessEqual(float(np.linalg.norm(command - np.array([1.0, 0.0]))), 0.15 + 1e-9)

    def test_predictive_gate_trips_before_qp_infeasibility_and_selects_yielder(self) -> None:
        config = ExecutionSupervisorConfig(
            tau_s=0.2,
            velocity_residual_limit_mps=1.0,
            position_residual_limit_m=1.0,
            trip_samples=1,
            enable_residual_gate=False,
            enable_qp_gate=False,
            enable_predictive_gate=True,
            safe_distance_m=1.6,
            response_delay_s=0.3,
            braking_deceleration_mps2=2.0,
            recoverability_buffer_m=0.2,
        )
        supervisor = ExecutionConformanceSupervisor(config)
        snapshots = {
            2: _snapshot(2, (-2.0, 0.0, 2.5), (1.5, 0.0, 0.0), 1000),
            3: _snapshot(3, (2.0, 0.0, 2.5), (-1.5, 0.0, 0.0), 1000),
        }
        decision = supervisor.assess(
            residuals={},
            qp_feasible=True,
            solve_elapsed_s=0.001,
            telemetry_ages_s={2: 0.01, 3: 0.01},
            snapshots=snapshots,
        )
        self.assertTrue(decision.active)
        self.assertIn("PREDICTIVE_RECOVERABILITY", decision.reasons)
        self.assertEqual(decision.backup_drones, (3,))
        pair = decision.recoverability[0]
        self.assertGreater(pair.separation_m, config.safe_distance_m)
        self.assertLess(pair.margin_m, 0.0)


if __name__ == "__main__":
    unittest.main()
