import unittest

import numpy as np

from swarm.ra.hocbf import beta_of_tau
from swarm.ra.sota_cbf import solve_prediction_based_cbf_qp, solve_zocbf_qp


class SotaCbfTest(unittest.TestCase):
    def setUp(self):
        self.positions = {2: np.array([-1.0, 0.0]), 3: np.array([1.0, 0.0])}
        self.velocities = {2: np.array([1.0, 0.0]), 3: np.array([-1.0, 0.0])}
        self.nominal = {2: np.array([2.0, 0.0]), 3: np.array([-2.0, 0.0])}

    def test_zocbf_modifies_unsafe_head_on_nominal(self):
        dt = 0.05
        beta = dt - 0.7 * (1.0 - np.exp(-dt / 0.7))
        positions = {2: np.array([-0.7501, 0.0]), 3: np.array([0.7501, 0.0])}
        velocities = {2: np.zeros(2), 3: np.zeros(2)}
        safe, feasible, _ = solve_zocbf_qp(
            a_nom=self.nominal,
            positions=positions,
            velocities=velocities,
            s_now={(2, 3): 1.5},
            s_next={(2, 3): 1.5},
            dt=dt,
            gamma=1.0,
            delta=0.0,
            a_max=3.0,
            beta={2: beta, 3: beta},
        )
        self.assertTrue(feasible)
        self.assertLess(safe[2][0], self.nominal[2][0])
        self.assertGreater(safe[3][0], self.nominal[3][0])

    def test_zocbf_reports_infeasible_when_one_step_set_is_unrecoverable(self):
        dt = 0.05
        beta = dt - 0.7 * (1.0 - np.exp(-dt / 0.7))
        _, feasible, _ = solve_zocbf_qp(
            a_nom=self.nominal,
            positions=self.positions,
            velocities=self.velocities,
            s_now={(2, 3): 1.5},
            s_next={(2, 3): 1.5},
            dt=dt,
            gamma=0.1,
            delta=0.0,
            a_max=3.0,
            beta={2: beta, 3: beta},
        )
        self.assertFalse(feasible)

    def test_zocbf_max_brake_fallback_uses_full_shared_budget(self):
        dt = 0.05
        beta = dt - 0.7 * (1.0 - np.exp(-dt / 0.7))
        safe, feasible, _ = solve_zocbf_qp(
            a_nom=self.nominal,
            positions=self.positions,
            velocities=self.velocities,
            s_now={(2, 3): 1.5},
            s_next={(2, 3): 1.5},
            dt=dt,
            gamma=0.1,
            delta=0.0,
            a_max=2.0,
            beta={2: beta, 3: beta},
            infeasible_fallback="max_brake",
        )
        self.assertFalse(feasible)
        np.testing.assert_allclose(safe[2], [-2.0, 0.0])
        np.testing.assert_allclose(safe[3], [2.0, 0.0])

    def test_pb_cbf_brakes_before_distance_boundary(self):
        safe, feasible, _ = solve_prediction_based_cbf_qp(
            a_nom=self.nominal,
            positions=self.positions,
            velocities=self.velocities,
            static_distance={(2, 3): 0.5},
            alpha=2.0,
            braking_accel=2.0,
            a_max=3.0,
        )
        self.assertTrue(feasible)
        self.assertLess(safe[2][0], self.nominal[2][0])
        self.assertGreater(safe[3][0], self.nominal[3][0])

    def test_zocbf_beta_matches_exact_px4_position_coefficient(self):
        dt = 0.05
        tau = 0.7
        expected = beta_of_tau(dt, tau)
        actual = dt - tau * (1.0 - np.exp(-dt / tau))
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
