import unittest

import numpy as np

from swarm.ra.hocbf import beta_of_tau
from swarm.ra.sota_cbf import (
    _solve_projection_qp,
    minimum_prediction_based_constraint_slack,
    solve_prediction_based_cbf_qp,
    solve_zocbf_qp,
)


class ProjectionQpTest(unittest.TestCase):
    def test_joint_box_and_oblique_constraint_returns_nearest_point(self):
        nominal = np.array([1.0, -1.0])
        projected, feasible, iterations = _solve_projection_qp(
            x_nom=nominal,
            halfspaces=[(np.array([1.0, 2.0]), 2.5)],
            a_max=1.0,
            max_iters=3000,
        )
        self.assertTrue(feasible)
        self.assertLess(iterations, 100)
        np.testing.assert_allclose(projected, [1.0, 0.75], atol=1e-7)
        np.testing.assert_array_equal(nominal, [1.0, -1.0])

    def test_two_active_constraints_have_same_minimizer_in_either_order(self):
        halfspaces = [
            (np.array([1.0, 1.0]), 1.0),
            (np.array([0.0, 1.0]), 0.75),
        ]
        for constraints in (halfspaces, list(reversed(halfspaces))):
            with self.subTest(first_normal=constraints[0][0].tolist()):
                projected, feasible, _ = _solve_projection_qp(
                    x_nom=np.zeros(2),
                    halfspaces=constraints,
                    a_max=1.0,
                    max_iters=3000,
                )
                self.assertTrue(feasible)
                np.testing.assert_allclose(projected, [0.25, 0.75], atol=1e-7)

    def test_incompatible_constraints_are_not_reported_feasible(self):
        _, feasible, _ = _solve_projection_qp(
            x_nom=np.zeros(2),
            halfspaces=[
                (np.array([1.0, 0.0]), 0.75),
                (np.array([-1.0, 0.0]), 0.75),
            ],
            a_max=1.0,
            max_iters=3000,
        )
        self.assertFalse(feasible)

    def test_transient_stall_does_not_end_projection(self):
        normals = np.array([
            [0.5736648876548924, -0.8190901029019333],
            [-0.9773374659406598, 0.21168721659252282],
            [-0.29484011607233285, 0.9555466006189616],
            [-0.9835269866454438, -0.18076135245160377],
        ])
        bounds = np.array([
            -0.20041016428359143, 0.13493650618070513,
            -0.7329750269316113, 0.06362111139644355,
        ])
        # The nearest feasible vertex is the intersection of rows 0 and 1.
        expected = np.linalg.solve(normals[:2], bounds[:2])
        projected, feasible, _ = _solve_projection_qp(
            x_nom=np.array([0.7555612926713484, 0.6991286969722033]),
            halfspaces=list(zip(normals, bounds)),
            a_max=1.0,
            max_iters=3000,
        )
        self.assertTrue(feasible)
        np.testing.assert_allclose(projected, expected, atol=1e-7)


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

    def test_zocbf_coincident_pair_fails_closed(self):
        dt = 0.05
        beta = dt - 0.7 * (1.0 - np.exp(-dt / 0.7))
        _, feasible, _ = solve_zocbf_qp(
            a_nom=self.nominal,
            positions={2: np.zeros(2), 3: np.zeros(2)},
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
        self.assertGreaterEqual(
            minimum_prediction_based_constraint_slack(
                accelerations=safe,
                positions=self.positions,
                velocities=self.velocities,
                static_distance={(2, 3): 0.5},
                alpha=2.0,
                braking_accel=2.0,
                a_max=3.0,
            ),
            -1e-6,
        )
        self.assertLess(
            minimum_prediction_based_constraint_slack(
                accelerations=self.nominal,
                positions=self.positions,
                velocities=self.velocities,
                static_distance={(2, 3): 0.5},
                alpha=2.0,
                braking_accel=2.0,
                a_max=3.0,
            ),
            0.0,
        )

    def test_pb_cbf_finds_feasible_oblique_input_at_box_boundary(self):
        # This safe initial state gives [0.5, 1, -0.5, -1] @ A >= 1.
        direction = np.array([1.0, 2.0]) / np.sqrt(5.0)
        distance = 0.8 + 2.0 * np.sqrt(5.0) + 1.25 - 2.0
        safe, feasible, iterations = solve_prediction_based_cbf_qp(
            a_nom={2: np.array([2.0, -1.0]), 3: np.array([-2.0, 1.0])},
            positions={2: distance * direction, 3: np.zeros(2)},
            velocities={2: np.array([-0.5, -1.0]), 3: np.array([0.5, 1.0])},
            static_distance={(2, 3): 0.8},
            alpha=0.5,
            braking_accel=2.0,
            a_max=2.0,
            infeasible_fallback="max_brake",
        )
        self.assertTrue(feasible)
        self.assertLess(iterations, 100)
        np.testing.assert_allclose(safe[2], [2.0, -0.5], atol=1e-7)
        np.testing.assert_allclose(safe[3], [-2.0, 0.5], atol=1e-7)

    def test_pb_cbf_respects_revoked_agent_fixed_input(self):
        zero = np.zeros(2)
        safe, feasible, iterations = solve_prediction_based_cbf_qp(
            a_nom={2: zero, 3: zero},
            positions={2: zero, 3: np.array([1.2, 0.0])},
            velocities={2: np.array([1.0, 0.0]), 3: zero},
            static_distance={(2, 3): 0.8},
            alpha=0.5,
            braking_accel=2.0,
            a_max=2.0,
            fixed_accelerations={3: zero},
        )
        self.assertTrue(feasible)
        self.assertLess(iterations, 100)
        np.testing.assert_allclose(safe[2], [-1.85, 0.0], atol=1e-7)
        np.testing.assert_array_equal(safe[3], zero)
        slack = -0.5 * (safe[2][0] - safe[3][0]) - 0.925
        self.assertGreaterEqual(slack, -1e-7)

    def test_infeasible_fallback_does_not_restore_revoked_authority(self):
        fixed = np.array([3.0, 0.0])
        safe, feasible, _ = solve_prediction_based_cbf_qp(
            a_nom={2: np.zeros(2), 3: fixed},
            positions={2: np.zeros(2), 3: np.array([0.1, 0.0])},
            velocities={2: np.array([1.0, 0.0]), 3: np.zeros(2)},
            static_distance={(2, 3): 2.0},
            alpha=4.0,
            braking_accel=2.0,
            a_max=2.0,
            infeasible_fallback="max_brake",
            fixed_accelerations={3: fixed},
        )
        self.assertFalse(feasible)
        np.testing.assert_array_equal(safe[3], fixed)

    def test_zocbf_beta_matches_exact_px4_position_coefficient(self):
        dt = 0.05
        tau = 0.7
        expected = beta_of_tau(dt, tau)
        actual = dt - tau * (1.0 - np.exp(-dt / tau))
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
