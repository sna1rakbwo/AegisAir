"""Frozen tests for the acceleration-aware centralized HOCBF."""

from __future__ import annotations

import unittest

import numpy as np

from swarm.ra.hocbf import (
    beta_of_tau,
    minimum_acceleration_box_reserve,
    solve_acceleration_qp,
    solve_robust_sampled_data_qp,
    solve_sampled_data_qp,
)


class HocbfQpTest(unittest.TestCase):
    def _four_drone_crossing(self) -> tuple[list, dict, dict, dict, dict]:
        drone_ids = [0, 1, 2, 3]
        positions = {
            0: np.array([-3.0, 0.8]),
            1: np.array([3.0, -0.8]),
            2: np.array([-3.0, -0.8]),
            3: np.array([3.0, 0.8]),
        }
        velocities = {
            0: np.array([1.0, 0.0]),
            1: np.array([-1.0, 0.0]),
            2: np.array([1.0, 0.0]),
            3: np.array([-1.0, 0.0]),
        }
        a_nom = {i: np.zeros(2) for i in drone_ids}
        d_safe = {}
        for a in range(4):
            for b in range(a + 1, 4):
                d_safe[(a, b)] = 2.0
        return drone_ids, positions, velocities, a_nom, d_safe

    def test_qp_satisfies_all_pair_constraints(self) -> None:
        drone_ids, positions, velocities, a_nom, d_safe = self._four_drone_crossing()
        a_safe, feasible, _ = solve_acceleration_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            d_safe=d_safe,
            k1=1.0,
            k2=1.0,
            a_max=2.0,
        )
        self.assertTrue(feasible)
        for (i, j), s in d_safe.items():
            r = positions[i] - positions[j]
            v = velocities[i] - velocities[j]
            lhs = 2.0 * float(np.dot(r, a_safe[i] - a_safe[j]))
            rhs = (
                -2.0 * float(np.dot(v, v))
                - 2.0 * 2.0 * float(np.dot(r, v))
                - 1.0 * (float(np.dot(r, r)) - s * s)
            )
            self.assertGreaterEqual(lhs, rhs - 1e-6)

    def test_qp_respects_acceleration_box(self) -> None:
        drone_ids, positions, velocities, a_nom, d_safe = self._four_drone_crossing()
        a_safe, feasible, _ = solve_acceleration_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            d_safe=d_safe,
            k1=1.0,
            k2=1.0,
            a_max=2.0,
        )
        for i in drone_ids:
            self.assertLessEqual(float(np.linalg.norm(a_safe[i])), 2.0 + 1e-6)

    def test_no_pairs_returns_nominal_acceleration(self) -> None:
        a_safe, feasible, _ = solve_acceleration_qp(
            a_nom={0: np.array([0.5, 0.0])},
            positions={0: np.array([0.0, 0.0])},
            velocities={0: np.array([0.0, 0.0])},
            d_safe={},
            k1=1.0,
            k2=1.0,
            a_max=2.0,
        )
        self.assertTrue(feasible)
        np.testing.assert_allclose(a_safe[0], [0.5, 0.0])

    def test_qp_uses_only_remaining_authority_for_revoked_agent(self) -> None:
        fixed = np.zeros(2)
        a_safe, feasible, iterations = solve_acceleration_qp(
            a_nom={0: np.zeros(2), 1: fixed},
            positions={0: np.array([0.0, 0.0]), 1: np.array([1.2, 0.0])},
            velocities={0: np.array([1.0, 0.0]), 1: np.zeros(2)},
            d_safe={(0, 1): 0.8},
            k1=0.5,
            k2=0.5,
            a_max=2.0,
            fixed_accelerations={1: fixed},
        )
        self.assertTrue(feasible)
        self.assertLess(iterations, 100)
        np.testing.assert_allclose(a_safe[0], [-1.0 / 12.0, 0.0], atol=1e-7)
        np.testing.assert_array_equal(a_safe[1], fixed)

    def test_reserve_excludes_revoked_agent_control_budget(self) -> None:
        common = dict(
            positions={0: np.array([0.0, 0.0]), 1: np.array([1.2, 0.0])},
            velocities={0: np.array([1.0, 0.0]), 1: np.zeros(2)},
            d_safe={(0, 1): 0.8},
            k1=0.5,
            k2=0.5,
            a_max=2.0,
        )
        cooperative = minimum_acceleration_box_reserve(**common)
        revoked = minimum_acceleration_box_reserve(
            **common, fixed_accelerations={1: np.zeros(2)}
        )
        self.assertLess(revoked, cooperative)

    def test_infeasible_max_brake_uses_full_available_acceleration(self) -> None:
        positions = {0: np.array([-0.5, 0.0]), 1: np.array([0.5, 0.0])}
        velocities = {0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])}
        a_safe, feasible, _ = solve_acceleration_qp(
            a_nom={0: np.zeros(2), 1: np.zeros(2)},
            positions=positions,
            velocities=velocities,
            d_safe={(0, 1): 2.0},
            k1=4.0,
            k2=6.0,
            a_max=2.0,
            infeasible_fallback="max_brake",
        )
        self.assertFalse(feasible)
        np.testing.assert_allclose(a_safe[0], [-2.0, 0.0])
        np.testing.assert_allclose(a_safe[1], [2.0, 0.0])

    def test_box_reserve_is_negative_when_pair_is_unrecoverable(self) -> None:
        reserve = minimum_acceleration_box_reserve(
            positions={0: np.array([-0.5, 0.0]), 1: np.array([0.5, 0.0])},
            velocities={0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])},
            d_safe={(0, 1): 2.0},
            k1=4.0,
            k2=6.0,
            a_max=2.0,
        )
        self.assertLess(reserve, 0.0)

    def test_sampled_data_qp_lag_alpha_reduces_responsiveness(self) -> None:
        a_nom = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0])}
        positions = {0: np.array([-2.0, 0.0]), 1: np.array([2.0, 0.0])}
        velocities = {0: np.array([0.5, 0.0]), 1: np.array([-0.5, 0.0])}
        s_now = {(0, 1): 1.0}
        s_next = {(0, 1): 1.0}
        a_instant, feasible_instant, _ = solve_sampled_data_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            s_now=s_now,
            s_next=s_next,
            dt=0.05,
            gamma=0.1,
            a_max=2.0,
            alpha=1.0,
        )
        a_lag, feasible_lag, _ = solve_sampled_data_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            s_now=s_now,
            s_next=s_next,
            dt=0.05,
            gamma=0.1,
            a_max=2.0,
            alpha=0.5,
        )
        self.assertTrue(feasible_instant)
        self.assertTrue(feasible_lag)
        # The lag model must not be less conservative than the instant model.
        self.assertLessEqual(float(a_lag[0][0]), float(a_instant[0][0]) + 1e-6)

    def test_beta_of_tau_exact_discretization(self) -> None:
        dt = 0.05
        self.assertAlmostEqual(beta_of_tau(dt, 0.0), dt)
        self.assertGreater(beta_of_tau(dt, 0.1), beta_of_tau(dt, 0.2))
        self.assertLess(beta_of_tau(dt, 1e9), 1e-3)

    def test_robust_sampled_data_qp_bounds_and_hard_brake(self) -> None:
        a_nom = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0])}
        positions = {0: np.array([-2.0, 0.0]), 1: np.array([2.0, 0.0])}
        velocities = {0: np.array([0.5, 0.0]), 1: np.array([-0.5, 0.0])}
        s_now = {(0, 1): 1.0}
        s_next = {(0, 1): 1.0}
        a_safe, feasible, _ = solve_robust_sampled_data_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            s_now=s_now,
            s_next=s_next,
            dt=0.05,
            gamma=0.1,
            a_max=2.0,
            beta={0: (0.01, 0.02), 1: (0.01, 0.02)},
        )
        self.assertTrue(feasible)
        self.assertLessEqual(float(np.linalg.norm(a_safe[0])), 2.0 + 1e-6)

        positions = {0: np.array([-0.6, 0.0]), 1: np.array([0.6, 0.0])}
        velocities = {0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])}
        s_now = {(0, 1): 1.2}
        s_next = {(0, 1): 1.2}
        a_safe, feasible, _ = solve_robust_sampled_data_qp(
            a_nom=a_nom,
            positions=positions,
            velocities=velocities,
            s_now=s_now,
            s_next=s_next,
            dt=0.05,
            gamma=0.1,
            a_max=2.0,
            beta={0: (0.01, 0.02), 1: (0.01, 0.02)},
        )
        self.assertFalse(feasible)
        np.testing.assert_allclose(a_safe[0], [-1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
