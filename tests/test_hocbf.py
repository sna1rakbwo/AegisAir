"""Frozen tests for the acceleration-aware centralized HOCBF."""

from __future__ import annotations

import unittest

import numpy as np

from swarm.ra.hocbf import solve_acceleration_qp


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


if __name__ == "__main__":
    unittest.main()
