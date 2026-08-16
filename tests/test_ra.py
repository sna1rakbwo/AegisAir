"""Phase 2 Risk-Adaptive Runtime Assurance tests."""

from __future__ import annotations

import unittest

import numpy as np

from swarm.ra.cbf import cbf_constraint, project_safe_action
from swarm.ra.margin import PairMarginTracker, normalized_margin
from swarm.ra.margins import (
    RuntimeAssuranceParams,
    closing_speed,
    dynamics_margin,
    dynamic_safety_boundary,
)
from swarm.ra.runtime_assurance import RuntimeAssurance
from swarm.ra.predictor import closest_point_of_approach, predicted_distance, predicted_min_margin
from swarm.safety import DroneSnapshot


class MarginsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = RuntimeAssuranceParams()

    def test_closing_speed(self) -> None:
        # Head-on: both moving toward each other.
        v_cl = closing_speed((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
        self.assertAlmostEqual(v_cl, 2.0)
        # Receding: closing speed should be zero.
        v_cl = closing_speed((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(v_cl, 0.0)

    def test_dynamics_margin_monotonic(self) -> None:
        self.assertGreater(dynamics_margin(2.0, self.params), dynamics_margin(1.0, self.params))

    def test_dynamic_safety_boundary_components(self) -> None:
        base = dynamic_safety_boundary(closing_speed=0.0, perception_sigma_i=0.0, perception_sigma_j=0.0, aoi=0.0, params=self.params)
        with_perception = dynamic_safety_boundary(closing_speed=0.0, perception_sigma_i=0.5, perception_sigma_j=0.0, aoi=0.0, params=self.params)
        with_aoi = dynamic_safety_boundary(closing_speed=0.0, perception_sigma_i=0.0, perception_sigma_j=0.0, aoi=1.0, params=self.params)
        self.assertGreater(with_perception, base)
        self.assertGreater(with_aoi, base)


class MarginTest(unittest.TestCase):
    def test_normalized_margin(self) -> None:
        self.assertGreater(normalized_margin(2.0, 1.0), 0.0)
        self.assertAlmostEqual(normalized_margin(1.0, 1.0), 0.0)
        self.assertLess(normalized_margin(0.5, 1.0), 0.0)

    def test_degradation_positive_when_margin_drops(self) -> None:
        tracker = PairMarginTracker(RuntimeAssuranceParams())
        tracker.update(1.0, 0.0)
        g = tracker.update(0.5, 0.1)
        self.assertGreater(g, 0.0)


class CbfTest(unittest.TestCase):
    def test_safe_action_stays_within_limit(self) -> None:
        a, b = cbf_constraint(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 1.0, 1.0)
        u = project_safe_action(np.array([2.0, 0.0]), [(a, b)], v_max=1.5)
        self.assertLessEqual(np.linalg.norm(u), 1.5)

    def test_projection_satisfies_constraint(self) -> None:
        a, b = cbf_constraint(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0]), 1.0, 1.0)
        u = project_safe_action(np.array([1.0, 0.0]), [(a, b)], v_max=2.0)
        self.assertGreaterEqual(np.dot(a, u) + 1e-9, b)


class RuntimeAssuranceHocbfTest(unittest.TestCase):
    def test_hocbf_filter_is_a_real_method(self) -> None:
        ra = RuntimeAssurance(use_hocbf=True)
        snapshots = {
            0: DroneSnapshot(
                drone_id=0, position=(-2.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0)
            ),
            1: DroneSnapshot(
                drone_id=1, position=(2.0, 0.0, 0.0), velocity=(-1.0, 0.0, 0.0)
            ),
        }
        nominal = {0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])}
        results = ra.filter(snapshots, nominal, t=0.0)
        self.assertEqual(set(results), {0, 1})
        self.assertIn(results[0].mode, {"normal", "warning", "override"})


class PredictorTest(unittest.TestCase):
    def test_predicted_distance_head_on(self) -> None:
        # Two agents 4 m apart closing at 2 m/s: after 1 s they are 2 m apart.
        d = predicted_distance(
            (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            1.0,
        )
        self.assertAlmostEqual(d, 2.0)

    def test_cpa_head_on(self) -> None:
        cpa = closest_point_of_approach(
            (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            horizon=5.0,
        )
        self.assertAlmostEqual(cpa.d_cpa, 0.0)
        self.assertAlmostEqual(cpa.t_cpa, 2.0)

    def test_predicted_margin_negative_head_on(self) -> None:
        rho_hat, tau = predicted_min_margin(
            p_i=(-2.0, 0.0, 0.0),
            p_j=(2.0, 0.0, 0.0),
            v_i=(1.0, 0.0, 0.0),
            v_j=(-1.0, 0.0, 0.0),
            d_safe=1.0,
            horizon=3.0,
        )
        self.assertLess(rho_hat, 0.0)


class RuntimeAssuranceTest(unittest.TestCase):
    def _snapshot(self, drone_id, position, velocity):
        from swarm.safety import DroneSnapshot
        return DroneSnapshot(drone_id=drone_id, position=position, velocity=velocity)

    def test_head_on_override(self) -> None:
        ra = RuntimeAssurance()
        snapshots = {
            1: self._snapshot(1, (-2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            2: self._snapshot(2, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        }
        nominal = {1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])}
        results = ra.filter(snapshots, nominal, t=0.0)
        # Both agents should be steered away from each other.
        self.assertEqual(results[1].mode, "override")
        self.assertEqual(results[2].mode, "override")
        self.assertLess(results[1].safe_action[0], results[1].nominal_action[0])
        self.assertGreater(results[2].safe_action[0], results[2].nominal_action[0])

    def test_far_apart_is_normal(self) -> None:
        ra = RuntimeAssurance()
        snapshots = {
            1: self._snapshot(1, (-10.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            2: self._snapshot(2, (10.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        }
        nominal = {1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])}
        results = ra.filter(snapshots, nominal, t=0.0)
        self.assertEqual(results[1].mode, "normal")
        self.assertFalse(results[1].intervened)

    def test_head_on_is_proactive(self) -> None:
        ra = RuntimeAssurance()
        snapshots = {
            1: self._snapshot(1, (-2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            2: self._snapshot(2, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        }
        nominal = {1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])}
        results = ra.filter(snapshots, nominal, t=0.0)
        self.assertTrue(results[1].proactive)
        self.assertLess(results[1].predicted_margin, 0.0)


if __name__ == "__main__":
    unittest.main()
