"""Regression test for a non-empty PX4 tau-interval activation case."""

from __future__ import annotations

import unittest

from marllib.robust_activation_search import evaluate_case
from swarm.ra.margins import RuntimeAssuranceParams


class RobustActivationSearchTest(unittest.TestCase):
    def test_interval_changes_one_step_controller(self) -> None:
        case = evaluate_case(
            delta=0.4553558741865649,
            closing_speed=0.9,
            yielding_speed=0.1,
            dt=0.05,
            gamma=0.1,
            tau_hat=0.7,
            tau_min=0.53,
            tau_max=1.76,
            dense_points=50,
            params=RuntimeAssuranceParams(),
            kv=2.0,
            a_max=2.0,
        )
        self.assertIsNotNone(case)
        assert case is not None
        self.assertGreaterEqual(case["g_nominal_projected_controller"], 0.0)
        self.assertLess(case["g_worst_projected_robust_controller"], 0.0)
        self.assertGreater(case["delta_u_mps2"], 1e-8)
        self.assertGreaterEqual(case["dense_exact_g_min_robust_action"], case["dense_exact_g_min_nominal_action"])


if __name__ == "__main__":
    unittest.main()
