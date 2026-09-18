"""Gate 0 tests for the Huang et al. PCBF adaptation."""

from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from swarm.ra.pcbf import PCBFConfig, solve_pcbf
from marllib.run_c1_sota_cbf_gazebo import _method_kwargs


class PCBFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PCBFConfig(horizon=24, iterations=1600, projection_iterations=800)

    def test_safe_hover_has_zero_slack_and_terminal_feasible(self) -> None:
        result = solve_pcbf(
            nominal_accelerations={0: np.zeros(2), 1: np.zeros(2)},
            positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
            velocities={0: np.zeros(2), 1: np.zeros(2)},
            safe_distances={(0, 1): 2.0}, config=self.config,
        )
        self.assertTrue(result.feasible)
        self.assertTrue(result.terminal_feasible)
        self.assertAlmostEqual(result.slack_sum, 0.0, places=8)
        self.assertEqual(result.status, "optimal_approx")

    def test_soft_constraint_is_positive_before_recovery(self) -> None:
        result = solve_pcbf(
            nominal_accelerations={0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])},
            positions={0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])},
            velocities={0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])},
            safe_distances={(0, 1): 2.0}, config=self.config,
        )
        self.assertTrue(result.feasible)
        self.assertGreater(result.slack_sum, 0.0)
        self.assertGreater(
            np.linalg.norm(result.accelerations[0] - np.array([2.0, 0.0])),
            1e-6,
        )

    def test_terminal_unreachable_fails_closed_with_braking(self) -> None:
        config = PCBFConfig(horizon=4, iterations=300, projection_iterations=200)
        result = solve_pcbf(
            nominal_accelerations={0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])},
            positions={0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])},
            velocities={0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])},
            safe_distances={(0, 1): 2.0}, config=config,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "fail_closed")
        self.assertIn(
            result.fail_closed_reason,
            {"terminal_infeasible", "terminal_projection_failed", "terminal_recovery_infeasible"},
        )
        np.testing.assert_allclose(result.accelerations[0], np.array([-2.0, 0.0]))
        np.testing.assert_allclose(result.accelerations[1], np.array([2.0, 0.0]))

    def test_replanning_after_first_control_remains_terminal_feasible(self) -> None:
        positions = {0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])}
        velocities = {0: np.zeros(2), 1: np.zeros(2)}
        nominal = {0: np.array([0.25, 0.0]), 1: np.array([-0.25, 0.0])}
        first = solve_pcbf(
            nominal_accelerations=nominal, positions=positions, velocities=velocities,
            safe_distances={(0, 1): 2.0}, config=self.config,
        )
        self.assertTrue(first.feasible)
        next_velocities = {i: velocities[i] + self.config.dt * first.accelerations[i] for i in positions}
        next_positions = {
            i: positions[i] + self.config.dt * velocities[i] + 0.5 * self.config.dt**2 * first.accelerations[i]
            for i in positions
        }
        second = solve_pcbf(
            nominal_accelerations=nominal, positions=next_positions, velocities=next_velocities,
            safe_distances={(0, 1): 2.0}, config=self.config,
        )
        self.assertTrue(second.feasible)
        self.assertTrue(second.terminal_feasible)

    def test_calibration_manifest_maps_only_to_explicit_pcbf(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "configs/c1_external_pcbf_sealed_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["trials"]), 20)
        self.assertIn("PCBF_HUANG_ECC2025", manifest["methods"])
        kwargs = _method_kwargs(
            "PCBF_HUANG_ECC2025", manifest["methods"]["PCBF_HUANG_ECC2025"], 0.70
        )
        self.assertTrue(kwargs["sampled_data"])
        self.assertEqual(kwargs["sampled_data_method"], "pcbf")
        self.assertEqual(kwargs["pcbf_horizon"], 24)


if __name__ == "__main__":
    unittest.main()
