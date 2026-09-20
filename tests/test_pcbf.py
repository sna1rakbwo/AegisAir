"""Regression and formulation tests for the Huang et al. PCBF baseline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from marllib.run_c1_sota_cbf_gazebo import _method_kwargs
from swarm.ra.pcbf import PCBFConfig, _predict, solve_pcbf


class PCBFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PCBFConfig(horizon=24, multistart_count=3)

    def test_safe_hover_has_zero_pcbf_value_and_tracks_nominal_first_input(self) -> None:
        nominal = {0: np.array([0.25, 0.0]), 1: np.array([-0.25, 0.0])}
        result = solve_pcbf(
            nominal_accelerations=nominal,
            positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
            velocities={0: np.zeros(2), 1: np.zeros(2)},
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        self.assertTrue(result.feasible)
        self.assertTrue(result.terminal_feasible)
        self.assertLessEqual(result.value, self.config.acceptable_tolerance)
        self.assertEqual(result.status, "solved_local")
        self.assertEqual(result.stage1_status, "Solve_Succeeded")
        self.assertEqual(result.stage2_status, "Solve_Succeeded")
        self.assertTrue(result.tie_break_applied)
        self.assertLessEqual(
            result.max_constraint_violation,
            10.0 * self.config.acceptable_tolerance,
        )
        np.testing.assert_allclose(result.accelerations[0], nominal[0], atol=1e-5)
        np.testing.assert_allclose(result.accelerations[1], nominal[1], atol=1e-5)

    def test_initial_state_violation_is_included_in_pcbf_value(self) -> None:
        result = solve_pcbf(
            nominal_accelerations={
                0: np.array([2.0, 0.0]),
                1: np.array([-2.0, 0.0]),
            },
            positions={0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])},
            velocities={0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])},
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        initial_squared_distance_deficit = 2.0**2 - 1.6**2
        self.assertTrue(result.feasible)
        self.assertGreaterEqual(
            result.value + self.config.acceptable_tolerance,
            initial_squared_distance_deficit,
        )
        self.assertGreater(result.slack_sum, 0.0)
        self.assertGreater(
            np.linalg.norm(result.accelerations[0] - np.array([2.0, 0.0])),
            1e-3,
        )

    def test_terminal_set_is_hard_and_unreachable_problem_fails_closed(self) -> None:
        config = PCBFConfig(horizon=4, max_iterations=100, multistart_count=3)
        result = solve_pcbf(
            nominal_accelerations={
                0: np.array([2.0, 0.0]),
                1: np.array([-2.0, 0.0]),
            },
            positions={0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])},
            velocities={0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])},
            safe_distances={(0, 1): 2.0},
            config=config,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "fail_closed")
        self.assertEqual(result.fail_closed_reason, "stage1_infeasible_or_failed")
        np.testing.assert_allclose(result.accelerations[0], np.array([-2.0, 0.0]))
        np.testing.assert_allclose(result.accelerations[1], np.array([2.0, 0.0]))

    def test_fail_closed_preserves_revoked_agent_input(self) -> None:
        fixed = np.array([3.0, 0.0])
        config = PCBFConfig(horizon=2, velocity_bound_mps=2.0, multistart_count=1)
        result = solve_pcbf(
            nominal_accelerations={0: np.zeros(2), 1: fixed},
            positions={0: np.zeros(2), 1: np.zeros(2)},
            velocities={0: np.array([3.0, 0.0]), 1: np.zeros(2)},
            safe_distances={(0, 1): 2.0},
            config=config,
            fixed_accelerations={1: fixed},
        )
        self.assertFalse(result.feasible)
        np.testing.assert_array_equal(result.accelerations[1], fixed)

    def test_terminal_invariance_requires_fixed_agent_to_stop(self) -> None:
        fixed = np.array([0.5, 0.0])
        result = solve_pcbf(
            nominal_accelerations={0: np.zeros(2), 1: fixed},
            positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
            velocities={0: np.zeros(2), 1: np.zeros(2)},
            safe_distances={(0, 1): 2.0},
            config=PCBFConfig(horizon=4, multistart_count=1),
            fixed_accelerations={1: fixed},
        )
        self.assertFalse(result.feasible)
        self.assertEqual(
            result.fail_closed_reason,
            "fixed_input_has_no_invariant_terminal_policy",
        )
        np.testing.assert_array_equal(result.accelerations[1], fixed)

    def test_returned_plan_satisfies_stopped_separated_terminal_set(self) -> None:
        positions = {0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])}
        velocities = {0: np.array([0.2, 0.0]), 1: np.array([-0.2, 0.0])}
        result = solve_pcbf(
            nominal_accelerations={
                0: np.array([0.25, 0.0]),
                1: np.array([-0.25, 0.0]),
            },
            positions=positions,
            velocities=velocities,
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        self.assertTrue(result.feasible)
        self.assertIsNotNone(result.plan)
        predicted_p, predicted_v = _predict(
            result.plan,
            positions=positions,
            velocities=velocities,
            drone_ids=[0, 1],
            config=self.config,
        )
        np.testing.assert_allclose(predicted_v[0][-1], np.zeros(2), atol=1e-4)
        np.testing.assert_allclose(predicted_v[1][-1], np.zeros(2), atol=1e-4)
        self.assertGreaterEqual(
            np.linalg.norm(predicted_p[0][-1] - predicted_p[1][-1]),
            2.0 + self.config.terminal_buffer_m - 1e-4,
        )

    def test_primary_value_is_independent_of_nominal_tie_break(self) -> None:
        common = dict(
            positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
            velocities={0: np.zeros(2), 1: np.zeros(2)},
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        first = solve_pcbf(
            nominal_accelerations={0: np.array([0.3, 0.0]), 1: np.array([-0.3, 0.0])},
            **common,
        )
        second = solve_pcbf(
            nominal_accelerations={0: np.array([-0.4, 0.0]), 1: np.array([0.4, 0.0])},
            **common,
        )
        self.assertAlmostEqual(first.value, second.value, places=6)
        self.assertGreater(
            np.linalg.norm(first.accelerations[0] - second.accelerations[0]),
            0.5,
        )

    def test_shifted_warm_start_preserves_recovery_quality(self) -> None:
        positions = {0: np.array([-0.8, 0.0]), 1: np.array([0.8, 0.0])}
        velocities = {0: np.array([0.1, 0.0]), 1: np.array([-0.1, 0.0])}
        nominal = {0: np.array([2.0, 0.0]), 1: np.array([-2.0, 0.0])}
        first = solve_pcbf(
            nominal_accelerations=nominal,
            positions=positions,
            velocities=velocities,
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        self.assertTrue(first.feasible)
        self.assertIsNotNone(first.plan)
        predicted_p, predicted_v = _predict(
            first.plan,
            positions=positions,
            velocities=velocities,
            drone_ids=[0, 1],
            config=self.config,
        )
        next_positions = {drone: predicted_p[drone][1] for drone in positions}
        next_velocities = {drone: predicted_v[drone][1] for drone in velocities}
        cold = solve_pcbf(
            nominal_accelerations=nominal,
            positions=next_positions,
            velocities=next_velocities,
            safe_distances={(0, 1): 2.0},
            config=self.config,
        )
        warm = solve_pcbf(
            nominal_accelerations=nominal,
            positions=next_positions,
            velocities=next_velocities,
            safe_distances={(0, 1): 2.0},
            config=self.config,
            warm_start_plan=first.plan,
        )
        self.assertTrue(warm.feasible)
        self.assertTrue(warm.warm_start_used)
        self.assertLessEqual(
            warm.value,
            cold.value + 10.0 * self.config.acceptable_tolerance,
        )
        self.assertLessEqual(
            warm.max_constraint_violation,
            10.0 * self.config.acceptable_tolerance,
        )

    def test_rejects_malformed_warm_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "warm-start plan"):
            solve_pcbf(
                nominal_accelerations={0: np.zeros(2), 1: np.zeros(2)},
                positions={0: np.array([-3.0, 0.0]), 1: np.array([3.0, 0.0])},
                velocities={0: np.zeros(2), 1: np.zeros(2)},
                safe_distances={(0, 1): 2.0},
                config=self.config,
                warm_start_plan=np.zeros((2, 2, 2)),
            )

    def test_manifest_maps_to_two_stage_nonlinear_pcbf(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "configs/c1_external_pcbf_sealed_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["trials"]), 20)
        method = manifest["methods"]["PCBF_HUANG_ECC2025"]
        self.assertEqual(method["solver"], "casadi_ipopt")
        self.assertNotIn("lateral_candidates_mps2", method)
        self.assertNotIn("tracking_weight", method)
        kwargs = _method_kwargs("PCBF_HUANG_ECC2025", method, 0.70)
        self.assertTrue(kwargs["sampled_data"])
        self.assertEqual(kwargs["sampled_data_method"], "pcbf")
        self.assertEqual(kwargs["pcbf_horizon"], 24)
        self.assertEqual(kwargs["pcbf_multistart_count"], 3)
        self.assertEqual(kwargs["ra_params"].v_max, 1.5)
        self.assertEqual(kwargs["ra_params"].a_max, 2.0)


if __name__ == "__main__":
    unittest.main()
