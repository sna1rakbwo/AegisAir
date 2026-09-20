"""Phase 2 Risk-Adaptive Runtime Assurance tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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
from swarm.ra.predictor import (
    PredictiveMonitor,
    closest_point_of_approach,
    predicted_distance,
    predicted_min_margin,
)
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


class PredictiveMonitorTest(unittest.TestCase):
    def test_filtered_ca_uses_observed_acceleration_direction(self) -> None:
        monitor = PredictiveMonitor()
        monitor.update_acceleration(7, (0.0, 0.0, 0.0), 0.1)
        monitor.update_acceleration(7, (0.0, 1.0, 0.0), 0.1)
        direction = monitor._acc_filter(7).direction
        np.testing.assert_allclose(direction, (0.0, 1.0, 0.0), atol=1e-12)


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
    def test_pb_filter_pins_published_action_after_authority_revocation(self) -> None:
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(
                d0=0.8,
                beta=0.0,
                degradation_dt=0.05,
            ),
            perception_sigma=0.0,
            sampled_data=True,
            sampled_data_method="pb_cbf",
            constraint_boundary="static",
            pb_alpha=0.5,
            pb_braking_accel=2.0,
            a_max=2.0,
            command_feedforward_tau_s=1.0,
        )
        snapshots = {
            2: DroneSnapshot(
                2, (0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0)
            ),
            3: DroneSnapshot(
                3, (1.2, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
            ),
        }
        results = ra.filter(
            snapshots,
            {2: np.array([1.0, 0.0]), 3: np.zeros(2)},
            t=0.0,
            fixed_actions={3: np.zeros(2)},
        )
        self.assertTrue(results[2].feasible)
        np.testing.assert_allclose(results[2].a_safe, [-1.85, 0.0], atol=1e-7)
        np.testing.assert_allclose(results[3].safe_action, [0.0, 0.0])
        self.assertFalse(results[3].control_authority)
        self.assertEqual(results[3].fixed_action, (0.0, 0.0))

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

    def test_hocbf_exposes_acceleration_for_px4_feedforward(self) -> None:
        ra = RuntimeAssurance(
            use_hocbf=True,
            command_feedforward_tau_s=0.7,
        )
        snapshots = {
            0: DroneSnapshot(
                drone_id=0, position=(-4.0, 0.5, 2.5), velocity=(0.0, 0.0, 0.0)
            ),
            1: DroneSnapshot(
                drone_id=1, position=(4.0, -0.5, 2.5), velocity=(0.0, 0.0, 0.0)
            ),
        }
        nominal = {0: np.array([1.5, -1.5]), 1: np.array([-1.5, 1.5])}
        results = ra.filter(snapshots, nominal, t=0.0)
        for result in results.values():
            self.assertIsNotNone(result.a_nom)
            self.assertIsNotNone(result.a_safe)
            self.assertIsNotNone(result.d_safe)
            self.assertIsNotNone(result.feasible)
            self.assertGreater(np.linalg.norm(result.a_safe), 0.0)

    def test_published_hocbf_audit_checks_clipped_joint_command(self) -> None:
        command_scale = 0.7
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(
                d0=0.8,
                beta=0.0,
                degradation_dt=0.05,
            ),
            perception_sigma=0.0,
            use_hocbf=True,
            a_max=2.0,
            v_max=1.5,
            command_feedforward_tau_s=command_scale,
        )
        snapshots = {
            0: DroneSnapshot(0, (-4.0, 0.0, 2.5), velocity=(0.0, 0.0, 0.0)),
            1: DroneSnapshot(1, (4.0, 0.0, 2.5), velocity=(0.0, 0.0, 0.0)),
        }
        results = ra.filter(
            snapshots,
            {0: np.array([1.5, 1.5]), 1: np.array([-1.5, -1.5])},
            t=0.0,
        )
        published_accelerations = {
            drone: np.asarray(result.safe_action) / command_scale
            for drone, result in results.items()
        }
        constraint_ok, minimum_slack = ra.audit_published_accelerations(
            published_accelerations
        )
        self.assertTrue(constraint_ok)
        self.assertIsNotNone(minimum_slack)
        self.assertGreaterEqual(minimum_slack, -1e-6)
        self.assertFalse(results[0].vel_saturated)

        published_accelerations[0] = np.array([2.1, 0.0])
        constraint_ok, minimum_slack = ra.audit_published_accelerations(
            published_accelerations
        )
        self.assertFalse(constraint_ok)
        self.assertLess(minimum_slack, 0.0)

    def test_published_audit_keeps_revoked_agent_outside_control_box(self) -> None:
        command_scale = 0.7
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(
                d0=0.8,
                beta=0.0,
                degradation_dt=0.05,
            ),
            perception_sigma=0.0,
            use_hocbf=True,
            a_max=2.0,
            command_feedforward_tau_s=command_scale,
        )
        snapshots = {
            0: DroneSnapshot(0, (-4.0, 0.0, 2.5), velocity=(1.5, 0.0, 0.0)),
            1: DroneSnapshot(1, (4.0, 0.0, 2.5), velocity=(0.0, 0.0, 0.0)),
        }
        results = ra.filter(
            snapshots,
            {0: np.zeros(2), 1: np.zeros(2)},
            t=0.0,
            fixed_actions={0: np.zeros(2)},
        )
        published_accelerations = {
            drone: (
                np.asarray(result.safe_action)
                - np.asarray(snapshots[drone].velocity[:2])
            )
            / command_scale
            for drone, result in results.items()
        }
        self.assertLess(published_accelerations[0][0], -2.0)
        constraint_ok, minimum_slack = ra.audit_published_accelerations(
            published_accelerations
        )
        self.assertTrue(constraint_ok)
        self.assertGreaterEqual(minimum_slack, -1e-6)

    def test_pcbf_is_an_explicit_runtime_assurance_method(self) -> None:
        ra = RuntimeAssurance(
            sampled_data=True,
            sampled_data_method="pcbf",
            tau_px4=0.7,
            command_feedforward_tau_s=0.7,
            pcbf_horizon=24,
        )
        snapshots = {
            0: DroneSnapshot(0, (-3.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            1: DroneSnapshot(1, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        }
        nominal = {0: np.array([0.25, 0.0]), 1: np.array([-0.25, 0.0])}
        results = ra.filter(snapshots, nominal, t=0.0)
        for result in results.values():
            self.assertEqual(result.selected_filter, "pcbf")
            self.assertEqual(result.pcbf_status, "solved_local")
            self.assertEqual(result.pcbf_stage1_status, "Solve_Succeeded")
            self.assertEqual(result.pcbf_stage2_status, "Solve_Succeeded")
            self.assertTrue(result.pcbf_terminal_feasible)
            self.assertIsNotNone(result.pcbf_value)
            self.assertIsNotNone(result.pcbf_slack_sum)
            self.assertIsNotNone(result.pcbf_tracking_cost)
            self.assertIsNotNone(result.pcbf_max_constraint_violation)
            self.assertTrue(result.pcbf_tie_break_applied)
            self.assertFalse(result.pcbf_warm_start_used)
            self.assertIsNone(result.pcbf_fail_closed_reason)
            self.assertTrue(result.feasible)
        results = ra.filter(snapshots, nominal, t=0.05)
        for result in results.values():
            self.assertTrue(result.pcbf_warm_start_used)

    def test_hocbf_v4_switches_before_predicted_infeasibility(self) -> None:
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(degradation_dt=0.05),
            use_hocbf=True,
            hocbf_k1=4.0,
            hocbf_k2=4.0,
            tau_px4=0.7,
            command_feedforward_tau_s=0.7,
            hocbf_boundary_guard=1.0,
            hocbf_boundary_buffer_m=0.3,
            hocbf_pb_recovery=True,
            hocbf_recovery_alpha=0.5,
            hocbf_recovery_braking_accel=2.0,
            hocbf_recovery_boundary_buffer_m=0.3,
            hocbf_recovery_clear_steps=3,
        )
        snapshots = {
            0: DroneSnapshot(0, (-2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            1: DroneSnapshot(1, (2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        }
        nominal = {0: np.array([1.5, 0.0]), 1: np.array([-1.5, 0.0])}
        primary = {0: np.array([-0.5, 0.0]), 1: np.array([0.5, 0.0])}
        backup = {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0])}
        with patch(
            "swarm.ra.runtime_assurance.solve_acceleration_qp",
            side_effect=[(primary, True, 2), (primary, False, 2)],
        ), patch(
            "swarm.ra.runtime_assurance.solve_prediction_based_cbf_qp",
            return_value=(backup, True, 2),
        ) as pb_solver:
            results = ra.filter(snapshots, nominal, t=0.0)

        self.assertTrue(pb_solver.called)
        self.assertEqual(results[0].selected_filter, "pb_recovery")
        self.assertTrue(results[0].primary_feasible)
        self.assertFalse(results[0].predictive_feasible)
        self.assertTrue(results[0].recovery_active)
        self.assertEqual(results[0].recovery_reason, "predictive_infeasible")
        self.assertTrue(results[0].feasible)
        self.assertEqual(ra.hocbf_recovery_entries, 1)
        self.assertEqual(ra.hocbf_recovery_steps, 1)

    def test_hocbf_v4_hysteresis_requires_clean_cycles(self) -> None:
        ra = RuntimeAssurance(
            params=RuntimeAssuranceParams(degradation_dt=0.05),
            use_hocbf=True,
            tau_px4=0.7,
            command_feedforward_tau_s=0.7,
            hocbf_pb_recovery=True,
            hocbf_recovery_clear_steps=2,
        )
        snapshots = {
            0: DroneSnapshot(0, (-3.0, 0.0, 0.0), (0.2, 0.0, 0.0)),
            1: DroneSnapshot(1, (3.0, 0.0, 0.0), (-0.2, 0.0, 0.0)),
        }
        nominal = {0: np.array([0.2, 0.0]), 1: np.array([-0.2, 0.0])}
        safe = {0: np.zeros(2), 1: np.zeros(2)}
        # First cycle enters recovery; the following two cycles are clean.
        feasibility = [
            (safe, True, 1), (safe, False, 1),
            (safe, True, 1), (safe, True, 1),
            (safe, True, 1), (safe, True, 1),
        ]
        with patch(
            "swarm.ra.runtime_assurance.solve_acceleration_qp",
            side_effect=feasibility,
        ), patch(
            "swarm.ra.runtime_assurance.solve_prediction_based_cbf_qp",
            return_value=(safe, True, 1),
        ):
            first = ra.filter(snapshots, nominal, t=0.0)[0]
            second = ra.filter(snapshots, nominal, t=0.05)[0]
            third = ra.filter(snapshots, nominal, t=0.10)[0]

        self.assertTrue(first.recovery_active)
        self.assertTrue(second.recovery_active)
        self.assertFalse(third.recovery_active)
        self.assertEqual(third.selected_filter, "hocbf")


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
