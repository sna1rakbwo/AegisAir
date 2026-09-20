from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from marllib.analyze_c_recoverability_admission_calibration import (
    _reconstruct_solver_feasible_publication_failures,
    analyze,
)
from marllib.run_c_recoverability_admission_gazebo import (
    CONDITIONS,
    IMPLEMENTATION_VERSION,
    PROTOCOL_IDS,
    _admission_config,
    _published_command_audit_summary,
)
from swarm.recovery.recoverability_admission import (
    RecoverabilityAdmissionCoordinator,
)
from swarm.safety import DroneSnapshot


ROOT = Path(__file__).resolve().parents[1]


class RecoverabilityAdmissionProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "configs/c_recoverability_admission_calibration_v4.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_freezes_three_comparators_and_both_outcomes(self) -> None:
        self.assertIn(self.manifest["protocol_id"], PROTOCOL_IDS)
        self.assertEqual(
            self.manifest["implementation_version"], IMPLEMENTATION_VERSION
        )
        self.assertEqual(
            self.manifest["adapter_mqtt_publish_mode"],
            "nonblocking_enqueue_rc_checked",
        )
        self.assertNotIn(
            "aegisair-c-recoverability-admission-calibration-v3",
            PROTOCOL_IDS,
        )
        self.assertNotIn(
            "aegisair-c-recoverability-admission-sealed-v1",
            PROTOCOL_IDS,
        )
        self.assertEqual(set(self.manifest["conditions"]), CONDITIONS)
        expected = {
            geometry["expected_admission"]
            for geometry in self.manifest["geometries"].values()
        }
        self.assertEqual(expected, {"admit", "reject"})
        for trial in self.manifest["trials"]:
            self.assertEqual(set(trial["condition_order"]), CONDITIONS)

    def test_v4_only_changes_runtime_transport_and_protocol_identity(self) -> None:
        historical = json.loads(
            (ROOT / "configs/c_recoverability_admission_calibration_v3.json").read_text(
                encoding="utf-8"
            )
        )
        for key in (
            "rate_hz",
            "max_steps",
            "goal_epsilon",
            "drone_ids",
            "failed_drone",
            "mission_change",
            "change_step",
            "velocity_command_mode",
            "tau_command_s",
            "conditions",
            "recoverability_admission",
            "ra_config",
            "geometries",
        ):
            self.assertEqual(self.manifest[key], historical[key], key)
        self.assertEqual(
            [
                (trial["geometry_id"], trial["seed"], trial["condition_order"])
                for trial in self.manifest["trials"]
            ],
            [
                (trial["geometry_id"], trial["seed"], trial["condition_order"])
                for trial in historical["trials"]
            ],
        )

    def test_admission_parameters_are_frozen(self) -> None:
        config = _admission_config(
            self.manifest["recoverability_admission"],
            rate_hz=float(self.manifest["rate_hz"]),
            execution_tau_s=float(
                self.manifest["ra_config"]["execution_tau_s"]
            ),
            command_feedforward_tau_s=float(self.manifest["tau_command_s"]),
        )
        self.assertEqual(config.clearance_m, 2.4)
        self.assertEqual(config.braking_accel_mps2, 2.0)
        self.assertEqual(config.ring_samples, 32)
        self.assertEqual(config.max_route_length_m, 18.0)
        self.assertEqual(config.dt_s, 0.05)
        self.assertEqual(config.rollout_horizon_s, 16.0)
        self.assertEqual(config.reaction_delay_s, 0.05)
        self.assertEqual(config.tracking_error_buffer_m, 0.20)
        self.assertEqual(config.effective_clearance_m, 2.60)
        self.assertEqual(config.conservative_braking_accel_mps2, 1.75)

    def test_expected_admit_starts_clear_of_effective_envelope(self) -> None:
        config = _admission_config(
            self.manifest["recoverability_admission"],
            rate_hz=float(self.manifest["rate_hz"]),
            execution_tau_s=float(
                self.manifest["ra_config"]["execution_tau_s"]
            ),
            command_feedforward_tau_s=float(self.manifest["tau_command_s"]),
        )
        failed = str(self.manifest["failed_drone"])
        healthy = next(
            str(drone)
            for drone in self.manifest["drone_ids"]
            if str(drone) != failed
        )
        for geometry_id, geometry in self.manifest["geometries"].items():
            if geometry["expected_admission"] != "admit":
                continue
            failed_position = geometry["reset_starts"][failed]
            healthy_position = geometry["reset_starts"][healthy]
            separation = sum(
                (healthy_position[axis] - failed_position[axis]) ** 2
                for axis in (0, 1)
            ) ** 0.5
            self.assertGreaterEqual(
                separation,
                config.effective_clearance_m,
                geometry_id,
            )

    def test_zero_velocity_geometry_sanity_matches_declared_outcomes(self) -> None:
        config = _admission_config(
            self.manifest["recoverability_admission"],
            rate_hz=float(self.manifest["rate_hz"]),
            execution_tau_s=float(
                self.manifest["ra_config"]["execution_tau_s"]
            ),
            command_feedforward_tau_s=float(self.manifest["tau_command_s"]),
        )
        base_change = self.manifest["mission_change"]
        for geometry_id, geometry in self.manifest["geometries"].items():
            snapshots = {
                int(drone): DroneSnapshot(
                    int(drone),
                    tuple(position),
                    velocity=(0.0, 0.0, 0.0),
                )
                for drone, position in geometry["reset_starts"].items()
            }
            coordinator = RecoverabilityAdmissionCoordinator(config)
            coordinator.step(
                step=int(self.manifest["change_step"]),
                snapshots=snapshots,
                base_goals={
                    int(drone): tuple(goal)
                    for drone, goal in geometry["base_goals"].items()
                },
                mission_change=dict(base_change),
            )
            actual = (
                "admit"
                if coordinator.summary()["state"] == "admitted"
                else "reject"
            )
            self.assertEqual(actual, geometry["expected_admission"], geometry_id)

    def test_empty_analysis_is_no_go_not_complete(self) -> None:
        result = analyze(self.manifest, [])
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["complete"])
        self.assertFalse(result["anti_vacuity"])

    def test_summary_preserves_solver_feasible_publication_failures(self) -> None:
        summary = _published_command_audit_summary(
            {
                "published_command_mismatch_count": 1,
                "published_command_constraint_unknown_count": 2,
                "published_command_constraint_failure_count": 3,
                "published_constraint_failure_while_solver_feasible_count": 4,
            }
        )
        self.assertEqual(
            summary,
            {
                "published_command_mismatch_count": 1,
                "published_command_constraint_unknown_count": 2,
                "published_command_constraint_failure_count": 3,
                "published_constraint_failure_while_solver_feasible_count": 4,
            },
        )

    def test_old_summary_metric_can_be_reconstructed_from_trajectory(self) -> None:
        rows = [
            {
                "drones": {
                    "2": {
                        "published_command_constraint_ok": False,
                        "solver_feasible": False,
                    },
                    "3": {
                        "published_command_constraint_ok": False,
                        "solver_feasible": False,
                    },
                }
            },
            {
                "drones": {
                    "2": {
                        "published_command_constraint_ok": False,
                        "solver_feasible": True,
                    },
                    "3": {
                        "published_command_constraint_ok": False,
                        "solver_feasible": True,
                    },
                }
            },
        ]
        self.assertEqual(
            _reconstruct_solver_feasible_publication_failures(rows),
            2,
        )

    def test_immediate_comparator_is_reported_without_gating_go(self) -> None:
        rows = []
        for trial in self.manifest["trials"]:
            expected = self.manifest["geometries"][trial["geometry_id"]][
                "expected_admission"
            ]
            common = {
                "trial_id": trial["trial_id"],
                "expected_admission": expected,
                "infrastructure_valid": True,
                "safety_bypass_count": 0,
                "published_command_mismatch_count": 0,
                "published_command_constraint_unknown_count": 0,
                "published_command_constraint_failure_count": 0,
                "published_constraint_failure_while_solver_feasible_count": 0,
                "collision": False,
                "min_rho": 0.1,
                "ra_solve_latency_summary_ms": {
                    "p99": 1.0,
                    "deadline_misses": 0,
                },
                "trajectory_audit": {
                    "ra_bypass_count": 0,
                    "failed_authority_revoked_all_steps": True,
                    "failed_horizontal_command_zero_all_steps": True,
                    "selected_qp_infeasible_steps": 0,
                    "hold_goal_frozen": True,
                },
            }
            admitted = expected == "admit"
            rows.append(
                {
                    **common,
                    "condition": "RECOVERABILITY_ADMISSION_RA",
                    "post_failure_critical_reached": admitted,
                    "recoverability_admission": {
                        "admission_count": int(admitted),
                        "rejection_count": int(not admitted),
                        "plans_committed": int(admitted),
                        "unsafe_commit_count": 0,
                        "decision_latency_ms": 1.0,
                        "completed": admitted,
                    },
                }
            )
            rows.append(
                {
                    **common,
                    "condition": "RA_ONLY_HOLD",
                    "post_failure_critical_reached": False,
                    "recoverability_admission": {"plans_committed": 0},
                }
            )
            rows.append(
                {
                    **common,
                    "condition": "IMMEDIATE_COMMIT_RA",
                    "collision": True,
                    "min_rho": -0.5,
                    "published_command_constraint_failure_count": 2,
                    "trajectory_audit": {
                        **common["trajectory_audit"],
                        "selected_qp_infeasible_steps": 1,
                    },
                    "post_failure_critical_reached": True,
                    "recoverability_admission": None,
                }
            )
        result = analyze(self.manifest, rows)
        self.assertEqual(result["decision"], "GO")
        self.assertTrue(result["complete"])
        self.assertTrue(
            all(item["passed"] for item in result["immediate_integrity_checks"])
        )

    def test_qualification_references_exact_calibration_manifest(self) -> None:
        calibration_path = ROOT / "configs/c_recoverability_admission_calibration_v4.json"
        qualification = json.loads(
            (
                ROOT / "configs/c_recoverability_admission_qualification_v4.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            qualification["implementation_version"],
            IMPLEMENTATION_VERSION,
        )
        self.assertEqual(
            qualification["parent_calibration_manifest_sha256"],
            hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
