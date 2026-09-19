from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from marllib.analyze_c_recoverability_admission_calibration import analyze
from marllib.run_c_recoverability_admission_gazebo import (
    CONDITIONS,
    IMPLEMENTATION_VERSION,
    PROTOCOL_IDS,
    _admission_config,
)
from swarm.recovery.recoverability_admission import (
    RecoverabilityAdmissionCoordinator,
)
from swarm.safety import DroneSnapshot


ROOT = Path(__file__).resolve().parents[1]


class RecoverabilityAdmissionProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "configs/c_recoverability_admission_calibration_v2.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_freezes_three_comparators_and_both_outcomes(self) -> None:
        self.assertIn(self.manifest["protocol_id"], PROTOCOL_IDS)
        self.assertEqual(
            self.manifest["implementation_version"], IMPLEMENTATION_VERSION
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

    def test_qualification_references_exact_calibration_manifest(self) -> None:
        calibration_path = (
            ROOT / "configs/c_recoverability_admission_calibration_v2.json"
        )
        qualification = json.loads(
            (
                ROOT / "configs/c_recoverability_admission_qualification_v2.json"
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
