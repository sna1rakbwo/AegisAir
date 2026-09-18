from __future__ import annotations

import json
from pathlib import Path
import unittest

from marllib.analyze_c_recoverability_admission_calibration import analyze
from marllib.run_c_recoverability_admission_gazebo import (
    CONDITIONS,
    PROTOCOL_IDS,
    _admission_config,
)


ROOT = Path(__file__).resolve().parents[1]


class RecoverabilityAdmissionProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(
            (ROOT / "configs/c_recoverability_admission_calibration_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_freezes_three_comparators_and_both_outcomes(self) -> None:
        self.assertIn(self.manifest["protocol_id"], PROTOCOL_IDS)
        self.assertEqual(set(self.manifest["conditions"]), CONDITIONS)
        expected = {
            geometry["expected_admission"]
            for geometry in self.manifest["geometries"].values()
        }
        self.assertEqual(expected, {"admit", "reject"})
        for trial in self.manifest["trials"]:
            self.assertEqual(set(trial["condition_order"]), CONDITIONS)

    def test_admission_parameters_are_frozen(self) -> None:
        config = _admission_config(self.manifest["recoverability_admission"])
        self.assertEqual(config.clearance_m, 2.4)
        self.assertEqual(config.braking_accel_mps2, 2.0)
        self.assertEqual(config.ring_samples, 32)
        self.assertEqual(config.max_route_length_m, 18.0)

    def test_empty_analysis_is_no_go_not_complete(self) -> None:
        result = analyze(self.manifest, [])
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["complete"])
        self.assertFalse(result["anti_vacuity"])


if __name__ == "__main__":
    unittest.main()
