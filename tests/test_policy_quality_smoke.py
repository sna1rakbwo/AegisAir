from __future__ import annotations

import unittest

from marllib.run_policy_quality_smoke import QUALITY_SCALES, SAFETY_CONDITIONS, ScaledPilot, summarize


class PolicyQualitySmokeTest(unittest.TestCase):
    def test_scaled_pilot_scales_each_action(self) -> None:
        class Pilot:
            def actions(self, **kwargs):
                return {0: [1.0, -0.5]}

        action = ScaledPilot(Pilot(), 0.2).actions()[0]
        self.assertEqual(action.tolist(), [0.2, -0.1])

    def test_protocol_conditions_are_frozen(self) -> None:
        self.assertEqual(QUALITY_SCALES, (0.2, 0.5, 0.8, 1.0))
        self.assertEqual(SAFETY_CONDITIONS, ("S0", "S6"))

    def test_summary_groups_training_seed_scale_and_condition(self) -> None:
        rows = [{"training_seed": 1, "quality_scale": 1.0, "condition": "S0", "collision": True, "completed": False, "min_rho": -1.0, "cbf_events": 0}]
        result = summarize(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["collision_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
