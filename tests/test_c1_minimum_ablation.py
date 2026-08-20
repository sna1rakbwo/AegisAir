from __future__ import annotations

import unittest

from marllib.run_c1_minimum_ablation import ABLATIONS, SCENARIOS, params_for, scenario_spec


class C1MinimumAblationTest(unittest.TestCase):
    def test_all_required_ablations_are_frozen(self) -> None:
        self.assertEqual(ABLATIONS, ("fixed_distance", "full_envelope", "no_perception_margin", "no_aoi_margin"))

    def test_ablation_changes_only_its_expected_margin_terms(self) -> None:
        fixed = params_for("fixed_distance")
        self.assertEqual(fixed.beta, 0.0)
        self.assertEqual(fixed.v_max, 0.0)
        self.assertEqual(params_for("no_perception_margin").beta, 0.0)
        self.assertEqual(params_for("no_aoi_margin").v_max, 0.0)

    def test_all_scenarios_are_four_uav(self) -> None:
        self.assertEqual(set(SCENARIOS), {"randomized_start_goal", "dense_intersection", "perception_dropout", "telemetry_delay"})
        for name in SCENARIOS:
            self.assertEqual(scenario_spec(name)["scenario"].num_agents, 4)


if __name__ == "__main__":
    unittest.main()
