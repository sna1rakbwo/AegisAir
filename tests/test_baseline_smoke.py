from __future__ import annotations

import unittest

from marllib.run_baseline_smoke import CONDITIONS, make_controller, summarize


class BaselineSmokeTest(unittest.TestCase):
    def test_all_frozen_conditions_have_a_controller(self) -> None:
        self.assertEqual(CONDITIONS, ("S0", "S1", "S2", "S3", "S6"))
        for condition in (*CONDITIONS, "S4", "S5", "E0", "E1", "E2"):
            self.assertTrue(callable(make_controller(condition, qp_max_iters=10).filter))

    def test_summary_keeps_conditions_separate(self) -> None:
        rows = [
            {"scenario": "head_on", "condition": "S0", "collision": True, "completed": False, "min_rho": -0.1, "cbf_events": 0, "rejected_commands": 0},
            {"scenario": "head_on", "condition": "S1", "collision": False, "completed": True, "min_rho": 0.2, "cbf_events": 3, "rejected_commands": 0},
        ]
        result = summarize(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual({row["condition"] for row in result}, {"S0", "S1"})


if __name__ == "__main__":
    unittest.main()
