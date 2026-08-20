from __future__ import annotations

import unittest

import numpy as np

from marllib.analyze_c1_minimum import paired_ci


class AnalyzeC1MinimumTest(unittest.TestCase):
    def test_constant_difference_has_exact_bootstrap_interval(self) -> None:
        result = paired_ci(np.array([0.25, 0.25, 0.25]), draws=100, seed=1)
        self.assertEqual(result["mean_difference"], 0.25)
        self.assertEqual(result["ci95_low"], 0.25)
        self.assertEqual(result["ci95_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
