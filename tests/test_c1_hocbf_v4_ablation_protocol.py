"""冻结四条件 reserve/prediction 消融协议的结构检查。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "c1_hocbf_v4_ablation_sealed_v1.json"


class C1HocbfV4AblationProtocolTest(unittest.TestCase):
    def test_manifest_uses_new_paired_seeds_and_balanced_orders(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase"], "sealed_paired_ablation")
        self.assertEqual([row["seed"] for row in manifest["trials"]], list(range(9601, 9621)))
        expected = set(manifest["methods"])
        self.assertEqual(len(manifest["trials"]), 20)
        for trial in manifest["trials"]:
            self.assertEqual(set(trial["condition_order"]), expected)
            self.assertEqual(len(trial["condition_order"]), 4)
        self.assertEqual(
            {method: sum(t["condition_order"][0] == method for t in manifest["trials"])
             for method in expected},
            {method: 5 for method in expected},
        )

    def test_only_prediction_and_reserve_switches_vary_across_v4_conditions(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        reactive = manifest["methods"]["AEGIS_HOCBF_V4_REACTIVE"]
        reserve_only = manifest["methods"]["AEGIS_HOCBF_V4_RESERVE_ONLY"]
        full = manifest["methods"]["AEGIS_HOCBF_V4"]
        self.assertFalse(reactive["predictive_recovery"])
        self.assertNotIn("recovery_reserve_threshold", reactive)
        self.assertFalse(reserve_only["predictive_recovery"])
        self.assertEqual(reserve_only["recovery_reserve_threshold"], 1.0)
        self.assertTrue(full["predictive_recovery"])
        self.assertEqual(full["recovery_reserve_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
