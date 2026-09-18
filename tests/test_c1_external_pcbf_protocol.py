from __future__ import annotations

import json
import unittest
from pathlib import Path


class ExternalPCBFProtocolTest(unittest.TestCase):
    def test_manifest_is_balanced_and_uses_fresh_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/c1_external_pcbf_sealed_v1.json").read_text())
        methods = set(manifest["methods"])
        self.assertEqual(len(methods), 5)
        self.assertEqual(len(manifest["trials"]), 20)
        self.assertEqual({trial["seed"] for trial in manifest["trials"]}, set(range(12901, 12921)))
        self.assertTrue(all(set(trial["condition_order"]) == methods for trial in manifest["trials"]))
        self.assertEqual(
            {method: sum(trial["condition_order"][0] == method for trial in manifest["trials"]) for method in methods},
            {method: 4 for method in methods},
        )


if __name__ == "__main__":
    unittest.main()
