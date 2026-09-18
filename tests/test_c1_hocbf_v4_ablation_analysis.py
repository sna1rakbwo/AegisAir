from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from marllib.analyze_c1_hocbf_v4_ablation import _rows


class C1HocbfV4AblationAnalysisTest(unittest.TestCase):
    def test_appledouble_summary_is_not_treated_as_experiment_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition = root / "abl01_AEGIS_HOCBF_V4"
            condition.mkdir()
            (condition / "summary.json").write_text(
                json.dumps({"trials": [{"trial_id": "abl01"}]}), encoding="utf-8"
            )
            (condition / "._summary.json").write_bytes(b"not json")
            self.assertEqual(_rows(root), [{"trial_id": "abl01", "condition_dir": condition.name}])


if __name__ == "__main__":
    unittest.main()
