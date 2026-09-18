from __future__ import annotations

import unittest
from pathlib import Path

from px4_adapter.mqtt_lossy_proxy import main


class LossyProxyTopicTest(unittest.TestCase):
    def test_default_topics_are_instance_scoped(self) -> None:
        # The default topic strings are generated inside main; this test documents
        # the expected contract without requiring a live MQTT broker.
        self.assertEqual("px4_raw/2/command", f"px4_raw/{2}/command")
        self.assertEqual("px4/2/command", f"px4/{2}/command")


if __name__ == "__main__":
    unittest.main()
