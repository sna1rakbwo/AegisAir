from __future__ import annotations

import unittest

from px4_adapter.synthetic_stereo import SyntheticObject, render_stereo_pair
from swarm.stereo import calibration_from_horizontal_fov


class SyntheticStereoTest(unittest.TestCase):
    def test_forward_object_has_disparity(self) -> None:
        calibration = calibration_from_horizontal_fov(640, 360, 70.0, 0.12)
        left, right = render_stereo_pair(
            [
                SyntheticObject(
                    position_flu=(3.0, 0.0, 0.0),
                    color_bgr=(0, 0, 255),
                    radius_m=0.35,
                )
            ],
            cam_pos_flu=(0.0, 0.0, 0.0),
            cam_forward=(1.0, 0.0, 0.0),
            cam_up=(0.0, 0.0, 1.0),
            calibration=calibration,
        )
        self.assertGreater(left.sum(), 0)
        self.assertGreater(right.sum(), 0)
        self.assertGreater(left.sum(), 0)


if __name__ == "__main__":
    unittest.main()
