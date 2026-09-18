from __future__ import annotations

import math
import unittest

from px4_adapter.stereo_ego import (
    ObserverPose,
    StereoMeasurement,
    camera_to_world_flu,
    world_position_from_measurement,
)
from swarm.stereo import calibration_from_horizontal_fov


class StereoEgoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = calibration_from_horizontal_fov(640, 360, 70.0, 0.12)

    def test_forward_measurement_maps_to_world_forward(self) -> None:
        self.assertEqual(camera_to_world_flu((0.0, 0.0, 3.0), 0.0), (3.0, 0.0, 0.0))

    def test_yaw_90_maps_forward_to_left(self) -> None:
        x, y, z = camera_to_world_flu((0.0, 0.0, 3.0), math.pi / 2.0)
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 3.0, places=9)
        self.assertAlmostEqual(z, 0.0, places=9)

    def test_world_position_adds_observer_pose(self) -> None:
        measurement = StereoMeasurement(
            target_drone=3,
            u_px=self.calibration.cx_px,
            v_px=self.calibration.cy_px,
            disparity_px=self.calibration.fx_px * self.calibration.baseline_m / 3.0,
            timestamp_ms=1000,
        )
        estimate = world_position_from_measurement(
            measurement,
            self.calibration,
            ObserverPose((1.0, 2.0, 3.0), 0.0),
        )
        self.assertEqual(estimate.target_drone, 3)
        self.assertAlmostEqual(estimate.depth_m, 3.0, places=9)
        self.assertAlmostEqual(estimate.position_flu[0], 4.0, places=9)
        self.assertAlmostEqual(estimate.position_flu[1], 2.0, places=9)
        self.assertAlmostEqual(estimate.position_flu[2], 3.0, places=9)


if __name__ == "__main__":
    unittest.main()
