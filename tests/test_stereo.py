import unittest

from swarm.stereo import calibration_from_horizontal_fov, camera_point_from_disparity, depth_from_disparity, disparity_from_depth


class StereoGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = calibration_from_horizontal_fov(640, 360, 70.0, 0.12)

    def test_depth_round_trip(self) -> None:
        depth_m = 4.2
        disparity_px = disparity_from_depth(depth_m, self.calibration)
        self.assertAlmostEqual(depth_from_disparity(disparity_px, self.calibration), depth_m, places=12)

    def test_principal_point_has_zero_lateral_offset(self) -> None:
        disparity_px = disparity_from_depth(3.0, self.calibration)
        x_m, y_m, z_m = camera_point_from_disparity(
            self.calibration.cx_px, self.calibration.cy_px, disparity_px, self.calibration
        )
        self.assertAlmostEqual(x_m, 0.0, places=12)
        self.assertAlmostEqual(y_m, 0.0, places=12)
        self.assertAlmostEqual(z_m, 3.0, places=12)

    def test_non_positive_disparity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            depth_from_disparity(0.0, self.calibration)


if __name__ == "__main__":
    unittest.main()
