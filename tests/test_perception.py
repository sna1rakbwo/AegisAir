from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from swarm.perception import (
    camera_intrinsics,
    color_hsv_ranges,
    detect_color_blob,
    pixel_to_ray,
    ray_to_ground,
    ray_to_world,
    stereo_camera_point,
    stereo_depth_from_disparity,
)
from swarm.stereo import calibration_from_horizontal_fov, disparity_from_depth


class PerceptionGeometryTest(unittest.TestCase):
    def test_center_pixel_points_forward(self) -> None:
        intrinsics = camera_intrinsics(640, 360, 70.0)
        ray = pixel_to_ray(intrinsics.cx_px, intrinsics.cy_px, intrinsics)
        self.assertAlmostEqual(ray[0], 0.0, places=9)
        self.assertAlmostEqual(ray[1], 0.0, places=9)
        self.assertAlmostEqual(ray[2], 1.0, places=9)

    def test_ray_to_world_is_orthonormal_for_forward_and_up(self) -> None:
        forward = (0.0, 0.0, -1.0)
        up = (0.0, 1.0, 0.0)
        world = ray_to_world(forward, up, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(world[0], 0.0, places=9)
        self.assertAlmostEqual(world[1], 0.0, places=9)
        self.assertAlmostEqual(world[2], -1.0, places=9)

    def test_ground_intersection_below_camera(self) -> None:
        cam_pos = (0.0, 0.0, 2.0)
        forward = (0.0, 0.0, -1.0)
        up = (0.0, 1.0, 0.0)
        ray = (0.0, 0.0, 1.0)
        ground = ray_to_ground(cam_pos, forward, up, ray, ground_z=0.0)
        self.assertIsNotNone(ground)
        self.assertAlmostEqual(ground[0], 0.0, places=9)
        self.assertAlmostEqual(ground[1], 0.0, places=9)
        self.assertAlmostEqual(ground[2], 0.0, places=9)

    def test_horizontal_ray_has_no_ground(self) -> None:
        cam_pos = (0.0, 0.0, 2.0)
        forward = (1.0, 0.0, 0.0)
        up = (0.0, 0.0, 1.0)
        ray = (0.0, 0.0, 1.0)
        self.assertIsNone(ray_to_ground(cam_pos, forward, up, ray, ground_z=0.0))

    def test_stereo_depth_round_trip(self) -> None:
        calibration = calibration_from_horizontal_fov(640, 360, 70.0, 0.12)
        disparity = disparity_from_depth(3.0, calibration)
        self.assertAlmostEqual(
            stereo_depth_from_disparity(disparity, calibration), 3.0, places=9
        )

    def test_stereo_camera_point_uses_up_axis(self) -> None:
        calibration = calibration_from_horizontal_fov(640, 360, 70.0, 0.12)
        disparity = disparity_from_depth(3.0, calibration)
        point = stereo_camera_point(
            calibration.cx_px, calibration.cy_px, disparity, calibration
        )
        self.assertAlmostEqual(point[0], 0.0, places=9)
        self.assertAlmostEqual(point[1], 0.0, places=9)
        self.assertAlmostEqual(point[2], 3.0, places=9)

    def test_detect_red_blob(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (60, 60), (0, 0, 255), -1)
        blob = detect_color_blob(frame, color_hsv_ranges("red"))
        self.assertIsNotNone(blob)
        if blob is not None:
            self.assertAlmostEqual(blob.centroid[0], 50.0, delta=2.0)
            self.assertAlmostEqual(blob.centroid[1], 50.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
