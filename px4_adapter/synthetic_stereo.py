"""Synthetic stereo rendering helpers for Plan B.

These functions generate simplified left/right camera images from FLU world
positions. They are only a simulation stand-in for a Gazebo camera sensor; no
real texture, lighting, or occlusion is rendered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from swarm.stereo import RectifiedStereoCalibration


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SyntheticObject:
    position_flu: Vector3
    color_bgr: tuple[int, int, int]
    radius_m: float = 0.35


def camera_basis(
    cam_forward_xyz: Vector3,
    cam_up_xyz: Vector3,
) -> tuple[Vector3, Vector3, Vector3]:
    def normalize(vector: Vector3) -> Vector3:
        norm = math.sqrt(sum(float(x) ** 2 for x in vector))
        if norm == 0:
            raise ValueError("zero camera basis vector")
        return tuple(float(x) / norm for x in vector)

    forward = normalize(cam_forward_xyz)
    right = normalize(
        (
            forward[1] * cam_up_xyz[2] - forward[2] * cam_up_xyz[1],
            forward[2] * cam_up_xyz[0] - forward[0] * cam_up_xyz[2],
            forward[0] * cam_up_xyz[1] - forward[1] * cam_up_xyz[0],
        )
    )
    up = normalize(
        (
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        )
    )
    return right, up, forward


def world_to_camera(
    position_flu: Vector3,
    cam_pos_flu: Vector3,
    cam_forward: Vector3,
    cam_up: Vector3,
) -> tuple[float, float, float]:
    right, up, forward = camera_basis(cam_forward, cam_up)
    dx = position_flu[0] - cam_pos_flu[0]
    dy = position_flu[1] - cam_pos_flu[1]
    dz = position_flu[2] - cam_pos_flu[2]
    return (
        dx * right[0] + dy * right[1] + dz * right[2],
        dx * up[0] + dy * up[1] + dz * up[2],
        dx * forward[0] + dy * forward[1] + dz * forward[2],
    )


def project_to_pixel(
    position_flu: Vector3,
    cam_pos_flu: Vector3,
    cam_forward: Vector3,
    cam_up: Vector3,
    calibration: RectifiedStereoCalibration,
) -> tuple[float, float, float, float] | None:
    x_c, y_c, z_c = world_to_camera(position_flu, cam_pos_flu, cam_forward, cam_up)
    if z_c <= 0:
        return None
    u = calibration.cx_px + calibration.fx_px * x_c / z_c
    v = calibration.cy_px - calibration.fy_px * y_c / z_c
    return u, v, x_c, z_c


def render_stereo_pair(
    objects: list[SyntheticObject],
    cam_pos_flu: Vector3,
    cam_forward: Vector3,
    cam_up: Vector3,
    calibration: RectifiedStereoCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
    right = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)

    for obj in objects:
        projected = project_to_pixel(obj.position_flu, cam_pos_flu, cam_forward, cam_up, calibration)
        if projected is None:
            continue
        u, v, _x_c, z_c = projected
        disparity_px = calibration.fx_px * calibration.baseline_m / z_c
        radius_px = max(2, int(round(obj.radius_m * calibration.fx_px / z_c)))
        cv2.circle(left, (int(round(u)), int(round(v))), radius_px, obj.color_bgr, -1)
        cv2.circle(right, (int(round(u - disparity_px)), int(round(v))), radius_px, obj.color_bgr, -1)

    return left, right
