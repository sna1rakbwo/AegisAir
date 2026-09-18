"""Group 2 pure perception helpers: camera geometry and HSV color detection.

This module has no MQTT/ROS dependency so the geometry can be unit-tested
without a live camera. It follows the project coordinate convention where
``z`` points up and the camera frame is ``x=right, y=up, z=forward``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from swarm.stereo import RectifiedStereoCalibration, camera_point_from_disparity, depth_from_disparity


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class HsvRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class ColorBlob:
    centroid: tuple[float, float]
    bbox_width_px: float
    area_px: float


def camera_intrinsics(width: int, height: int, fov_deg: float) -> CameraIntrinsics:
    """Return fx/fy/cx/cy for Unity's vertical field-of-view convention."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0 < fov_deg < 180:
        raise ValueError("fov_deg must be in (0, 180)")
    fy = height / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return CameraIntrinsics(
        fx_px=fy,
        fy_px=fy,
        cx_px=(width - 1) / 2.0,
        cy_px=(height - 1) / 2.0,
    )


def pixel_to_ray(u_px: float, v_px: float, intrinsics: CameraIntrinsics) -> Vector3:
    dx = (u_px - intrinsics.cx_px) / intrinsics.fx_px
    dy = (intrinsics.cy_px - v_px) / intrinsics.fy_px
    norm = math.sqrt(dx * dx + dy * dy + 1.0)
    return (dx / norm, dy / norm, 1.0 / norm)


def _normalize(value: Vector3) -> Vector3:
    norm = math.sqrt(sum(component * component for component in value))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def ray_to_world(
    cam_forward_xyz: Vector3,
    cam_up_xyz: Vector3,
    ray: Vector3,
) -> Vector3:
    forward = _normalize(cam_forward_xyz)
    up_raw = (
        cam_up_xyz[0] - _dot(cam_up_xyz, forward) * forward[0],
        cam_up_xyz[1] - _dot(cam_up_xyz, forward) * forward[1],
        cam_up_xyz[2] - _dot(cam_up_xyz, forward) * forward[2],
    )
    up = _normalize(up_raw)
    right = _cross(forward, up)
    return _normalize(
        (
            ray[0] * right[0] + ray[1] * up[0] + ray[2] * forward[0],
            ray[0] * right[1] + ray[1] * up[1] + ray[2] * forward[1],
            ray[0] * right[2] + ray[1] * up[2] + ray[2] * forward[2],
        )
    )


def ray_to_ground(
    cam_pos_xyz: Vector3,
    cam_forward_xyz: Vector3,
    cam_up_xyz: Vector3,
    ray: Vector3,
    ground_z: float = 0.0,
) -> Vector3 | None:
    world_ray = ray_to_world(cam_forward_xyz, cam_up_xyz, ray)
    if abs(world_ray[2]) < 1e-12:
        return None
    t = (ground_z - cam_pos_xyz[2]) / world_ray[2]
    if t < 0:
        return None
    return (
        cam_pos_xyz[0] + t * world_ray[0],
        cam_pos_xyz[1] + t * world_ray[1],
        cam_pos_xyz[2] + t * world_ray[2],
    )


def depth_from_apparent_size(
    pixel_width: float,
    real_width_m: float,
    fx_px: float,
) -> float:
    """Legacy monocular apparent-size depth fallback.

    Group 2's primary depth source is stereo disparity, not this function.
    """
    if pixel_width <= 0:
        raise ValueError("pixel_width must be positive")
    if real_width_m <= 0 or fx_px <= 0:
        raise ValueError("real_width_m and fx_px must be positive")
    return fx_px * real_width_m / pixel_width


def stereo_depth_from_disparity(
    disparity_px: float,
    calibration: RectifiedStereoCalibration,
) -> float:
    return depth_from_disparity(disparity_px, calibration)


def stereo_camera_point(
    u_px: float,
    v_px: float,
    disparity_px: float,
    calibration: RectifiedStereoCalibration,
) -> Vector3:
    """Return a Group 2 camera point in (right, up, forward) metres.

    ``swarm.stereo`` returns ``(right, down, forward)``; this converts it to the
    Group 2 camera convention used by ``ray_to_world``.
    """
    right_m, down_m, forward_m = camera_point_from_disparity(
        u_px, v_px, disparity_px, calibration
    )
    return (right_m, -down_m, forward_m)


def color_hsv_ranges(color: str) -> list[HsvRange]:
    color = color.strip().lower()
    if color == "red":
        return [
            HsvRange((0, 100, 80), (10, 255, 255)),
            HsvRange((170, 100, 80), (180, 255, 255)),
        ]
    if color == "blue":
        return [HsvRange((95, 80, 60), (130, 255, 255))]
    if color == "green":
        return [HsvRange((60, 60, 50), (90, 255, 255))]
    if color == "yellow":
        return [HsvRange((15, 80, 80), (40, 255, 255))]
    raise ValueError(f"unknown color: {color}")


def detect_color_blob(
    frame_bgr: np.ndarray,
    hsv_ranges: Iterable[HsvRange],
) -> ColorBlob | None:
    if frame_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(hsv_range.lower), np.array(hsv_range.upper))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area <= 0:
        return None
    x, y, width, height = cv2.boundingRect(largest)
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    centroid = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    return ColorBlob(centroid=centroid, bbox_width_px=float(width), area_px=area)
