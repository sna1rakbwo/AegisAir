"""Stereo measurement to ego-position geometry.

This module converts a rectified stereo measurement into a world FLU position
using the observing drone's own pose and yaw.  It is independent of MQTT and
ROS 2 so the geometry can be unit-tested without a live runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from swarm.stereo import RectifiedStereoCalibration, camera_point_from_disparity, depth_from_disparity


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class StereoMeasurement:
    target_drone: int
    u_px: float
    v_px: float
    disparity_px: float
    timestamp_ms: int
    uncertainty_m: float | None = None


@dataclass(frozen=True)
class ObserverPose:
    position_flu: Vector3
    yaw_rad: float


@dataclass(frozen=True)
class EgoEstimate:
    target_drone: int
    position_flu: Vector3
    depth_m: float
    uncertainty_m: float | None
    timestamp_ms: int


def camera_to_world_flu(camera_point: Vector3, yaw_rad: float) -> Vector3:
    """Transform rectified-left camera RDF to world FLU.

    Camera frame from :func:`swarm.stereo.camera_point_from_disparity` is
    ``(right, down, forward)``.  World FLU is ``(forward, left, up)``.
    ``yaw_rad`` rotates the forward direction from world +x toward +y.
    """
    right_m, down_m, forward_m = camera_point
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return (
        forward_m * cos_yaw - right_m * sin_yaw,
        forward_m * sin_yaw + right_m * cos_yaw,
        -down_m,
    )


def world_position_from_measurement(
    measurement: StereoMeasurement,
    calibration: RectifiedStereoCalibration,
    observer: ObserverPose,
) -> EgoEstimate:
    """Return an absolute FLU ego position from one stereo measurement."""
    camera_point = camera_point_from_disparity(
        measurement.u_px,
        measurement.v_px,
        measurement.disparity_px,
        calibration,
    )
    relative_flu = camera_to_world_flu(camera_point, observer.yaw_rad)
    position_flu = (
        observer.position_flu[0] + relative_flu[0],
        observer.position_flu[1] + relative_flu[1],
        observer.position_flu[2] + relative_flu[2],
    )
    depth_m = depth_from_disparity(measurement.disparity_px, calibration)
    return EgoEstimate(
        target_drone=measurement.target_drone,
        position_flu=position_flu,
        depth_m=depth_m,
        uncertainty_m=measurement.uncertainty_m,
        timestamp_ms=measurement.timestamp_ms,
    )
