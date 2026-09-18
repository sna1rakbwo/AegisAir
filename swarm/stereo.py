"""Small, deterministic stereo-geometry helpers for the perception pipeline.

The functions here deliberately do not publish MQTT messages and do not make
control decisions.  They convert a rectified stereo pair measurement into a
camera-frame position; callers may later transform it to world coordinates and
apply the existing freshness/confidence gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RectifiedStereoCalibration:
    """Intrinsics for the left image plus the physical left-right baseline."""

    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    baseline_m: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.fx_px <= 0 or self.fy_px <= 0 or self.baseline_m <= 0:
            raise ValueError("focal lengths and baseline must be positive")


def calibration_from_horizontal_fov(
    width: int, height: int, horizontal_fov_deg: float, baseline_m: float
) -> RectifiedStereoCalibration:
    """Create an approximate rectified calibration for a known horizontal FOV.

    This is adequate for a smoke test only.  Live use must load fx/fy/cx/cy
    obtained from a chessboard calibration after stereo rectification.
    """
    if not 0 < horizontal_fov_deg < 180:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")
    fx_px = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return RectifiedStereoCalibration(
        width=width,
        height=height,
        fx_px=fx_px,
        fy_px=fx_px,
        cx_px=(width - 1) / 2.0,
        cy_px=(height - 1) / 2.0,
        baseline_m=baseline_m,
    )


def depth_from_disparity(disparity_px: float, calibration: RectifiedStereoCalibration) -> float:
    """Return forward depth in metres from positive left-minus-right disparity."""
    if not math.isfinite(disparity_px) or disparity_px <= 0:
        raise ValueError("disparity_px must be finite and positive")
    return calibration.fx_px * calibration.baseline_m / disparity_px


def camera_point_from_disparity(
    u_px: float, v_px: float, disparity_px: float, calibration: RectifiedStereoCalibration
) -> tuple[float, float, float]:
    """Return (right, down, forward) metres in the rectified left camera frame."""
    depth_m = depth_from_disparity(disparity_px, calibration)
    return (
        (u_px - calibration.cx_px) * depth_m / calibration.fx_px,
        (v_px - calibration.cy_px) * depth_m / calibration.fy_px,
        depth_m,
    )


def disparity_from_depth(depth_m: float, calibration: RectifiedStereoCalibration) -> float:
    """Forward projection helper used only for test and synthetic validation."""
    if not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be finite and positive")
    return calibration.fx_px * calibration.baseline_m / depth_m
