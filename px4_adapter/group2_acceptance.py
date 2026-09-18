#!/usr/bin/env python3
"""Deterministic group-2 acceptance checks using Plan-B synthetic frames."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from px4_adapter.synthetic_stereo import SyntheticObject, render_stereo_pair
from scripts.sim_cam_perception import (
    calibration_from_meta,
    median_disparity_in_blob,
    stereo_disparity,
    world_from_stereo,
)
from swarm.perception import camera_intrinsics, color_hsv_ranges, detect_color_blob
from swarm.stereo import RectifiedStereoCalibration, calibration_from_horizontal_fov


def vertical_fov(hfov_deg: float, width: int, height: int) -> float:
    return 2.0 * math.degrees(
        math.atan(math.tan(math.radians(hfov_deg) / 2.0) * height / width)
    )


def meta_calibration(width: int, height: int, hfov_deg: float, baseline_m: float) -> RectifiedStereoCalibration:
    intrinsics = camera_intrinsics(width, height, vertical_fov(hfov_deg, width, height))
    return RectifiedStereoCalibration(
        width=width,
        height=height,
        fx_px=intrinsics.fx_px,
        fy_px=intrinsics.fy_px,
        cx_px=intrinsics.cx_px,
        cy_px=intrinsics.cy_px,
        baseline_m=baseline_m,
    )


def reconstruct(
    position: tuple[float, float, float],
    color: str,
    cam_pos: tuple[float, float, float],
    cam_forward: tuple[float, float, float],
    cam_up: tuple[float, float, float],
    render_cal: RectifiedStereoCalibration,
    meta_cal: RectifiedStereoCalibration,
) -> tuple[float, float, float] | None:
    color_bgr = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
    }[color]
    left, right = render_stereo_pair(
        [SyntheticObject(position, color_bgr, radius_m=0.35)],
        cam_pos,
        cam_forward,
        cam_up,
        render_cal,
    )
    blob = detect_color_blob(left, color_hsv_ranges(color))
    if blob is None:
        return None
    disparity = median_disparity_in_blob(stereo_disparity(left, right), blob)
    if disparity is None:
        return None
    return world_from_stereo(
        blob.centroid[0],
        blob.centroid[1],
        disparity,
        meta_cal,
        list(cam_pos),
        list(cam_forward),
        list(cam_up),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    width, height = 640, 360
    hfov_deg = 70.0
    baseline_m = 0.12
    render_cal = calibration_from_horizontal_fov(width, height, hfov_deg, baseline_m)
    meta_cal = meta_calibration(width, height, hfov_deg, baseline_m)
    cam_pos = (0.0, 0.0, 2.0)
    cam_forward = (1.0, 0.0, 0.0)
    cam_up = (0.0, 0.0, 1.0)

    ground_targets = [(5.0, 1.0, 0.0), (6.0, -1.0, 0.0), (7.0, 2.0, 0.0)]
    ground_errors: list[float] = []
    for target in ground_targets:
        estimate = reconstruct(target, "red", cam_pos, cam_forward, cam_up, render_cal, meta_cal)
        if estimate is None:
            ground_errors.append(float("inf"))
        else:
            ground_errors.append(math.hypot(estimate[0] - target[0], estimate[1] - target[1]))

    moving_errors: list[float] = []
    valid_frames = 0
    max_gap = 0
    current_gap = 0
    total_frames = 20
    for index in range(total_frames):
        x = 1.0 + index * 0.2
        target = (x, 0.0, 1.0)
        estimate = reconstruct(target, "green", (0.0, 0.0, 1.0), cam_forward, cam_up, render_cal, meta_cal)
        if estimate is None:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0
            valid_frames += 1
            moving_errors.append(
                math.dist(estimate, target)
            )

    report = {
        "interface_contract": {
            "frame_topics": [
                "swarm/cam/drone/{id}/left/frame",
                "swarm/cam/drone/{id}/right/frame",
                "swarm/cam/drone/{id}/meta",
            ],
            "output_topic": "swarm/drone_seen/{id}/position",
            "source": "plan_b_synthetic_frames",
            "note": "Unity removed from scope; Plan-B synthetic stereo is the authoritative simulation camera source",
        },
        "ground_target": {
            "errors_m": [round(value, 4) for value in ground_errors],
            "max_plane_error_m": round(max(ground_errors), 4),
            "pass": max(ground_errors) < 0.2,
        },
        "moving_target": {
            "frames": total_frames,
            "valid_frames": valid_frames,
            "max_error_m": round(max(moving_errors), 4) if moving_errors else None,
            "max_gap_frames": max_gap,
            "pass": (
                bool(moving_errors)
                and max(moving_errors) < 0.5
                and max_gap <= 2
                and valid_frames / total_frames >= 0.8
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ground_target"]["pass"] and report["moving_target"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
