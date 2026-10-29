"""
Monocular distance PROXY from bounding-box height.

A single webcam has no depth sensor, so we cannot measure metres directly.
We use the fact that under a pinhole camera model the image height of a person
is inversely proportional to their distance:

        h_px  ~=  f_px * H_person / Z      =>   bigger box  ==  closer person

Primary output: a RELATIVE zone FAR / IDEAL / CLOSE from
    ratio = box_height / frame_height
with four thresholds (the gaps act as hysteresis, see config.py).

Secondary output (display only, clearly labelled "approx."): a rough metric
estimate assuming the person is ASSUMED_PERSON_HEIGHT_M tall and the camera has
CAMERA_HFOV_DEG field of view. It is NOT calibrated and is wrong for sitting
people or cropped boxes; when the box touches the top/bottom edge we only report
an upper bound ("< X m").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from utils.smoothing import ExponentialSmoother


class DistanceZone(str, Enum):
    FAR = "FAR"
    IDEAL = "IDEAL"
    CLOSE = "CLOSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class DistanceInfo:
    height_px: float
    height_ratio: float          # raw
    smoothed_ratio: float
    zone: DistanceZone
    approx_m: Optional[float]    # rough, uncalibrated
    truncated: bool              # box touches top/bottom edge -> person partly out of view

    def approx_text(self) -> str:
        if self.approx_m is None:
            return "n/a"
        return ("< " if self.truncated else "") + f"{self.approx_m:.1f} m"


class DistanceEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.smoother = ExponentialSmoother(cfg.SMOOTHING_FACTOR)
        self.zone = DistanceZone.UNKNOWN

    def reset(self) -> None:
        self.smoother.reset()
        self.zone = DistanceZone.UNKNOWN

    def calibrate(self, current_ratio: float) -> None:
        """Make the CURRENT distance the ideal following distance (key 'c')."""
        b = self.cfg.CALIBRATION_BAND
        r = min(max(current_ratio, b + 0.06), 1.0 - b - 0.01)
        self.cfg.IDEAL_MIN_HEIGHT = r - b / 2
        self.cfg.IDEAL_MAX_HEIGHT = r + b / 2
        self.cfg.TOO_FAR_HEIGHT = r - b
        self.cfg.TOO_CLOSE_HEIGHT = min(r + b, 0.995)
        self.zone = DistanceZone.IDEAL

    def classify(self, r: float) -> DistanceZone:
        c, z = self.cfg, self.zone
        if z == DistanceZone.UNKNOWN:
            if r < c.IDEAL_MIN_HEIGHT:
                return DistanceZone.FAR
            if r > c.TOO_CLOSE_HEIGHT:
                return DistanceZone.CLOSE
            return DistanceZone.IDEAL
        if r < c.TOO_FAR_HEIGHT:
            return DistanceZone.FAR
        if r > c.TOO_CLOSE_HEIGHT:
            return DistanceZone.CLOSE
        if c.IDEAL_MIN_HEIGHT <= r <= c.IDEAL_MAX_HEIGHT:
            return DistanceZone.IDEAL
        return z  # inside a hysteresis gap: keep previous zone

    def approx_metric(self, height_px: float, frame_w: int, frame_h: int) -> Optional[float]:
        if height_px <= 1:
            return None
        half_h = math.radians(self.cfg.CAMERA_HFOV_DEG / 2.0)
        # vertical FOV from horizontal FOV and aspect ratio (square pixels)
        tan_half_v = math.tan(half_h) * frame_h / frame_w
        f_px = (frame_h / 2.0) / tan_half_v
        return self.cfg.ASSUMED_PERSON_HEIGHT_M * f_px / height_px

    def update(self, box: Optional[Sequence[float]], frame_w: int, frame_h: int) -> Optional[DistanceInfo]:
        if box is None:
            self.reset()
            return None
        x1, y1, x2, y2 = box
        hpx = max(0.0, y2 - y1)
        ratio = hpx / float(frame_h)
        smoothed = float(self.smoother.update(ratio))
        self.zone = self.classify(smoothed)
        edge = 0.02 * frame_h   # box within 2 % of the top/bottom edge -> person is cut off
        truncated = y1 <= edge or y2 >= frame_h - 1 - edge
        approx = self.approx_metric(smoothed * frame_h, frame_w, frame_h) if self.cfg.SHOW_APPROX_METRIC else None
        return DistanceInfo(hpx, ratio, smoothed, self.zone, approx, truncated)
