"""
Horizontal position of the target relative to the robot's optical axis.

    cx            = (x1 + x2) / 2
    error_x       = cx - frame_width / 2                  (pixels, + = right)
    error_norm    = error_x / (frame_width / 2)           (range -1 .. +1)
    bearing (deg) = atan(error_norm * tan(HFOV / 2))      (approximate angle)

Classification with hysteresis (T = HORIZONTAL_THRESHOLD, H = HORIZONTAL_HYSTERESIS):

    from CENTER: error < -T -> LEFT,  error > +T -> RIGHT
    from LEFT  : stays LEFT  until error > -(T - H)
    from RIGHT : stays RIGHT until error <  (T - H)

Closed loop (navigation/heading.py): the error is measured from where the robot
currently FACES (`aim_deg`), not from the frame centre. With aim_deg = 0 both are
the same. The camera measurement is smoothed first, then the (exact) aim is
subtracted, so smoothing adds no lag to the robot's own turning.

Note on image orientation: frames are NOT mirrored. The camera is the robot's
eye, so a person in the LEFT half of the image is on the robot's left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from utils.smoothing import ExponentialSmoother


class HorizontalPosition(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


@dataclass
class PositionInfo:
    cx: float
    cy: float
    frame_center_x: float
    error_px: float
    error_norm: float          # raw
    smoothed_error_norm: float
    position: HorizontalPosition
    bearing_deg: float           # relative to where the robot faces
    cam_bearing_raw_deg: float = 0.0   # unsmoothed, relative to the camera axis
    aim_x: Optional[float] = None      # image column the robot faces


class PositionEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.smoother = ExponentialSmoother(cfg.SMOOTHING_FACTOR)
        self.position = HorizontalPosition.UNKNOWN

    def reset(self) -> None:
        self.smoother.reset()
        self.position = HorizontalPosition.UNKNOWN

    def classify(self, e: float) -> HorizontalPosition:
        T, H = self.cfg.HORIZONTAL_THRESHOLD, self.cfg.HORIZONTAL_HYSTERESIS
        p = self.position
        if p == HorizontalPosition.LEFT:
            if e > T:
                return HorizontalPosition.RIGHT
            return HorizontalPosition.LEFT if e < -(T - H) else HorizontalPosition.CENTER
        if p == HorizontalPosition.RIGHT:
            if e < -T:
                return HorizontalPosition.LEFT
            return HorizontalPosition.RIGHT if e > (T - H) else HorizontalPosition.CENTER
        # CENTER or UNKNOWN
        if e < -T:
            return HorizontalPosition.LEFT
        if e > T:
            return HorizontalPosition.RIGHT
        return HorizontalPosition.CENTER

    def update(self, box: Optional[Sequence[float]], frame_width: int,
               aim_deg: float = 0.0) -> Optional[PositionInfo]:
        if box is None:
            self.reset()
            return None
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        fcx = frame_width / 2.0
        tan_half = math.tan(math.radians(self.cfg.CAMERA_HFOV_DEG / 2.0))
        cam_norm = (cx - fcx) / fcx
        smoothed_cam = float(self.smoother.update(cam_norm))
        raw_cam_bearing = math.degrees(math.atan(cam_norm * tan_half))
        cam_bearing = math.degrees(math.atan(smoothed_cam * tan_half))

        def to_norm(bearing: float) -> float:   # direction -> normalised image error
            return math.tan(math.radians(max(-89.0, min(89.0, bearing)))) / tan_half

        bearing = cam_bearing - aim_deg
        aim_x = fcx + fcx * to_norm(aim_deg)
        err_norm = to_norm(raw_cam_bearing - aim_deg)
        smoothed = to_norm(bearing)
        self.position = self.classify(smoothed)
        return PositionInfo(cx, cy, fcx, cx - aim_x, err_norm, smoothed, self.position, bearing,
                            raw_cam_bearing, aim_x)
