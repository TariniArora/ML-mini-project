"""
Camera (laptop) ego-motion from background optical flow.

If someone rotates the laptop, the WHOLE background slides sideways in the
image. We track ~200 corner features on the background (the target person and
other detected people are masked out) with pyramidal Lucas-Kanade flow and take
the median horizontal shift:

    dx (px/frame)  ->  yaw = atan(-dx / f_px)      (+ = camera turned right)
    f_px = (frame_width / 2) / tan(HFOV / 2)

The median ignores a minority of moving things (a waving arm, a passer-by). If
the flow vectors disagree (most of the scene is moving on its own) or too few
background features are visible, the estimate is rejected and reported as 0.
Tiny shifts below EGO_DEADBAND_PX are treated as sensor noise so a static
laptop does not slowly drift.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

REDETECT_EVERY = 15   # frames between fresh feature detections
BOX_MARGIN = 0.15     # mask people a bit larger than their box (arms, motion blur)


@dataclass
class EgoMotion:
    dx_px: float = 0.0     # median background shift, full-resolution pixels
    dy_px: float = 0.0
    yaw_deg: float = 0.0   # camera rotation this frame, + = turned right
    points: int = 0
    ok: bool = False
    latency_ms: float = 0.0


class CameraMotionEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.prev: Optional[np.ndarray] = None
        self.pts: Optional[np.ndarray] = None
        self.frame_count = 0

    def _mask(self, shape, boxes, s: float) -> np.ndarray:
        mask = np.full(shape, 255, np.uint8)
        for b in boxes:
            x1, y1, x2, y2 = (v * s for v in b)
            mx, my = (x2 - x1) * BOX_MARGIN, (y2 - y1) * BOX_MARGIN
            mask[max(0, int(y1 - my)):max(0, int(y2 + my)),
                 max(0, int(x1 - mx)):max(0, int(x2 + mx))] = 0
        return mask

    def update(self, frame: np.ndarray, exclude: Sequence[Sequence[float]] = ()) -> EgoMotion:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        s = min(1.0, self.cfg.EGO_WIDTH / float(w))
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        mask = self._mask(gray.shape, exclude, s)
        res = EgoMotion()
        min_pts = self.cfg.EGO_MIN_POINTS

        if self.prev is not None and self.prev.shape == gray.shape and \
                self.pts is not None and len(self.pts) >= min_pts:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None,
                                                      winSize=(21, 21), maxLevel=3)
            good = status.reshape(-1) == 1
            p0, p1 = self.pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
            # drop features that ended up on a person (they move by themselves)
            xi = np.clip(p1[:, 0].astype(int), 0, gray.shape[1] - 1)
            yi = np.clip(p1[:, 1].astype(int), 0, gray.shape[0] - 1)
            keep = mask[yi, xi] > 0
            p0, p1 = p0[keep], p1[keep]
            res.points = len(p0)
            if len(p0) >= min_pts:
                d = p1 - p0
                dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
                agree = np.mean(np.abs(d[:, 0] - dx) < max(1.0, 0.3 * abs(dx)))
                if agree >= 0.5:  # most of the background moved together -> it was the camera
                    res.ok = True
                    res.dx_px, res.dy_px = dx / s, dy / s
                    if abs(res.dx_px) >= self.cfg.EGO_DEADBAND_PX:
                        f_px = (w / 2.0) / math.tan(math.radians(self.cfg.CAMERA_HFOV_DEG / 2.0))
                        res.yaw_deg = math.degrees(math.atan(-res.dx_px / f_px))
                    else:
                        res.dx_px = res.dy_px = 0.0
                self.pts = p1.reshape(-1, 1, 2).astype(np.float32)
            else:
                self.pts = None

        self.frame_count += 1
        if self.pts is None or len(self.pts) < 2 * min_pts or self.frame_count % REDETECT_EVERY == 0:
            self.pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                               minDistance=7, mask=mask)
        self.prev = gray
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res
