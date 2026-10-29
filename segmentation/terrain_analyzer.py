"""
Scheduler around the segmentation methods.

Runs the selected method only every N frames (segmentation is not needed at
camera frame rate), caches the last result, and disables a method that throws
instead of crashing the robot loop.
"""

from __future__ import annotations

from typing import Optional

from segmentation.common import SegmentationResult

METHODS = ("off", "kmeans", "felzenszwalb")


class TerrainAnalyzer:
    def __init__(self, cfg, method: str = "off"):
        self.cfg = cfg
        self._segmenters = {}
        self.method = "off"
        self.last: Optional[SegmentationResult] = None
        self.error = ""
        self._frame_counter = 0
        self.set_method(method)

    def _get(self, method: str):
        if method not in self._segmenters:
            if method == "kmeans":
                from segmentation.kmeans_segmentation import KMeansSegmenter
                self._segmenters[method] = KMeansSegmenter(self.cfg)
            elif method == "felzenszwalb":
                from segmentation.felzenszwalb_segmentation import FelzenszwalbSegmenter
                self._segmenters[method] = FelzenszwalbSegmenter(self.cfg)
        return self._segmenters.get(method)

    def set_method(self, method: str) -> str:
        method = (method or "off").lower()
        if method not in METHODS:
            raise ValueError(f"unknown segmentation method '{method}', choose from {METHODS}")
        self.last, self.error, self._frame_counter = None, "", 0
        if method != "off":
            try:
                self._get(method)
            except Exception as exc:
                self.error = f"{method} unavailable: {exc}"
                print(f"[WARN] {self.error}")
                method = "off"
        self.method = method
        return method

    def cycle(self) -> str:
        nxt = METHODS[(METHODS.index(self.method) + 1) % len(METHODS)]
        return self.set_method(nxt)

    @property
    def interval(self) -> int:
        return self.cfg.FELZ_INTERVAL if self.method == "felzenszwalb" else self.cfg.KMEANS_INTERVAL

    def update(self, frame, exclude_box=None) -> Optional[SegmentationResult]:
        """Returns the latest (possibly cached) result, or None if off."""
        if self.method == "off":
            return None
        if self.last is None or self._frame_counter % self.interval == 0:
            try:
                self.last = self._get(self.method).segment(frame, exclude_box)
            except Exception as exc:
                self.error = f"{self.method} failed: {exc}"
                print(f"[WARN] {self.error} - segmentation disabled")
                self.method, self.last = "off", None
        self._frame_counter += 1
        return self.last
