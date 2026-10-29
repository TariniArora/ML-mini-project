"""
Single-target tracker: detection -> initialise/refresh -> track -> periodic refresh.

Components
----------
* Kalman filter (tracking/kalman.py)          - smooths the box, predicts through short gaps.
* TargetSelector (detection/target_selector.py) - associates new detections with the track.
* AppearanceModel                               - colour histogram used to re-identify the
                                                  same person after they were lost.
* Optional OpenCV visual tracker (CSRT/KCF/MIL) - used on frames where the detector is
                                                  skipped (DETECT_INTERVAL > 1).

State machine
-------------
    SEARCHING --(person acquired)--> TRACKING
    TRACKING  --(detector misses)--> COASTING   (Kalman prediction, confidence decays)
    COASTING  --(re-detected)------> TRACKING
    COASTING  --(LOST_TIMEOUT_FRAMES)--> LOST   (robot must STOP)
    LOST      --(re-identified / new person)--> TRACKING
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detection.person_detector import Detection
from detection.target_selector import AppearanceModel, TargetSelector
from tracking.kalman import KalmanBoxFilter
from utils.geometry import clip_box, is_valid_box, xyxy_to_xywh, xywh_to_xyxy


class TrackState(str, Enum):
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    COASTING = "COASTING"   # temporarily not detected - predicting
    LOST = "LOST"


@dataclass
class TrackResult:
    state: TrackState
    box: Optional[Tuple[float, float, float, float]]  # smoothed (Kalman) box
    raw_box: Optional[Tuple[float, float, float, float]]  # last measurement
    confidence: float          # current track confidence (decays while coasting)
    frames_missed: int
    source: str                # "detector" | "visual-tracker" | "prediction" | "-"
    event: str = ""            # human-readable note, e.g. "acquired: only one person"
    latency_ms: float = 0.0

    @property
    def has_target(self) -> bool:
        return self.box is not None and self.state in (TrackState.TRACKING, TrackState.COASTING)


def create_visual_tracker(kind: str):
    """Return an OpenCV single-object tracker instance, or None if unavailable.

    CSRT/KCF live in opencv-contrib (and moved to cv2.legacy in some versions);
    MIL ships with plain opencv-python. "auto" only uses CSRT/KCF: we measured MIL
    at ~50 ms/update on a CPU - slower than YOLOv8n itself - so in that case it is
    better to bridge skipped frames with the (almost free) Kalman prediction.
    """
    kind = kind.lower()
    if kind == "none":
        return None
    order = ["csrt", "kcf"] if kind == "auto" else [kind]
    for name in order:
        fname = f"Tracker{name.upper()}_create"
        for mod in (cv2, getattr(cv2, "legacy", None)):
            if mod is not None and hasattr(mod, fname):
                try:
                    return getattr(mod, fname)()
                except Exception:
                    continue
    return None


class TargetTracker:
    def __init__(self, cfg, selector: Optional[TargetSelector] = None):
        self.cfg = cfg
        self.selector = selector or TargetSelector(cfg)
        self.appearance = AppearanceModel(cfg.APPEARANCE_UPDATE_RATE)
        self.kf: Optional[KalmanBoxFilter] = None
        self.visual = None
        self.visual_name = "none"
        self.state = TrackState.SEARCHING
        self.confidence = 0.0
        self.frames_missed = 0
        self.raw_box = None
        self.target_switches = 0      # number of (re)acquisitions - useful statistic
        self._ever_acquired = False

    # ---------------------------------------------------------------- public
    def reset(self, forget_appearance: bool = True) -> None:
        self.kf = None
        self.visual = None
        self.state = TrackState.SEARCHING
        self.confidence = 0.0
        self.frames_missed = 0
        self.raw_box = None
        if forget_appearance:
            self.appearance.reset()

    def shift(self, dx: float, dy: float) -> None:
        """The camera moved: move the track with the background so the next
        prediction lands where the person now appears (tracking/ego_motion.py)."""
        if self.kf is not None:
            self.kf.x[0] += dx
            self.kf.x[1] += dy

    def step(self, frame: np.ndarray, detections: Optional[List[Detection]]) -> TrackResult:
        """Advance the tracker by one frame.

        detections: list from the detector, or None if the detector was skipped
        on this frame (then the visual tracker / Kalman prediction is used).
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        event, source = "", "-"

        if self.kf is not None:
            self.kf.predict()

        if self.state in (TrackState.TRACKING, TrackState.COASTING):
            measured = None
            if detections is not None:
                match = self.selector.match(detections, self.kf.box, frame.shape, frame, self.appearance)
                if match is not None:
                    measured, source = match.as_tuple(), "detector"
                    self.confidence = match.confidence
                    self._refresh_visual(frame, measured)
            else:
                # Detector deliberately skipped on this frame (DETECT_INTERVAL > 1).
                measured = self._run_visual(frame)
                if measured is not None:
                    source = "visual-tracker"
                    self.confidence *= 0.99  # slowly decays until next detector refresh
                elif self.state == TrackState.TRACKING:
                    # No visual tracker: bridge the skipped frame with the Kalman
                    # prediction. This is NOT a miss - the next detector run decides.
                    source = "prediction"
                    self.confidence *= 0.99
                    box = clip_box(self.kf.box, w, h)
                    return TrackResult(self.state, box, self.raw_box, float(self.confidence),
                                       self.frames_missed, source, "",
                                       (time.perf_counter() - t0) * 1000.0)

            if measured is not None and is_valid_box(measured, w, h):
                self.kf.update(measured)
                self.raw_box = measured
                if self.state == TrackState.COASTING:
                    event = "target re-acquired"
                self.state = TrackState.TRACKING
                self.frames_missed = 0
                if source == "detector":
                    self.appearance.update(frame, measured)
            else:
                self.frames_missed += 1
                self.confidence *= self.cfg.COAST_CONFIDENCE_DECAY
                self.kf.damp_velocity(0.7)
                source = "prediction"
                if self.state == TrackState.TRACKING:
                    event = "target not detected - coasting"
                self.state = TrackState.COASTING
                if self.frames_missed > self.cfg.LOST_TIMEOUT_FRAMES or \
                        not is_valid_box(self.kf.box, w, h):
                    self.reset(forget_appearance=False)
                    self.state = TrackState.LOST
                    event = "TARGET LOST - stopping"

        elif detections is not None:  # SEARCHING or LOST -> try to acquire
            det, reason = self.selector.acquire(detections, frame.shape, frame, self.appearance)
            if det is not None:
                box = det.as_tuple()
                self.kf = KalmanBoxFilter(box)
                self.raw_box = box
                self.confidence = det.confidence
                self.state = TrackState.TRACKING
                self.frames_missed = 0
                self.appearance.update(frame, box)
                self._refresh_visual(frame, box)
                self.target_switches += 1
                event = f"acquired: {reason}"
                source = "detector"
                self._ever_acquired = True

        box = None
        if self.kf is not None and self.state in (TrackState.TRACKING, TrackState.COASTING):
            box = clip_box(self.kf.box, w, h)
            if not is_valid_box(box, w, h):
                box = None
        return TrackResult(self.state, box, self.raw_box if box else None, float(self.confidence),
                           self.frames_missed, source, event,
                           (time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------- visual tracker
    def _refresh_visual(self, frame: np.ndarray, box) -> None:
        if self.cfg.DETECT_INTERVAL <= 1:
            return  # detector runs every frame, visual tracker not needed
        trk = create_visual_tracker(self.cfg.VISUAL_TRACKER)
        if trk is None:
            self.visual, self.visual_name = None, "none (Kalman prediction only)"
            return
        try:
            x, y, bw, bh = xyxy_to_xywh(box)
            trk.init(frame, (x, y, max(bw, 2), max(bh, 2)))
            self.visual = trk
            self.visual_name = type(trk).__name__
        except Exception:
            self.visual = None

    def _run_visual(self, frame: np.ndarray):
        if self.visual is None:
            return None
        try:
            ok, rect = self.visual.update(frame)
        except Exception:
            ok = False
        if not ok:
            self.visual = None  # tracker failure -> fall back to prediction
            return None
        return xywh_to_xyxy(rect)
