"""
The complete perception -> decision pipeline for ONE frame, independent of any
GUI. main.py (live webcam), the headless test mode and the evaluation script all
use this same class, so what you test is exactly what runs in the demo.

    frame
      -> CameraMotionEstimator        tracking/ego_motion.py  (was the laptop rotated?)
      -> PersonDetector (ML)          detection/person_detector.py
      -> TargetTracker                tracking/tracker.py  (+ target_selector.py)
      -> PositionEstimator            navigation/position_estimator.py
      -> DistanceEstimator            navigation/distance_estimator.py
      -> TerrainAnalyzer (optional)   segmentation/*
      -> DecisionEngine               navigation/decision_engine.py
      -> RobotHeading                 navigation/heading.py  (closed-loop virtual turning)
      -> FrameState (everything the UI / simulator / logger needs)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detection.person_detector import Detection, create_detector
from navigation.decision_engine import Decision, DecisionEngine, DecisionInput
from navigation.heading import HeadingState, RobotHeading, TargetMotion
from navigation.distance_estimator import DistanceEstimator, DistanceInfo, DistanceZone
from navigation.position_estimator import HorizontalPosition, PositionEstimator, PositionInfo
from segmentation.common import SegmentationResult
from segmentation.terrain_analyzer import TerrainAnalyzer
from tracking.ego_motion import CameraMotionEstimator, EgoMotion
from tracking.tracker import TargetTracker, TrackResult, TrackState


@dataclass
class FrameState:
    frame_index: int
    frame_shape: tuple
    detections: List[Detection]
    detector_ran: bool
    track: TrackResult
    position: Optional[PositionInfo]
    distance: Optional[DistanceInfo]
    decision: Decision
    segmentation: Optional[SegmentationResult]
    detect_ms: float
    track_ms: float
    total_ms: float
    events: List[str] = field(default_factory=list)
    heading: Optional[HeadingState] = None
    ego: Optional[EgoMotion] = None
    target_motion: str = ""           # "LEFT" / "RIGHT" / "STILL" (robot's view), "" = no target

    @property
    def target_visible(self) -> bool:
        return self.track.has_target

    def target_relative(self) -> Optional[tuple]:
        """(distance m, bearing rad) for the simulator, or None."""
        if not self.target_visible or self.position is None or self.distance is None:
            return None
        d = self.distance.approx_m if self.distance.approx_m is not None else 2.0
        return d, math.radians(self.position.bearing_deg)


class FollowerPipeline:
    def __init__(self, cfg, detector=None, segmentation: str = "off"):
        self.cfg = cfg
        self.detector = detector if detector is not None else create_detector(cfg)
        self.tracker = TargetTracker(cfg)
        self.position = PositionEstimator(cfg)
        self.distance = DistanceEstimator(cfg)
        self.decision = DecisionEngine(cfg)
        self.terrain = TerrainAnalyzer(cfg, segmentation)
        self.ego = CameraMotionEstimator(cfg)
        self.heading = RobotHeading(cfg)
        self.target_motion = TargetMotion(cfg)
        self.frame_index = 0
        self.last_detections: List[Detection] = []
        self._last_now: Optional[float] = None
        self._last_box = None

    def calibrate(self, st: "FrameState") -> str:
        """Key 'c': the robot faces the camera centre again and the person's
        current distance becomes the ideal following distance."""
        self.heading.reset()
        self.target_motion.reset()
        if st.distance is None:
            return "calibrated: robot faces camera centre (no person for distance)"
        self.distance.calibrate(st.distance.smoothed_ratio)
        return f"calibrated: centre + ideal distance ({st.distance.smoothed_ratio:.2f})"

    def reset_target(self) -> None:
        self.tracker.reset(forget_appearance=True)
        self.position.reset()
        self.distance.reset()
        self.decision.reset()

    def process(self, frame: np.ndarray, now: Optional[float] = None) -> FrameState:
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        dt = min(max(now - self._last_now, 0.0), 0.25) if self._last_now is not None else 0.0
        self._last_now = now
        h, w = frame.shape[:2]
        events: List[str] = []

        # 0) did the laptop itself rotate? (background optical flow, people masked out)
        ego = None
        if self.cfg.EGO_MOTION:
            exclude = [d.as_tuple() for d in self.last_detections]
            if self._last_box is not None:
                exclude.append(self._last_box)
            ego = self.ego.update(frame, exclude)
            if ego.yaw_deg:
                self.tracker.shift(ego.dx_px, ego.dy_px)
            e = self.heading.apply_camera_rotation(ego.yaw_deg, dt, now)
            if e:
                events.append(e)

        # 1) ML detection - every DETECT_INTERVAL frames, or every frame when not locked on
        run_det = (self.frame_index % self.cfg.DETECT_INTERVAL == 0
                   or self.tracker.state != TrackState.TRACKING)
        detect_ms = 0.0
        detections = None
        if run_det:
            try:
                detections = self.detector.detect(frame)
                detect_ms = self.detector.last_latency_ms
            except Exception as exc:  # model failure must never move the robot
                events.append(f"detector error: {exc}")
                detections = []
            self.last_detections = detections

        # 2) tracking + target selection
        track = self.tracker.step(frame, detections)
        if track.event:
            events.append(track.event)

        # 3) geometry: position & distance proxy (reset when no target)
        box = track.box if track.has_target else None
        pos = self.position.update(box, w, self.heading.aim)
        dist = self.distance.update(box, w, h)

        # 4) optional terrain analysis
        seg = self.terrain.update(frame, box)

        # 5) decision
        inp = DecisionInput(
            target_visible=track.has_target,
            confidence=track.confidence if track.has_target else 0.0,
            position=pos.position if pos else HorizontalPosition.UNKNOWN,
            distance=dist.zone if dist else DistanceZone.UNKNOWN,
            path_blocked=bool(seg and seg.path_blocked),
        )
        decision = self.decision.update(inp, now)

        # 6) closed loop: the command turns the (virtual) robot -> new aim next frame
        self.heading.turn(decision.command, pos.bearing_deg if pos else None, dt)
        moving = self.target_motion.update(
            self.heading.cam_yaw + pos.cam_bearing_raw_deg if pos else None, dt)
        self._last_box = box

        state = FrameState(self.frame_index, frame.shape, self.last_detections, run_det, track,
                           pos, dist, decision, seg, detect_ms, track.latency_ms,
                           (time.perf_counter() - t0) * 1000.0, events,
                           self.heading.state(now), ego, moving)
        self.frame_index += 1
        return state
