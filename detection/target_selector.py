"""
Target selection: decide WHICH detected person the robot should follow.

Rules (as required by the project brief)
---------------------------------------
1. Only one person visible and no current target -> follow that person.
2. While a target is being tracked -> only accept the detection that is
   spatially consistent with the tracker's prediction (high IoU / small jump).
   We never jump to a different person just because they are more confident.
3. No previous target -> pick the person closest to the image centre.
4. After the target was LOST -> first try to re-identify the *same* person by
   appearance (HSV colour histogram of the torso), otherwise fall back to rule 3.

The appearance model is a classic hand-crafted feature (colour histogram) +
similarity matching, the same idea used in many re-identification baselines.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.person_detector import Detection
from utils.geometry import center_distance, iou


class AppearanceModel:
    """HSV hue/saturation histogram of the target's torso region."""

    H_BINS, S_BINS = 16, 8

    def __init__(self, update_rate: float = 0.05):
        self.update_rate = update_rate
        self.hist: Optional[np.ndarray] = None

    @classmethod
    def describe(cls, frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        # torso = central 60 % horizontally, 20-60 % vertically (avoids background + legs)
        tx1, tx2 = int(max(0, x1 + 0.2 * bw)), int(min(w, x2 - 0.2 * bw))
        ty1, ty2 = int(max(0, y1 + 0.2 * bh)), int(min(h, y1 + 0.6 * bh))
        if tx2 - tx1 < 4 or ty2 - ty1 < 4:
            return None
        patch = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([patch], [0, 1], None, [cls.H_BINS, cls.S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist

    def update(self, frame: np.ndarray, box: Sequence[float]) -> None:
        hist = self.describe(frame, box)
        if hist is None:
            return
        if self.hist is None:
            self.hist = hist
        else:
            self.hist = (1 - self.update_rate) * self.hist + self.update_rate * hist

    def similarity(self, frame: np.ndarray, box: Sequence[float]) -> float:
        """1.0 = identical colour distribution, 0.0 = completely different."""
        if self.hist is None:
            return 0.0
        hist = self.describe(frame, box)
        if hist is None:
            return 0.0
        dist = cv2.compareHist(self.hist.astype(np.float32), hist.astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)
        return float(1.0 - dist)

    def reset(self) -> None:
        self.hist = None

    @property
    def ready(self) -> bool:
        return self.hist is not None


class TargetSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def _usable(self, detections: List[Detection], frame_shape: Sequence[int]) -> List[Detection]:
        """Drop low-confidence and invalid (NaN / inverted / tiny) boxes."""
        h, w = frame_shape[:2]
        return [d for d in detections
                if d.confidence >= self.cfg.CONFIDENCE_THRESHOLD and d.is_valid(w, h)]

    # ------------------------------------------------------------ association
    def match(self, detections: List[Detection], predicted_box: Sequence[float],
              frame_shape: Sequence[int], frame: Optional[np.ndarray] = None,
              appearance: Optional[AppearanceModel] = None) -> Optional[Detection]:
        """Find the detection that continues the current track (or None).

        Gating: a candidate must overlap the predicted box (IoU) OR be within
        MAX_ASSOCIATION_DIST * frame_diagonal of it. Among gated candidates the
        best combined score wins: IoU + proximity (+ appearance if available).
        """
        cands = self._usable(detections, frame_shape)
        if not cands:
            return None
        h, w = frame_shape[:2]
        diag = float(np.hypot(w, h))
        max_dist = self.cfg.MAX_ASSOCIATION_DIST * diag
        best, best_score = None, -1.0
        for d in cands:
            ov = iou(d.as_tuple(), predicted_box)
            dist = center_distance(d.as_tuple(), predicted_box)
            if ov < self.cfg.MIN_IOU_MATCH and dist > max_dist:
                continue  # gated out: physically implausible jump
            score = ov + (1.0 - dist / max_dist if dist < max_dist else 0.0)
            if appearance is not None and appearance.ready and frame is not None:
                score += 0.5 * appearance.similarity(frame, d.as_tuple())
            if score > best_score:
                best, best_score = d, score
        return best

    # ------------------------------------------------------------ acquisition
    def acquire(self, detections: List[Detection], frame_shape: Sequence[int],
                frame: Optional[np.ndarray] = None,
                appearance: Optional[AppearanceModel] = None) -> Tuple[Optional[Detection], str]:
        """Pick a brand-new target. Returns (detection, human-readable reason)."""
        cands = self._usable(detections, frame_shape)
        if not cands:
            return None, "no person detected"

        # Re-identification of a previously followed person.
        if appearance is not None and appearance.ready and frame is not None:
            sims = [(appearance.similarity(frame, d.as_tuple()), d) for d in cands]
            sim, best = max(sims, key=lambda t: t[0])
            if sim >= self.cfg.REID_SIMILARITY_THRESHOLD:
                return best, f"re-identified previous target (similarity {sim:.2f})"
            # Nobody looks like the old target -> fall through to the normal rules.

        if len(cands) == 1:
            return cands[0], "only one person visible"

        h, w = frame_shape[:2]
        fcx, fcy = w / 2.0, h / 2.0
        best = min(cands, key=lambda d: (d.center[0] - fcx) ** 2 + (d.center[1] - fcy) ** 2)
        return best, f"closest to image centre (of {len(cands)} people)"
