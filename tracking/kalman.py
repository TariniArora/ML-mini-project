"""
Constant-velocity Kalman filter for a bounding box (the motion model used by SORT).

State      x = [cx, cy, w, h, vcx, vcy, vw, vh]^T   (pixels, pixels/frame)
Measurement z = [cx, cy, w, h]^T                    (from the detector / visual tracker)

predict():  x = F x,          P = F P F^T + Q
update(z):  K = P H^T (H P H^T + R)^-1,  x = x + K (z - H x),  P = (I - K H) P

Why: detections are noisy and sometimes missing. The filter smooths the box and
can *predict* where the person will be for a few frames when the detector misses
them (short occlusion, motion blur).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class KalmanBoxFilter:
    def __init__(self, box: Sequence[float], process_noise: float = 1.0,
                 measurement_noise: float = 4.0):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0          # position += velocity * dt (dt = 1 frame)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        cx, cy, w, h = self._to_z(box)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        scale = max(w, h, 1.0)
        # Uncertainty scales with box size so the same tuning works near and far.
        self.P = np.diag([scale, scale, scale, scale, 4 * scale, 4 * scale, 4 * scale, 4 * scale]) * 0.1
        self.q = process_noise
        self.r = measurement_noise

    @staticmethod
    def _to_z(box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=float)

    def _Q(self) -> np.ndarray:
        s = max(self.x[2], self.x[3], 1.0) / 100.0
        return np.diag([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]) * self.q * s

    def _R(self) -> np.ndarray:
        s = max(self.x[2], self.x[3], 1.0) / 100.0
        return np.eye(4) * self.r * s

    def predict(self) -> Tuple[float, float, float, float]:
        self.x = self.F @ self.x
        self.x[2] = max(self.x[2], 2.0)
        self.x[3] = max(self.x[3], 2.0)
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return self.box

    def update(self, box: Sequence[float]) -> Tuple[float, float, float, float]:
        z = self._to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self._R()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.box

    def damp_velocity(self, factor: float = 0.5) -> None:
        """Reduce velocity while coasting so predictions do not fly off-screen."""
        self.x[4:] *= factor

    @property
    def box(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
