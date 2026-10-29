"""
Temporal smoothing utilities.

Why this matters
----------------
A detector's box jitters by a few pixels every frame even when the person stands
still. If the robot reacted to every raw frame it would oscillate
LEFT-RIGHT-LEFT-RIGHT, waste energy, and look broken. We use three layers:

1. ExponentialSmoother  - low-pass filter on the measurements (box centre, height).
2. Hysteresis           - different thresholds to ENTER and to LEAVE a state
                          (implemented in navigation/position_estimator.py and
                          navigation/distance_estimator.py).
3. CommandDebouncer     - a new command must persist for N frames AND the old one
                          must have been active for a minimum duration before the
                          robot switches. Safety STOPs bypass the debouncer.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Generic, Optional, TypeVar

import numpy as np

T = TypeVar("T")


class ExponentialSmoother:
    """y_t = alpha * x_t + (1 - alpha) * y_{t-1}   (works on scalars and arrays)."""

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = np.asarray(x, dtype=float)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value.item() if self.value.ndim == 0 else self.value

    def reset(self) -> None:
        self.value = None


class CommandDebouncer(Generic[T]):
    """Only switch the output command when the candidate is stable.

    update(candidate, now, force=False):
        * force=True  -> switch immediately (used for safety STOP).
        * otherwise   -> switch only if `candidate` was proposed for
                         `persistence_frames` consecutive frames AND the current
                         command has been held for >= `min_duration` seconds.
    """

    def __init__(self, initial: T, persistence_frames: int = 4, min_duration: float = 0.4):
        self.current: T = initial
        self.persistence_frames = max(1, int(persistence_frames))
        self.min_duration = float(min_duration)
        self._candidate: Optional[T] = None
        self._count = 0
        self._since: Optional[float] = None   # time of the last switch
        self.switches = 0

    def update(self, candidate: T, now: Optional[float] = None, force: bool = False) -> T:
        now = time.monotonic() if now is None else now
        if candidate == self.current:
            self._candidate, self._count = None, 0
            return self.current
        if force:
            self._switch(candidate, now)
            return self.current
        if candidate == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = candidate, 1
        # (now < _since only happens if the caller switched clocks, e.g. after a reset
        #  with wall-clock time during video playback - treat as "held long enough")
        held_long_enough = (self._since is None or now < self._since
                            or (now - self._since) >= self.min_duration)
        if self._count >= self.persistence_frames and held_long_enough:
            self._switch(candidate, now)
        return self.current

    def _switch(self, cmd: T, now: float) -> None:
        self.current = cmd
        self._since = now
        self._candidate, self._count = None, 0
        self.switches += 1

    @property
    def pending(self) -> Optional[T]:
        return self._candidate

    @property
    def pending_count(self) -> int:
        return self._count


class FPSCounter:
    """Frames-per-second over a sliding window of recent frame timestamps."""

    def __init__(self, window: int = 30):
        self.times: Deque[float] = deque(maxlen=window)

    def tick(self, now: Optional[float] = None) -> float:
        self.times.append(time.perf_counter() if now is None else now)
        return self.fps

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class LatencyStats:
    """Collects latency samples (ms) and reports mean / p95 / max."""

    def __init__(self):
        self.samples = []

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))

    def summary(self) -> dict:
        if not self.samples:
            return {"n": 0}
        a = np.asarray(self.samples)
        return {"n": int(a.size), "mean_ms": round(float(a.mean()), 2),
                "median_ms": round(float(np.median(a)), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2),
                "max_ms": round(float(a.max()), 2)}
