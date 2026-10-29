"""
Frame source abstraction: webcam, video file, or a single image.

* Webcam: on Windows we try the DirectShow backend first (opens much faster
  than the default MSMF backend on many laptops), then fall back to the default.
* Video file: used for the fallback test mode (no webcam needed).
* Image: the same image is returned every frame (handy for quick checks).
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CameraUnavailableError(RuntimeError):
    pass


class VideoSource:
    def __init__(self, source: Union[int, str], width: int = 640, height: int = 480,
                 loop: bool = False):
        self.source = source
        self.width, self.height = width, height
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.image: Optional[np.ndarray] = None
        self.is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        self.is_file = not self.is_camera
        self.fps = 30.0
        self.frame_count = 0
        self.open()

    # ------------------------------------------------------------------ open
    def open(self) -> None:
        if self.is_camera:
            idx = int(self.source)
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.cap = cap
                        return
                    cap.release()
            raise CameraUnavailableError(
                f"Could not open webcam index {idx}. Is it connected / used by another app "
                f"(Zoom, Teams, browser)? Try --camera 1, or test with --source data/samples/test_walk.mp4")

        path = Path(str(self.source))
        if not path.exists():
            raise CameraUnavailableError(f"Input file not found: {path}")
        if path.suffix.lower() in IMAGE_EXTS:
            self.image = cv2.imread(str(path))
            if self.image is None:
                raise CameraUnavailableError(f"Could not read image: {path}")
            return
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise CameraUnavailableError(f"Could not open video: {path}")
        self.cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def reopen(self) -> bool:
        self.release()
        try:
            self.open()
            return True
        except CameraUnavailableError:
            return False

    # ------------------------------------------------------------------ read
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.image is not None:
            return True, self.image.copy()
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None or frame.size == 0:
            return False, None
        # keep processing resolution predictable
        if frame.shape[1] != self.width and self.is_file:
            scale = self.width / frame.shape[1]
            frame = cv2.resize(frame, (self.width, int(round(frame.shape[0] * scale))))
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
