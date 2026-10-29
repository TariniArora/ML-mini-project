"""
Person detection with a pretrained deep-learning object detector.

This is the MACHINE LEARNING core of the project:

    BGR frame  ->  CNN object detector  ->  list[Detection(box, confidence, class)]

Two interchangeable detectors are provided:

* ``YoloPersonDetector``  - Ultralytics YOLOv8n (nano), pretrained on MS-COCO.
  Default: single-stage, anchor-free, ~3.2 M parameters, fast on a laptop CPU.
* ``SsdPersonDetector``   - torchvision SSDLite320 + MobileNetV3-Large, pretrained
  on MS-COCO. This is the same *family* (MobileNet backbone + SSD head) as the
  MobileNet-SSD detector used in the reference golf-bag-following-robot paper,
  so you can compare the two during the demo with ``--detector ssd``.

Both only return the COCO class ``person``; everything else is discarded.
No training happens here - we use the publicly released pretrained weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


class DetectorUnavailableError(RuntimeError):
    """Raised when no person detector could be loaded."""


@dataclass
class Detection:
    """One detected person. Coordinates are pixels in the original frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "person"

    # --------------------------------------------------------------- helpers
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_int(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def clipped(self, frame_w: int, frame_h: int) -> "Detection":
        return Detection(
            float(np.clip(self.x1, 0, frame_w - 1)), float(np.clip(self.y1, 0, frame_h - 1)),
            float(np.clip(self.x2, 0, frame_w - 1)), float(np.clip(self.y2, 0, frame_h - 1)),
            self.confidence, self.class_id, self.class_name)

    def is_valid(self, frame_w: int, frame_h: int, min_area_ratio: float = 0.0) -> bool:
        """Reject NaNs, inverted boxes and boxes that are too small."""
        vals = np.array(self.as_tuple(), dtype=float)
        if not np.all(np.isfinite(vals)):
            return False
        if self.width <= 2 or self.height <= 2:
            return False
        return self.area >= min_area_ratio * frame_w * frame_h


def _postprocess(boxes: np.ndarray, scores: np.ndarray, frame_shape: Sequence[int],
                 conf_threshold: float, min_area_ratio: float) -> List[Detection]:
    h, w = frame_shape[:2]
    out: List[Detection] = []
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        if s < conf_threshold:
            continue
        det = Detection(float(x1), float(y1), float(x2), float(y2), float(s)).clipped(w, h)
        if det.is_valid(w, h, min_area_ratio):
            out.append(det)
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out


class BasePersonDetector:
    name = "base"

    def __init__(self, conf_threshold: float, min_area_ratio: float):
        self.conf_threshold = conf_threshold
        self.min_area_ratio = min_area_ratio
        self.last_latency_ms = 0.0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []
        t0 = time.perf_counter()
        dets = self._detect(frame)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return dets

    def _detect(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover
        raise NotImplementedError


class YoloPersonDetector(BasePersonDetector):
    """YOLOv8-nano person detector (Ultralytics)."""

    name = "YOLOv8n"
    PERSON_CLASS_ID = 0  # COCO index of "person"

    def __init__(self, model_path: str, conf_threshold: float = 0.45,
                 min_area_ratio: float = 0.004, imgsz: int = 320, device: str = "cpu"):
        super().__init__(conf_threshold, min_area_ratio)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                "The 'ultralytics' package is missing. Install it with: pip install ultralytics"
            ) from exc
        path = Path(model_path)
        # If the local file is missing, Ultralytics downloads "yolov8n.pt" automatically.
        source = str(path) if path.exists() else path.name
        try:
            self.model = YOLO(source)
        except Exception as exc:  # network error, corrupt file, ...
            raise DetectorUnavailableError(f"Could not load YOLO model '{source}': {exc}") from exc
        self.imgsz = imgsz
        self.device = device
        # Warm-up: the very first inference is slow (graph build / memory alloc).
        self._detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID], device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return _postprocess(boxes, scores, frame.shape, self.conf_threshold, self.min_area_ratio)


class SsdPersonDetector(BasePersonDetector):
    """MobileNetV3-Large + SSDLite (torchvision) person detector."""

    name = "MobileNetV3-SSDLite"
    PERSON_LABEL = 1  # torchvision COCO labels: 0 = background, 1 = person

    def __init__(self, weights_path: str, conf_threshold: float = 0.45,
                 min_area_ratio: float = 0.004, device: str = "cpu"):
        super().__init__(conf_threshold, min_area_ratio)
        try:
            import torch
            from torchvision.models.detection import (
                SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
        except ImportError as exc:
            raise DetectorUnavailableError(
                "torch/torchvision are missing (they are installed together with ultralytics).") from exc
        self.torch = torch
        try:
            if Path(weights_path).exists():
                model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:  # downloads from download.pytorch.org on first use
                model = ssdlite320_mobilenet_v3_large(
                    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        except Exception as exc:
            raise DetectorUnavailableError(f"Could not load MobileNet-SSD weights: {exc}") from exc
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = model.eval().to(self.device)
        self._detect(np.zeros((320, 320, 3), dtype=np.uint8))

    def _detect(self, frame: np.ndarray) -> List[Detection]:
        torch = self.torch
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        with torch.inference_mode():
            out = self.model([tensor])[0]
        keep = out["labels"] == self.PERSON_LABEL
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        return _postprocess(boxes, scores, frame.shape, self.conf_threshold, self.min_area_ratio)


def create_detector(cfg, preferred: Optional[str] = None) -> BasePersonDetector:
    """Create the configured detector, falling back to the other one if it fails.

    Raises DetectorUnavailableError only if *no* detector can be loaded.
    """
    order = [(preferred or cfg.DETECTOR).lower()]
    order += [d for d in ("yolo", "ssd") if d not in order]
    errors = []
    for kind in order:
        try:
            if kind == "yolo":
                return YoloPersonDetector(cfg.YOLO_MODEL_PATH, cfg.CONFIDENCE_THRESHOLD,
                                          cfg.MIN_BOX_AREA_RATIO, cfg.DETECTOR_IMGSZ, cfg.DEVICE)
            if kind == "ssd":
                return SsdPersonDetector(cfg.SSD_WEIGHTS_PATH, cfg.CONFIDENCE_THRESHOLD,
                                         cfg.MIN_BOX_AREA_RATIO, cfg.DEVICE)
            errors.append(f"unknown detector '{kind}'")
        except DetectorUnavailableError as exc:
            errors.append(f"{kind}: {exc}")
            print(f"[WARN] {kind} detector unavailable -> trying fallback. ({exc})")
    raise DetectorUnavailableError("No person detector could be loaded:\n  " + "\n  ".join(errors))
