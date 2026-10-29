"""Small bounding-box helpers shared by the tracker and the target selector."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union of two boxes (0 = disjoint, 1 = identical)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(box: Sequence[float]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def clip_box(box: Sequence[float], w: int, h: int) -> Box:
    x1 = min(max(box[0], 0.0), w - 1.0)
    y1 = min(max(box[1], 0.0), h - 1.0)
    x2 = min(max(box[2], 0.0), w - 1.0)
    y2 = min(max(box[3], 0.0), h - 1.0)
    return x1, y1, x2, y2


def is_valid_box(box, w: int, h: int, min_size: float = 4.0) -> bool:
    if box is None or len(box) != 4:
        return False
    if any(not math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return False
    # must overlap the frame at least partially
    return x2 > 0 and y2 > 0 and x1 < w and y1 < h


def xyxy_to_xywh(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])


def xywh_to_xyxy(box: Sequence[float]) -> Box:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)
