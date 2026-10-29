"""
Shared traversability logic used by BOTH segmentation methods.

Idea (classic "ground seed" free-space detection)
-------------------------------------------------
The robot is standing on traversable ground, so the patch of image directly in
front of it (bottom-centre "seed region") is assumed to be ground. Any segment
(K-Means cluster or Felzenszwalb region) that covers a large share of that seed
region is labelled TRAVERSABLE, and only pixels connected to the seed are kept.

The lower half of the image is then split into LEFT / CENTER / RIGHT corridors.
For each corridor we measure the free-space DEPTH: how far up from the bottom
edge the ground stays traversable (median over the corridor's columns, 1.0 =
free up to the image middle). If the CENTER depth is below FREE_SPACE_MIN, an
obstacle is close in front and the path is flagged as "blocked".

Pixels inside the target person's box are ignored (the person is the goal,
not an obstacle).

Limitations (stated honestly): this is colour/region based - a floor with the
same colour as a wall, strong shadows, or a laptop webcam that sees only a desk
will confuse it. That is why it is an *experimental* navigation cue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    method: str
    labels: np.ndarray                  # (h, w) int segment ids at small resolution
    traversable: np.ndarray             # (h, w) bool
    valid: np.ndarray                   # (h, w) bool (False inside the person box)
    free_ratio: Dict[str, float]        # corridor -> fraction traversable
    path_blocked: bool
    ground_visible: bool                # enough seed pixels to reason at all
    n_segments: int
    latency_ms: float
    vis: Optional[np.ndarray] = None    # BGR visualisation (small)
    note: str = ""


def scaled_shape(frame_shape: Sequence[int], width: int) -> Tuple[int, int]:
    h, w = frame_shape[:2]
    return max(8, int(round(h * width / float(w)))), width


def person_mask(shape: Tuple[int, int], box, frame_shape) -> np.ndarray:
    """True = valid pixel (not part of the target person)."""
    valid = np.ones(shape, dtype=bool)
    if box is None:
        return valid
    sh, sw = shape
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = int(x1 * sw / fw), int(np.ceil(x2 * sw / fw))
    y1, y2 = int(y1 * sh / fh), int(np.ceil(y2 * sh / fh))
    valid[max(0, y1):min(sh, y2), max(0, x1):min(sw, x2)] = False
    return valid


def seed_mask(shape: Tuple[int, int], seed_region) -> np.ndarray:
    sh, sw = shape
    rx1, ry1, rx2, ry2 = seed_region
    m = np.zeros(shape, dtype=bool)
    m[int(ry1 * sh):int(ry2 * sh), int(rx1 * sw):int(rx2 * sw)] = True
    return m


def traversability(labels: np.ndarray, valid: np.ndarray, cfg, method: str,
                   t0: float, extra_note: str = "",
                   small_bgr: Optional[np.ndarray] = None) -> SegmentationResult:
    sh, sw = labels.shape
    seed = seed_mask(labels.shape, cfg.SEED_REGION) & valid
    n_seg = int(labels.max()) + 1 if labels.size else 0
    trav = np.zeros(labels.shape, dtype=bool)
    ground_visible = seed.sum() >= 0.3 * seed_mask(labels.shape, cfg.SEED_REGION).sum()

    if ground_visible:
        seed_labels = labels[seed]
        counts = np.bincount(seed_labels, minlength=n_seg)
        share = counts / max(1, seed_labels.size)
        ground_ids = np.flatnonzero(share >= cfg.TRAVERSABLE_SEED_SHARE)
        if ground_ids.size == 0:
            ground_ids = np.array([int(np.argmax(counts))])
        if small_bgr is not None and n_seg > 1:
            # Region growing in colour space: also accept segments whose mean L*a*b*
            # colour is close to the ground colour (an over-segmented floor - e.g.
            # wooden planks split by Felzenszwalb - is still one floor).
            lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
            flat = labels.ravel()
            cnt = np.maximum(np.bincount(flat, minlength=n_seg), 1)
            means = np.stack([np.bincount(flat, weights=lab[:, c], minlength=n_seg) / cnt
                              for c in range(3)], axis=1)
            ground_col = np.average(means[ground_ids], axis=0, weights=counts[ground_ids] + 1e-6)
            dist = np.linalg.norm(means - ground_col, axis=1)
            ground_ids = np.union1d(ground_ids, np.flatnonzero(dist < cfg.GROUND_COLOR_TOLERANCE))
        cand = np.isin(labels, ground_ids) & valid
        # keep only regions connected to the seed (removes same-coloured blobs on the wall)
        n, cc = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
        keep = np.unique(cc[seed & cand])
        keep = keep[keep > 0]
        trav = np.isin(cc, keep)

    # Free-space DEPTH per corridor: for every image column of the lower half, how far
    # (from the bottom edge upwards) the ground stays traversable before the first
    # non-traversable pixel. Pixels of the target person count as passable (the
    # person is the goal, not an obstacle). Value 1.0 = free up to the image middle.
    lower = (trav | ~valid)[sh // 2:, :][::-1, :]          # bottom row first
    blocked_px = ~lower
    first_block = np.where(blocked_px.any(axis=0), blocked_px.argmax(axis=0), lower.shape[0])
    depth = first_block / float(lower.shape[0])
    thirds = [(0, sw // 3), (sw // 3, 2 * sw // 3), (2 * sw // 3, sw)]
    free = {name: float(np.median(depth[a:b]))
            for name, (a, b) in zip(("left", "center", "right"), thirds)}
    blocked = ground_visible and free["center"] < cfg.FREE_SPACE_MIN
    note = extra_note or ("" if ground_visible else "ground not visible (seed region covered)")
    return SegmentationResult(method, labels, trav, valid, free, blocked, ground_visible, n_seg,
                              (time.perf_counter() - t0) * 1000.0, note=note)


_PALETTE = np.array([
    [180, 119, 31], [14, 127, 255], [44, 160, 44], [40, 39, 214], [189, 103, 148],
    [75, 86, 140], [194, 119, 227], [127, 127, 127], [34, 189, 188], [207, 190, 23]],
    dtype=np.uint8)


def render(result: SegmentationResult, small_bgr: np.ndarray, cfg, out_size=(320, 240)) -> np.ndarray:
    """Side-by-side debug view: segments | traversable overlay."""
    labels = result.labels
    if result.n_segments <= len(_PALETTE):
        seg_col = _PALETTE[labels % len(_PALETTE)]
    else:  # many Felzenszwalb regions: colour each by its mean image colour
        seg_col = np.zeros_like(small_bgr)
        flat = labels.ravel()
        for c in range(3):
            sums = np.bincount(flat, weights=small_bgr[:, :, c].ravel().astype(float))
            cnts = np.maximum(np.bincount(flat), 1)
            seg_col[:, :, c] = (sums / cnts)[labels].astype(np.uint8)
        bnd = np.zeros(labels.shape, bool)
        bnd[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        bnd[1:, :] |= labels[1:, :] != labels[:-1, :]
        seg_col[bnd] = (255, 255, 255)

    over = small_bgr.copy()
    green = np.zeros_like(over); green[:] = (60, 200, 60)
    red = np.zeros_like(over); red[:] = (40, 40, 220)
    over = np.where(result.traversable[..., None], cv2.addWeighted(over, 0.4, green, 0.6, 0), over)
    blocked_px = (~result.traversable) & result.valid
    over = np.where(blocked_px[..., None], cv2.addWeighted(over, 0.6, red, 0.4, 0), over)
    over[~result.valid] = (over[~result.valid] * 0.3).astype(np.uint8)

    w, h = out_size
    a = cv2.resize(seg_col, (w // 2, h), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(over, (w // 2, h), interpolation=cv2.INTER_NEAREST)
    # seed region outline
    rx1, ry1, rx2, ry2 = cfg.SEED_REGION
    cv2.rectangle(b, (int(rx1 * w / 2), int(ry1 * h)), (int(rx2 * w / 2) - 1, int(ry2 * h) - 1),
                  (255, 255, 0), 1)
    # corridor bars
    for i, k in enumerate(("left", "center", "right")):
        x0 = int(i * (w / 2) / 3) + 4
        bar = int(result.free_ratio[k] * 40)
        col = (60, 200, 60) if result.free_ratio[k] >= cfg.FREE_SPACE_MIN else (40, 40, 220)
        cv2.rectangle(b, (x0, h - 4 - bar), (x0 + 12, h - 4), col, -1)
    canvas = np.hstack([a, b])
    title = f"{result.method}  segs={result.n_segments}  {result.latency_ms:.0f} ms"
    cv2.rectangle(canvas, (0, 0), (w, 16), (0, 0, 0), -1)
    cv2.putText(canvas, title, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    status = ("PATH BLOCKED" if result.path_blocked else "PATH CLEAR") if result.ground_visible \
        else "GROUND NOT VISIBLE"
    col = (40, 40, 220) if result.path_blocked else ((60, 200, 60) if result.ground_visible else (0, 200, 255))
    cv2.putText(canvas, status, (w // 2 + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return canvas
