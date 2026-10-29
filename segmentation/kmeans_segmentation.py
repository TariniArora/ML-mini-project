"""
K-Means colour segmentation (unsupervised ML), inspired by the reference paper's
K-Means terrain-segmentation experiment.

Pipeline
--------
frame -> resize to SEG_WIDTH px -> Gaussian blur -> CIE L*a*b* colour space
      -> feature per pixel [L, a, b, w * row]  -> K-Means (K clusters)
      -> cluster label image -> ground-seed traversability (segmentation/common.py)

Why L*a*b* instead of RGB?  Euclidean distance in Lab is closer to perceived
colour difference, and lightness (L) is separated from chroma (a, b), which
makes clusters a bit less sensitive to illumination than raw RGB.

Why a row (y) feature?  Ground usually occupies the bottom of the image; a small
spatial weight encourages spatially coherent clusters without forcing it.

Temporal stability: K-Means labels are arbitrary, so we warm-start each run with
the previous centres (fewer iterations, consistent colours between runs).
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import cv2
import numpy as np

from segmentation.common import (SegmentationResult, person_mask, render, scaled_shape,
                                 traversability)


class KMeansSegmenter:
    name = "K-Means (Lab)"

    def __init__(self, cfg):
        self.cfg = cfg
        self.centers: Optional[np.ndarray] = None

    def features(self, small_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        h, w = lab.shape[:2]
        rows = np.repeat(np.linspace(0, 1, h, dtype=np.float32)[:, None], w, axis=1)
        feats = np.dstack([lab, rows[..., None] * self.cfg.KMEANS_SPATIAL_WEIGHT])
        return feats.reshape(-1, 4)

    def cluster(self, feats: np.ndarray) -> np.ndarray:
        k = self.cfg.KMEANS_K
        if self.centers is not None and self.centers.shape == (k, feats.shape[1]):
            # warm start: assign to previous centres, then refine a few iterations
            d = ((feats[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
            init = d.argmin(1).astype(np.int32).reshape(-1, 1)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1e-3)
            _, labels, centers = cv2.kmeans(feats, k, init, crit, 1, cv2.KMEANS_USE_INITIAL_LABELS)
        else:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1e-3)
            _, labels, centers = cv2.kmeans(feats, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        self.centers = centers
        return labels.ravel()

    def segment(self, frame: np.ndarray, exclude_box: Optional[Sequence[float]] = None,
                render_vis: bool = True) -> SegmentationResult:
        t0 = time.perf_counter()
        sh, sw = scaled_shape(frame.shape, self.cfg.SEG_WIDTH)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        labels = self.cluster(self.features(small)).reshape(sh, sw).astype(np.int32)
        valid = person_mask((sh, sw), exclude_box, frame.shape)
        res = traversability(labels, valid, self.cfg, self.name, t0, small_bgr=small)
        if render_vis:
            res.vis = render(res, small, self.cfg)
        return res
