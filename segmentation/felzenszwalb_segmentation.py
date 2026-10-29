"""
Felzenszwalb-Huttenlocher graph-based segmentation (optional, experimental).

The reference paper also investigated Felzenszwalb segmentation for terrain.
The algorithm builds a graph whose nodes are pixels and whose edge weights are
colour differences, then greedily merges regions when the edge between them is
weaker than the internal variation of each region (scale parameter k).

Unlike K-Means it produces spatially connected regions and does not need the
number of segments in advance - but it is slower, so it runs only every
FELZ_INTERVAL frames and on a small image. Requires scikit-image; if that is
not installed the module reports itself unavailable instead of crashing.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import cv2
import numpy as np

from segmentation.common import (SegmentationResult, person_mask, render, scaled_shape,
                                 traversability)

try:
    from skimage.segmentation import felzenszwalb as _felzenszwalb
    AVAILABLE = True
    IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    _felzenszwalb = None
    AVAILABLE = False
    IMPORT_ERROR = str(exc)


class FelzenszwalbSegmenter:
    name = "Felzenszwalb"

    def __init__(self, cfg):
        if not AVAILABLE:
            raise ImportError("scikit-image is required for Felzenszwalb segmentation "
                              f"(pip install scikit-image). Details: {IMPORT_ERROR}")
        self.cfg = cfg

    def segment(self, frame: np.ndarray, exclude_box: Optional[Sequence[float]] = None,
                render_vis: bool = True) -> SegmentationResult:
        t0 = time.perf_counter()
        sh, sw = scaled_shape(frame.shape, self.cfg.SEG_WIDTH)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        labels = _felzenszwalb(rgb, scale=self.cfg.FELZ_SCALE, sigma=self.cfg.FELZ_SIGMA,
                               min_size=self.cfg.FELZ_MIN_SIZE).astype(np.int32)
        valid = person_mask((sh, sw), exclude_box, frame.shape)
        res = traversability(labels, valid, self.cfg, self.name, t0, small_bgr=small)
        if render_vis:
            res.vis = render(res, small, self.cfg)
        return res
