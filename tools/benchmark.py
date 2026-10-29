"""
Benchmark several pipeline configurations on the synthetic test video.

    python tools/benchmark.py            # writes logs/benchmark.json + prints a table

Configurations: YOLOv8n vs MobileNetV3-SSDLite, detector every frame vs every 3rd
frame (visual tracker in between), K-Means vs Felzenszwalb segmentation cost.
All numbers are measured on YOUR machine when you run it - re-run on the laptop
used for the demo and copy the table into the report.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from detection.person_detector import create_detector  # noqa: E402
from pipeline import FollowerPipeline  # noqa: E402
from tools.evaluate_synthetic import gt_labels, load  # noqa: E402
from utils.geometry import iou  # noqa: E402

VIDEO = ROOT / "data" / "samples" / "test_walk.mp4"
GT = ROOT / "data" / "samples" / "test_walk_gt.csv"


class NoisyDetector:
    """Wraps a detector and adds Gaussian jitter to its boxes (robustness ablation,
    imitating the frame-to-frame box noise of a real webcam)."""

    def __init__(self, inner, sigma_px: float, seed: int = 0):
        self.inner, self.sigma = inner, sigma_px
        self.rng = np.random.default_rng(seed)
        self.name = inner.name + f" +noise{sigma_px:g}px"
        self.last_latency_ms = 0.0

    def detect(self, frame):
        from detection.person_detector import Detection
        dets = self.inner.detect(frame)
        self.last_latency_ms = self.inner.last_latency_ms
        out = []
        for d in dets:
            n = self.rng.normal(0, self.sigma, 4)
            out.append(Detection(d.x1 + n[0], d.y1 + n[1], d.x2 + n[2], d.y2 + n[3], d.confidence))
        return out


def run(detector_kind: str, detect_interval: int = 1, segmentation: str = "off",
        smoothing: bool = True, box_noise_px: float = 0.0):
    cfg = Config()
    # labels are scored against the camera frame -> measure the open-loop pipeline
    cfg.CLOSED_LOOP_HEADING, cfg.EGO_MOTION = False, False
    if not smoothing:  # ablation: raw per-frame decisions
        cfg.SMOOTHING_FACTOR, cfg.HORIZONTAL_HYSTERESIS = 1.0, 0.0
        cfg.COMMAND_PERSISTENCE_FRAMES, cfg.MIN_COMMAND_DURATION = 1, 0.0
        cfg.TOO_FAR_HEIGHT, cfg.TOO_CLOSE_HEIGHT = cfg.IDEAL_MIN_HEIGHT, cfg.IDEAL_MAX_HEIGHT
    cfg.DETECTOR = detector_kind
    cfg.DETECT_INTERVAL = detect_interval
    det = create_detector(cfg)
    if box_noise_px > 0:
        det = NoisyDetector(det, box_noise_px)
    pipe = FollowerPipeline(cfg, detector=det, segmentation=segmentation)
    gt = {int(r["frame"]): r for r in load(GT)}
    cap = cv2.VideoCapture(str(VIDEO))
    det_ms, pipe_ms, seg_ms, ious, pos_ok, n = [], [], [], [], 0, 0
    switches_seen = 0
    t0 = time.perf_counter()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = pipe.process(frame, now=i / 20.0)
        if st.detector_ran:
            det_ms.append(st.detect_ms)
        pipe_ms.append(st.total_ms)
        if st.segmentation is not None and st.segmentation.latency_ms > 0 and \
                pipe.terrain._frame_counter % pipe.terrain.interval == 1:
            seg_ms.append(st.segmentation.latency_ms)
        g = gt[i]
        if g["target_visible"] == "1" and st.track.has_target:
            tb = tuple(float(g[k]) for k in ("tx1", "ty1", "tx2", "ty2"))
            ious.append(iou(st.track.box, tb))
            n += 1
            pos_ok += int(st.position.position.value == gt_labels(tb, cfg)[0])
            if g["distractor_visible"] == "1":
                db = tuple(float(g[k]) for k in ("dx1", "dy1", "dx2", "dy2"))
                switches_seen += int(iou(st.track.box, db) > iou(st.track.box, tb))
        i += 1
    wall = time.perf_counter() - t0
    vis = sum(1 for r in gt.values() if r["target_visible"] == "1")
    return {
        "detector": det.name, "detect_interval": detect_interval, "segmentation": segmentation,
        "smoothing": "on" if smoothing else "OFF",
        "frames": i, "fps_pipeline_only": round(i / wall, 1),
        "detector_ms_mean": round(float(np.mean(det_ms)), 1),
        "pipeline_ms_mean": round(float(np.mean(pipe_ms)), 1),
        "segmentation_ms_mean": round(float(np.mean(seg_ms)), 1) if seg_ms else None,
        "tracked_while_visible_pct": round(100 * n / vis, 1),
        "mean_iou": round(float(np.mean(ious)), 3) if ious else 0,
        "position_agreement_pct": round(100 * pos_ok / max(1, n), 1),
        "identity_switch_frames": switches_seen,
        "command_switches": pipe.decision.debouncer.switches,
    }


def main():
    if not VIDEO.exists():
        sys.exit("run: python tools/make_test_video.py  first")
    configs = [("yolo", 1, "off"), ("yolo", 3, "off"), ("ssd", 1, "off"),
               ("yolo", 1, "kmeans"), ("yolo", 1, "felzenszwalb"),
               # smoothing ablation: with and without temporal smoothing, on clean and jittered boxes
               ("yolo", 1, "off", False), ("yolo", 1, "off", True, 12.0), ("yolo", 1, "off", False, 12.0)]
    rows = []
    for c in configs:
        print("running", c, "...")
        rows.append(run(*c))
    meta = {"machine": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "opencv": cv2.__version__}
    out = ROOT / "logs" / "benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": rows}, indent=2))
    keys = ["detector", "detect_interval", "segmentation", "smoothing", "fps_pipeline_only", "detector_ms_mean",
            "segmentation_ms_mean", "tracked_while_visible_pct", "mean_iou", "position_agreement_pct",
            "identity_switch_frames", "command_switches"]
    print("\n| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
