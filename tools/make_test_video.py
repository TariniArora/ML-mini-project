"""
Generate a SYNTHETIC test video (+ ground truth) for testing without a webcam.

Real person crops (from the sample images bundled with the ultralytics package)
are pasted onto a synthetic room background and moved along a scripted path
that exercises every behaviour of the robot:

    phase  frames     target position / size        expected behaviour
    A      0-59       centre, small (far)           FORWARD
    B      60-119     moves left, far               LEFT + FORWARD
    C      120-179    grows to ideal, moves right   RIGHT
    D      180-239    centre, ideal                 STOP (hold)
    E      240-289    grows very large (close)      BACKWARD
    F      290-314    disappears (occlusion)        coast -> LOST -> STOP
    G      315-439    re-appears; a SECOND person   keep following the original
                      walks across the scene        target (no identity switch)

Outputs  data/samples/test_walk.mp4, data/samples/test_walk_gt.csv,
         data/samples/test_frame.jpg

The detector has NOT seen these exact composites, but they are synthetic - they
test the software pipeline, not real-world accuracy.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "samples"
W, H, FPS = 640, 480, 20


def find_assets() -> Path:
    try:
        import ultralytics
        p = Path(ultralytics.__file__).parent / "assets"
        if (p / "bus.jpg").exists():
            return p
    except ImportError:
        pass
    sys.exit("Needs the ultralytics package (for its bundled sample images): pip install ultralytics")


def background(rng) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    horizon = int(H * 0.55)
    img[:horizon] = (175, 190, 205)                       # wall
    for x in range(0, W, 80):                             # some wall texture
        cv2.rectangle(img, (x + 10, 60), (x + 50, 140), (150, 170, 190), -1)
    floor = np.zeros((H - horizon, W, 3), np.uint8)
    floor[:] = (70, 95, 120)                              # wooden floor
    for y in range(0, H - horizon, 18):
        cv2.line(floor, (0, y), (W, y), (60, 82, 105), 2)
    img[horizon:] = floor
    cv2.rectangle(img, (40, horizon - 70), (150, horizon + 40), (60, 60, 60), -1)  # cabinet
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def paste(canvas, crop, cx, height_ratio):
    h = int(np.clip(height_ratio, 0.05, 1.2) * H)
    w = int(crop.shape[1] * h / crop.shape[0])
    person = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    bottom = int(min(H * 0.62 + h * 0.5, h + 4))   # feet move down as the person gets closer
    x1, y1 = int(cx * W - w / 2), bottom - h
    x2, y2 = x1 + w, y1 + h
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    canvas[cy1:cy2, cx1:cx2] = person[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    return cx1, cy1, cx2, cy2


def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0, 1))


def target_path(f):
    """(cx ratio, height ratio, visible) for frame f."""
    if f < 60:
        return 0.50, 0.38, True
    if f < 120:
        t = (f - 60) / 40
        return lerp(0.50, 0.18, t), 0.38, True
    if f < 180:
        t = (f - 120) / 40
        return lerp(0.18, 0.80, t), lerp(0.38, 0.66, t), True
    if f < 240:
        t = (f - 180) / 25
        return lerp(0.80, 0.50, t), 0.66, True
    if f < 290:
        t = (f - 240) / 25
        return 0.50, lerp(0.66, 0.97, t), True
    if f < 315:
        return 0.50, 0.97, False
    t = (f - 315) / 30
    return lerp(0.45, 0.40, t), lerp(0.50, 0.42, t), True


def main():
    rng = np.random.default_rng(7)
    assets = find_assets()
    bus = cv2.imread(str(assets / "bus.jpg"))
    target_crop = bus[394:903, 46:232]           # pedestrian 1
    other_crop = bus[378:872, 670:809]           # pedestrian 3 (different clothes)
    bg = background(rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(OUT_DIR / "test_walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    with open(OUT_DIR / "test_walk_gt.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "phase", "target_visible", "tx1", "ty1", "tx2", "ty2",
                     "distractor_visible", "dx1", "dy1", "dx2", "dy2"])
        for f in range(440):
            frame = bg.copy()
            cx, hr, vis = target_path(f)
            cx += rng.normal(0, 0.004)                   # small natural sway
            dbox = None
            if f >= 345:                                  # distractor crosses right -> left, behind
                t = (f - 345) / 95
                dbox = paste(frame, other_crop, lerp(0.95, 0.05, t), 0.36)
            tbox = paste(frame, target_crop, cx, hr) if vis else None
            phase = "ABCDEFG"[sum(f >= b for b in (60, 120, 180, 240, 290, 315))]
            vw.write(frame)
            if f == 30:
                cv2.imwrite(str(OUT_DIR / "test_frame.jpg"), frame)
            wr.writerow([f, phase, int(tbox is not None), *(tbox or ("", "", "", "")),
                         int(dbox is not None), *(dbox or ("", "", "", ""))])
    vw.release()
    print(f"wrote {OUT_DIR / 'test_walk.mp4'} (+ _gt.csv, test_frame.jpg)")


if __name__ == "__main__":
    main()
