"""
Evaluate a run on the synthetic test video against its ground truth.

    python main.py --source data/samples/test_walk.mp4 --headless --log
    python tools/evaluate_synthetic.py logs/run_<timestamp>.csv

Reports (all computed from the log + ground truth, nothing hard-coded):
  * tracking: mean IoU with the true target box, % frames tracked while visible
  * identity: frames where the followed box overlaps the DISTRACTOR more than
    the target (= identity switch)
  * position / distance classification agreement with labels derived from the
    ground-truth box using the same thresholds (config.py)
  * per-phase command distribution
  * safety: frames where the robot moved while the target was invisible

Without an argument the newest logs/run_*.csv is used.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from utils.geometry import iou  # noqa: E402

W, H = 640, 480


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fbox(r, p):
    try:
        return tuple(float(r[p + k]) for k in ("x1", "y1", "x2", "y2"))
    except (ValueError, KeyError):
        return None


def gt_labels(box, cfg):
    cx = (box[0] + box[2]) / 2
    e = (cx - W / 2) / (W / 2)
    pos = "LEFT" if e < -cfg.HORIZONTAL_THRESHOLD else ("RIGHT" if e > cfg.HORIZONTAL_THRESHOLD else "CENTER")
    r = (box[3] - box[1]) / H
    if r < cfg.TOO_FAR_HEIGHT:
        zone = "FAR"
    elif r > cfg.TOO_CLOSE_HEIGHT:
        zone = "CLOSE"
    elif cfg.IDEAL_MIN_HEIGHT <= r <= cfg.IDEAL_MAX_HEIGHT:
        zone = "IDEAL"
    else:
        zone = None  # inside a hysteresis band: either answer is acceptable
    return pos, zone


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        logs = sorted((ROOT / "logs").glob("run_*.csv"))
        if not logs:
            sys.exit("no logs/run_*.csv found - run main.py with --log first")
        log_path = logs[-1]
    gt = {int(r["frame"]): r for r in load(ROOT / "data" / "samples" / "test_walk_gt.csv")}
    log = load(log_path)
    cfg = Config()

    ious, tracked_vis, vis_frames = [], 0, 0
    id_switch, moving_blind = 0, 0
    pos_ok = pos_n = zone_ok = zone_n = 0
    per_phase = defaultdict(Counter)
    for r in log:
        f = int(r["frame"])
        g = gt.get(f)
        if g is None:
            continue
        per_phase[g["phase"]][r["command"]] += 1
        tb = (tuple(float(g[k]) for k in ("tx1", "ty1", "tx2", "ty2"))
              if g["target_visible"] == "1" else None)
        db = (tuple(float(g[k]) for k in ("dx1", "dy1", "dx2", "dy2"))
              if g["distractor_visible"] == "1" else None)
        box = fbox(r, "")
        if tb is None and r["command"] != "STOP" and r["state"] == "LOST":
            moving_blind += 1
        if tb is None:
            continue
        vis_frames += 1
        if box is not None:
            tracked_vis += 1
            ious.append(iou(box, tb))
            if db is not None and iou(box, db) > iou(box, tb):
                id_switch += 1
            pos, zone = gt_labels(tb, cfg)
            pos_n += 1
            pos_ok += int(r["position"] == pos)
            if zone is not None:
                zone_n += 1
                zone_ok += int(r["zone"] == zone)

    res = {
        "log": str(log_path.name),
        "frames_evaluated": len(log),
        "target_visible_frames": vis_frames,
        "tracked_while_visible_pct": round(100 * tracked_vis / max(1, vis_frames), 1),
        "mean_iou_with_ground_truth": round(sum(ious) / max(1, len(ious)), 3),
        "identity_switch_frames": id_switch,
        "position_agreement_pct": round(100 * pos_ok / max(1, pos_n), 1),
        "distance_zone_agreement_pct": round(100 * zone_ok / max(1, zone_n), 1),
        "moving_while_target_lost_frames": moving_blind,
        "commands_per_phase": {k: dict(v) for k, v in sorted(per_phase.items())},
    }
    print(json.dumps(res, indent=2))
    out = log_path.with_name(log_path.stem + "_eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
