"""
Webcam data-collection utility (optional - the robot works WITHOUT collecting data).

Use it to build your own test set (e.g. your lab / corridor, different lighting,
several people) for qualitative evaluation, or as a starting point if you later
want to fine-tune the detector.

    python data/collect_data.py                    # SPACE = save frame, A = auto mode, Q = quit
    python data/collect_data.py --label corridor   # saves into data/collected/corridor_<time>/
    python data/collect_data.py --interval 0.5     # auto-capture period in seconds
    python data/collect_data.py --autolabel        # also write YOLO-format PSEUDO-labels

--autolabel runs the pretrained detector and writes one .txt per image in YOLO
format (class cx cy w h, normalised). These are machine-generated pseudo-labels:
check/correct them (e.g. with LabelImg or CVAT) before using them as ground truth.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from utils.video_source import CameraUnavailableError, VideoSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Collect webcam frames")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--label", type=str, default="session")
    ap.add_argument("--interval", type=float, default=1.0, help="auto-capture period (s)")
    ap.add_argument("--autolabel", action="store_true", help="write YOLO pseudo-labels")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    out = ROOT / "data" / "collected" / f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    detector = None
    if args.autolabel:
        from config import Config
        from detection.person_detector import create_detector
        detector = create_detector(Config())

    try:
        src = VideoSource(args.camera, args.width, args.height)
    except CameraUnavailableError as exc:
        sys.exit(f"[ERROR] {exc}")

    auto, last, count = False, 0.0, 0
    print(f"Saving to {out}\n  SPACE = save   A = toggle auto-capture   Q/ESC = quit")
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                print("[WARN] frame grab failed"); time.sleep(0.1); continue
            view = frame.copy()
            status = f"saved: {count}   auto: {'ON' if auto else 'off'} ({args.interval}s)"
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Data collection", view)
            key = cv2.waitKey(1) & 0xFF
            save = key == ord(" ") or (auto and time.time() - last >= args.interval)
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
            if save:
                last = time.time()
                name = f"img_{count:05d}"
                cv2.imwrite(str(out / f"{name}.jpg"), frame)
                if detector is not None:
                    h, w = frame.shape[:2]
                    lines = []
                    for d in detector.detect(frame):
                        cx, cy = d.center
                        lines.append(f"0 {cx / w:.6f} {cy / h:.6f} {d.width / w:.6f} {d.height / h:.6f}")
                    (out / f"{name}.txt").write_text("\n".join(lines))
                count += 1
    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()
    print(f"Collected {count} images in {out}")


if __name__ == "__main__":
    main()
