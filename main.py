"""
Autonomous Computer-Vision-Based Human-Following Robot  -  entry point.

Examples
--------
    python main.py                      # webcam 0, camera view + status panel
    python main.py --demo               # faculty demo: everything incl. simulated robot
    python main.py --simulate           # add the simulated robot window
    python main.py --debug              # extra overlays, latencies, K-Means debug view
    python main.py --segmentation       # K-Means terrain segmentation view
    python main.py --segmentation felzenszwalb
    python main.py --camera 1           # another webcam
    python main.py --detector ssd       # MobileNet-SSD (reference-paper detector family)
    python main.py --source data/samples/test_walk.mp4   # no webcam needed
    python main.py --source data/samples/test_walk.mp4 --headless --log   # benchmark

Keys (in the window)
--------------------
    q / ESC  quit            p  pause (robot STOPs)      h  help
    d  toggle debug          s  cycle segmentation (off -> kmeans -> felzenszwalb)
    r  reset / re-pick target
    c  calibrate: robot faces the camera centre, current distance = IDEAL distance
    o  save a screenshot to logs/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check_packages() -> None:
    missing = []
    for mod, pip_name in (("cv2", "opencv-python"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("Missing required packages: " + ", ".join(missing))
        print("Install everything with:  pip install -r requirements.txt")
        sys.exit(1)


_check_packages()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import LOGS_DIR, Config  # noqa: E402
from detection.person_detector import DetectorUnavailableError, create_detector  # noqa: E402
from navigation.decision_engine import RobotCommand  # noqa: E402
from pipeline import FollowerPipeline  # noqa: E402
from simulation.robot_simulator import RobotSimulator  # noqa: E402
from utils.run_logger import RunLogger  # noqa: E402
from utils.smoothing import FPSCounter  # noqa: E402
from utils.video_source import CameraUnavailableError, VideoSource  # noqa: E402
from visualization.dashboard import Dashboard  # noqa: E402

SEG_WINDOW = "Terrain segmentation (experimental)"
HELP_LINES = [
    "q / ESC : quit",
    "p       : pause (robot stops)",
    "d       : toggle debug overlays",
    "s       : segmentation off/kmeans/felzenszwalb",
    "r       : reset target (re-select person)",
    "c       : calibrate (stand centred, ideal distance)",
    "o       : save screenshot",
    "h       : show / hide this help",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Autonomous CV-based human-following robot (simulated)")
    p.add_argument("--camera", type=int, default=None, help="webcam index (default 0)")
    p.add_argument("--source", type=str, default=None,
                   help="video/image file instead of the webcam (fallback test mode)")
    p.add_argument("--demo", action="store_true", help="faculty demo mode (camera + panel + simulator)")
    p.add_argument("--debug", action="store_true", help="debug overlays, latencies and K-Means view")
    p.add_argument("--simulate", action="store_true", help="show the simulated robot")
    p.add_argument("--segmentation", nargs="?", const="kmeans", default=None,
                   choices=["off", "kmeans", "felzenszwalb"], help="terrain segmentation method")
    p.add_argument("--terrain-veto", action="store_true",
                   help="EXPERIMENTAL: let segmentation block forward motion")
    p.add_argument("--detector", choices=["yolo", "ssd"], default=None, help="person detector")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO input size (320/416/640)")
    p.add_argument("--conf", type=float, default=None, help="confidence threshold")
    p.add_argument("--detect-interval", type=int, default=None,
                   help="run detector every N frames, visual tracker in between")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda:0")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--headless", action="store_true", help="no windows (tests/benchmarks)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--loop", action="store_true", help="loop the --source video")
    p.add_argument("--open-loop", action="store_true",
                   help="robot aim stays at the frame centre (old behaviour, used for evaluation)")
    p.add_argument("--no-ego", action="store_true", help="do not measure laptop rotation")
    p.add_argument("--log", action="store_true", help="write logs/run_<time>.csv + .json summary")
    p.add_argument("--save-video", type=str, default=None, help="save the dashboard video (mp4)")
    return p.parse_args(argv)


def build_config(args) -> Config:
    cfg = Config()
    if args.camera is not None:
        cfg.CAMERA_INDEX = args.camera
    if args.detector:
        cfg.DETECTOR = args.detector
    if args.imgsz:
        cfg.DETECTOR_IMGSZ = args.imgsz
    if args.conf is not None:
        cfg.CONFIDENCE_THRESHOLD = args.conf
    if args.detect_interval:
        cfg.DETECT_INTERVAL = args.detect_interval
    if args.device:
        cfg.DEVICE = args.device
    if args.width:
        cfg.FRAME_WIDTH = args.width
    if args.height:
        cfg.FRAME_HEIGHT = args.height
    if args.terrain_veto:
        cfg.USE_TERRAIN_VETO = True
    if args.open_loop:
        cfg.CLOSED_LOOP_HEADING = False
    if args.no_ego:
        cfg.EGO_MOTION = False
    cfg.validate()
    return cfg


def error_frame(w: int, h: int, msg: str) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, msg, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 235), 2, cv2.LINE_AA)
    cv2.putText(img, "ROBOT COMMAND: STOP", (20, h // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (60, 60, 235), 2, cv2.LINE_AA)
    return img


def run(args) -> int:
    try:
        cfg = build_config(args)
    except ValueError as exc:
        print(f"[ERROR] invalid configuration: {exc}")
        return 1
    debug = args.debug
    show_sim = args.simulate or args.demo
    seg_method = args.segmentation or ("kmeans" if debug else "off")
    if args.terrain_veto and seg_method == "off":
        seg_method = "kmeans"
    mode = "DEMO" if args.demo else ("DEBUG" if debug else "NORMAL")
    gui = not args.headless

    print("=" * 64)
    print(" Autonomous CV Human-Following Robot  |  mode:", mode)
    print("=" * 64)

    # ---- model ---------------------------------------------------------------
    print(f"[INFO] loading person detector ({cfg.DETECTOR}) ...")
    try:
        detector = create_detector(cfg)
    except DetectorUnavailableError as exc:
        print(f"[ERROR] {exc}\n        Robot stays STOPPED. Run: pip install -r requirements.txt")
        return 2
    print(f"[INFO] detector ready: {detector.name}")

    # ---- input ---------------------------------------------------------------
    src = args.source if args.source is not None else cfg.CAMERA_INDEX
    try:
        source = VideoSource(src, cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT, loop=args.loop)
    except CameraUnavailableError as exc:
        print(f"[ERROR] {exc}")
        return 3
    print(f"[INFO] input: {'webcam ' + str(src) if source.is_camera else src}")

    pipeline = FollowerPipeline(cfg, detector=detector, segmentation=seg_method)
    sim = RobotSimulator(cfg)
    dash = Dashboard(cfg, mode)
    fps_counter = FPSCounter()
    log_path = LOGS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv" if args.log else None
    logger = RunLogger(log_path)
    writer = None

    if gui:
        cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    show_help_until = time.monotonic() + (8.0 if args.demo else 0.0)
    paused, read_failures, seg_window_open = False, 0, False
    last_t = time.monotonic()
    last_composed = None
    exit_code = 0

    try:
        while True:
            loop_t0 = time.perf_counter()
            if paused:
                key = cv2.waitKey(30) & 0xFF if gui else ord("p")
                if key in (ord("p"), ord(" ")):
                    paused = False
                    last_t = time.monotonic()
                elif key in (ord("q"), 27):
                    break
                continue

            ok, frame = source.read()
            if not ok:
                if source.is_file and not args.loop:
                    print("[INFO] end of input video")
                    break
                read_failures += 1
                pipeline.decision.reset()            # SAFETY: camera failure -> STOP
                sim.step(RobotCommand.STOP, 0.0)
                if gui:
                    cv2.imshow(cfg.WINDOW_NAME, error_frame(cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT,
                                                            "CAMERA ERROR - reconnecting..."))
                    if (cv2.waitKey(100) & 0xFF) in (ord("q"), 27):
                        break
                if read_failures % cfg.CAMERA_REOPEN_AFTER_FAILURES == 0:
                    print("[WARN] camera read failed repeatedly - reopening")
                    if not source.reopen() and read_failures > 5 * cfg.CAMERA_REOPEN_AFTER_FAILURES:
                        print("[ERROR] camera lost permanently. Robot STOPPED.")
                        exit_code = 3
                        break
                continue
            read_failures = 0

            now = time.monotonic()
            # Files: simulate at the video's own frame rate so playback is consistent.
            dt = (1.0 / source.fps) if source.is_file else (now - last_t)
            last_t = now

            st = pipeline.process(frame, now if source.is_camera else st_time(pipeline, source))
            for e in st.events:
                dash.log_event(f"[{st.frame_index}] {e}")
                if args.headless or debug:
                    print(f"[EVENT] frame {st.frame_index}: {e}")

            sim.step(st.decision.command, dt, st.target_relative(),
                     st.heading.robot_yaw if cfg.CLOSED_LOOP_HEADING and st.heading else None)
            fps = fps_counter.tick()

            if gui or args.save_video:
                cam = dash.draw_camera(frame, st, debug)
                panel = dash.draw_panel(st, cam.shape[0], fps, detector.name,
                                        pipeline.terrain.method, debug)
                parts = [cam, panel]
                if show_sim:
                    parts.append(sim.render(st.target_visible))
                composed = dash.compose(parts)
                if time.monotonic() < show_help_until:
                    dash.help_overlay(composed, ["DEMO MODE - controls"] + HELP_LINES)
                last_composed = composed
                if args.save_video:
                    if writer is None:
                        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source.fps if source.is_file else 20.0,
                                                 (composed.shape[1], composed.shape[0]))
                    writer.write(composed)
                if gui:
                    cv2.imshow(cfg.WINDOW_NAME, composed)
                    seg = st.segmentation
                    if seg is not None and seg.vis is not None:
                        cv2.imshow(SEG_WINDOW, seg.vis)
                        seg_window_open = True
                    elif seg_window_open:
                        _close_window(SEG_WINDOW)
                        seg_window_open = False

            logger.log(st, fps, (time.perf_counter() - loop_t0) * 1000.0)

            if gui and source.is_file:  # play videos at their real speed, not faster
                spare = 1.0 / source.fps - (time.perf_counter() - loop_t0)
                if spare > 0:
                    time.sleep(spare)

            if gui:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if st.frame_index > 5 and \
                        cv2.getWindowProperty(cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
                if key == ord("p"):
                    paused = True
                    pipeline.decision.reset()
                    sim.step(RobotCommand.STOP, 0.0)
                    print("[INFO] paused - robot STOPPED (press p to resume)")
                elif key == ord("d"):
                    debug = not debug
                elif key == ord("s"):
                    m = pipeline.terrain.cycle()
                    dash.log_event(f"segmentation: {m}")
                elif key == ord("r"):
                    pipeline.reset_target()
                    dash.log_event("target reset by user")
                elif key == ord("c"):
                    dash.log_event(pipeline.calibrate(st))
                    sim.reset()
                elif key == ord("o") and last_composed is not None:
                    LOGS_DIR.mkdir(exist_ok=True)
                    fn = LOGS_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(fn), last_composed)
                    dash.log_event(f"saved {fn.name}")
                elif key == ord("h"):
                    show_help_until = 0 if time.monotonic() < show_help_until else time.monotonic() + 1e9

            if args.max_frames and st.frame_index + 1 >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user (Ctrl+C)")
    finally:
        # SAFETY: whatever happens, the last command sent is STOP.
        sim.step(RobotCommand.STOP, 0.0)
        source.release()
        if writer is not None:
            writer.release()
        if gui:
            cv2.destroyAllWindows()
        summary = logger.summary(
            {"detector": detector.name, "input": str(src), "mode": mode,
             "segmentation": pipeline.terrain.method,
             "target_acquisitions": pipeline.tracker.target_switches,
             "simulator_odometer_m": round(sim.odometer, 2),
             "config": {k: v for k, v in cfg.to_dict().items() if not isinstance(v, tuple)}},
            command_switches=pipeline.decision.debouncer.switches)
        logger.close(summary)
        brief = {k: summary[k] for k in ("frames", "average_fps_end_to_end", "detector_latency",
                                         "tracker_latency", "pipeline_latency",
                                         "command_distribution", "debounced_command_switches",
                                         "target_acquisitions")}
        print("[SUMMARY]", json.dumps(brief, indent=2))
        if log_path:
            print(f"[INFO] log written to {log_path} (+ .json summary)")
        print("[INFO] robot command: STOP. Bye.")
    return exit_code


def st_time(pipeline, source) -> float:
    """Synthetic clock for video files: frame index / fps (makes runs reproducible)."""
    return pipeline.frame_index / max(source.fps, 1.0)


def _close_window(name: str) -> None:
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


if __name__ == "__main__":
    sys.exit(run(parse_args()))
