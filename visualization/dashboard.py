"""
Live visualisation: camera overlay + status panel (+ simulator, + segmentation).

Layout of the main window:

    +---------------------------+-----------+------------------+
    |  camera feed with overlay |  status   |  robot simulator |
    |  (boxes, centres, error)  |  panel    |  (top view)      |
    +---------------------------+-----------+------------------+

The segmentation debug view is shown in a second window when enabled.
OpenCV's Hershey fonts cannot draw unicode arrows, so command arrows are drawn
with cv2.arrowedLine.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from navigation.decision_engine import RobotCommand
from navigation.distance_estimator import DistanceZone
from navigation.heading import bearing_to_x
from navigation.position_estimator import HorizontalPosition
from simulation.robot_simulator import COMMAND_COLORS
from tracking.tracker import TrackState

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, DARK = (235, 235, 235), (150, 150, 150), (24, 22, 20)

STATE_COLORS = {
    TrackState.TRACKING: (80, 210, 80),     # green
    TrackState.COASTING: (0, 200, 255),     # amber
    TrackState.LOST: (60, 60, 235),         # red
    TrackState.SEARCHING: (170, 170, 170),  # grey
}
POSITION_COLORS = {
    HorizontalPosition.LEFT: (230, 170, 40), HorizontalPosition.CENTER: (80, 210, 80),
    HorizontalPosition.RIGHT: (40, 160, 255), HorizontalPosition.UNKNOWN: GREY,
}
ZONE_COLORS = {
    DistanceZone.FAR: (230, 170, 40), DistanceZone.IDEAL: (80, 210, 80),
    DistanceZone.CLOSE: (200, 90, 200), DistanceZone.UNKNOWN: GREY,
}


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _dashed_vline(img, x, color, dash=8):
    h = img.shape[0]
    for y in range(36, h, dash * 2):
        cv2.line(img, (x, y), (x, min(h, y + dash)), color, 1)


def _dir(deg: float) -> str:
    """12.3 -> '12 deg R' (+ = right)."""
    if abs(deg) < 0.5:
        return "0 deg"
    return f"{abs(deg):.0f} deg {'R' if deg > 0 else 'L'}"


def draw_command_glyph(img, cmd: RobotCommand, center, size=34):
    """Arrow (or stop sign) icon for a command."""
    cx, cy = center
    color = COMMAND_COLORS[cmd]
    cv2.circle(img, (cx, cy), size, color, 2, cv2.LINE_AA)
    if cmd == RobotCommand.STOP:
        pts = []
        for i in range(8):
            a = np.pi / 8 + i * np.pi / 4
            pts.append((int(cx + 0.7 * size * np.cos(a)), int(cy + 0.7 * size * np.sin(a))))
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
        cv2.putText(img, "STOP", (cx - 19, cy + 5), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        return
    lin, ang = cmd.motion
    dx, dy = -ang, -lin            # screen: up = forward, left = turn left
    n = max(1e-6, np.hypot(dx, dy))
    dx, dy = dx / n, dy / n
    p0 = (int(cx - dx * size * 0.6), int(cy - dy * size * 0.6))
    p1 = (int(cx + dx * size * 0.7), int(cy + dy * size * 0.7))
    cv2.arrowedLine(img, p0, p1, color, 5, cv2.LINE_AA, tipLength=0.45)


class Dashboard:
    def __init__(self, cfg, mode_name: str = "NORMAL"):
        self.cfg = cfg
        self.mode_name = mode_name
        self.event_log: List[str] = []

    def log_event(self, msg: str) -> None:
        self.event_log.append(msg)
        self.event_log = self.event_log[-4:]

    # ----------------------------------------------------------- camera view
    def draw_camera(self, frame: np.ndarray, st, debug: bool = False) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        fcx = w // 2
        hfov = self.cfg.CAMERA_HFOV_DEG
        hd = st.heading
        aim = hd.aim if hd else 0.0
        aim_x = int(bearing_to_x(aim, w, hfov))
        thr = np.degrees(np.arctan(self.cfg.HORIZONTAL_THRESHOLD * np.tan(np.radians(hfov / 2))))

        # left / centre / right regions, measured from where the robot FACES
        for b in (aim - thr, aim + thr):
            x = int(bearing_to_x(b, w, hfov))
            if 0 <= x < w:
                _dashed_vline(img, x, (200, 200, 200))
        _dashed_vline(img, fcx, (120, 120, 120), 4)        # camera centre (fixed)
        if 0 <= aim_x < w:
            cv2.line(img, (aim_x, 36), (aim_x, h), (255, 220, 0), 2)
            _text(img, "ROBOT FACING", (min(max(aim_x - 50, 4), w - 110), h - 30), 0.42, (255, 220, 0))
            if hd and abs(hd.turn_rate) > 0.1:            # robot is turning: show which way
                d = 1 if hd.turn_rate > 0 else -1
                cv2.arrowedLine(img, (aim_x, h - 52), (aim_x + d * 40, h - 52), (255, 220, 0), 3,
                                cv2.LINE_AA, tipLength=0.4)
        else:
            side = "<<" if aim_x < 0 else ">>"
            _text(img, f"{side} ROBOT FACING OFF-SCREEN", (8 if aim_x < 0 else w - 250, h - 30),
                  0.42, (255, 220, 0))
        _text(img, "LEFT", (8, h - 10), 0.45, GREY)
        _text(img, "CENTER", (fcx - 28, h - 10), 0.45, GREY)
        _text(img, "RIGHT", (w - 55, h - 10), 0.45, GREY)

        # all detections (thin), useful when several people are present
        if debug or len(st.detections) > 1:
            for d in st.detections:
                x1, y1, x2, y2 = d.as_int()
                cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
                _text(img, f"person {d.confidence:.2f}", (x1 + 2, y2 - 4), 0.4, (190, 190, 190))

        tr = st.track
        if tr.has_target:
            col = STATE_COLORS[tr.state]
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            label = f"TARGET  {tr.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), col, -1)
            cv2.putText(img, label, (x1 + 4, y1 - 6), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            if st.position:
                cx, cy = int(st.position.cx), int(st.position.cy)
                cv2.circle(img, (cx, cy), 7, col, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 9, (0, 0, 0), 2, cv2.LINE_AA)
                # horizontal error vector from where the robot faces to the target centre
                ax = int(min(max(st.position.aim_x if st.position.aim_x is not None else fcx, 0), w - 1))
                cv2.arrowedLine(img, (ax, cy), (cx, cy), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.08)
                _text(img, f"error = {st.position.bearing_deg:+.0f} deg",
                      ((cx + ax) // 2 - 50, cy + 24), 0.45, (0, 230, 255))
                if st.target_motion in ("LEFT", "RIGHT"):   # which way the person is walking
                    d = -1 if st.target_motion == "LEFT" else 1
                    cv2.arrowedLine(img, (cx, cy - 40), (cx + d * 70, cy - 40), (255, 0, 255), 4,
                                    cv2.LINE_AA, tipLength=0.35)
            if debug and tr.raw_box is not None:
                rx1, ry1, rx2, ry2 = [int(v) for v in tr.raw_box]
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 0, 255), 1)
        # frame centre marker
        cv2.drawMarker(img, (fcx, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)

        # status banner
        cmd = st.decision.command
        banner_col = COMMAND_COLORS[cmd]
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 32), (w, 35), banner_col, -1)
        _text(img, f"COMMAND: {cmd.label}", (10, 24), 0.7, banner_col, 2)
        state_col = STATE_COLORS[tr.state]
        _text(img, tr.state.value, (w - 120, 24), 0.6, state_col, 2)

        # what the person did (their own left/right: the camera faces them, so the
        # robot's LEFT is the person's RIGHT) and what happened to the laptop
        y = 62
        if st.target_motion in ("LEFT", "RIGHT"):
            yours = "RIGHT" if st.target_motion == "LEFT" else "LEFT"
            _text(img, f"YOU MOVED TO YOUR {yours}  (robot's {st.target_motion.lower()})",
                  (10, y), 0.6, (255, 0, 255), 2)
            y += 26
        if hd and hd.laptop_moving:
            side = "RIGHT" if hd.laptop_rate > 0 else "LEFT"
            _text(img, f"LAPTOP MOVING ({side}, {abs(hd.laptop_rate):.0f} deg/s)", (10, y), 0.6,
                  (0, 200, 255), 2)
        elif hd and hd.bump_deg is not None:
            side = "RIGHT" if hd.bump_deg > 0 else "LEFT"
            _text(img, f"LAPTOP TURNED {abs(hd.bump_deg):.0f} deg {side} - ROBOT CORRECTING",
                  (10, y), 0.6, (0, 200, 255), 2)

        if tr.state in (TrackState.LOST, TrackState.SEARCHING):
            msg = "TARGET LOST - ROBOT STOPPED" if tr.state == TrackState.LOST else "SEARCHING FOR A PERSON..."
            (tw, _), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
            _text(img, msg, ((w - tw) // 2, h // 2 + 40), 0.8, state_col, 2)
        return img

    # ------------------------------------------------------------ side panel
    def draw_panel(self, st, height: int, fps: float, detector_name: str,
                   segmentation_method: str, debug: bool = False) -> np.ndarray:
        W = self.cfg.PANEL_WIDTH
        p = np.full((height, W, 3), DARK, np.uint8)
        y = 26

        def row(label, value, color=WHITE, big=False):
            nonlocal y
            _text(p, label, (12, y), 0.42, GREY)
            _text(p, value, (112, y), 0.6 if big else 0.48, color, 2 if big else 1)
            y += 25 if big else 20

        _text(p, f"{self.mode_name} MODE", (12, y), 0.55, (0, 200, 255), 2); y += 26
        tr = st.track
        row("TARGET", ("PERSON  " + tr.state.value) if tr.has_target else tr.state.value,
            STATE_COLORS[tr.state], big=True)
        row("CONFIDENCE", f"{tr.confidence:.2f}" if tr.has_target else "-",
            (80, 210, 80) if tr.confidence >= self.cfg.CONFIDENCE_THRESHOLD else (60, 60, 235))
        pos = st.position
        row("POSITION", pos.position.value if pos else "-",
            POSITION_COLORS[pos.position] if pos else GREY, big=True)
        row("ERROR", f"{_dir(pos.bearing_deg)}  ({pos.error_px:+.0f}px)" if pos else "-")
        hd = st.heading
        if hd:
            row("ROBOT HEAD", _dir(hd.robot_yaw) + (f"  turning {_dir(hd.turn_rate)}/s"
                                                    if abs(hd.turn_rate) > 0.1 else ""))
            row("LAPTOP", _dir(hd.cam_yaw) + ("  MOVING" if hd.laptop_moving else ""),
                (0, 200, 255) if hd.laptop_moving else WHITE)
        dist = st.distance
        row("DISTANCE", dist.zone.value if dist else "-", ZONE_COLORS[dist.zone] if dist else GREY, big=True)
        approx = ""
        if dist and self.cfg.SHOW_APPROX_METRIC:
            approx = f"  approx {dist.approx_text()}"
        row("BOX HEIGHT", f"{dist.smoothed_ratio:.2f}{approx}" if dist else "-")

        # distance proxy bar with the four thresholds
        if dist:
            bx, bw, by = 12, W - 24, y - 6
            cv2.rectangle(p, (bx, by), (bx + bw, by + 10), (70, 70, 70), -1)
            c = self.cfg
            for thr, col in ((c.TOO_FAR_HEIGHT, ZONE_COLORS[DistanceZone.FAR]),
                             (c.IDEAL_MIN_HEIGHT, ZONE_COLORS[DistanceZone.IDEAL]),
                             (c.IDEAL_MAX_HEIGHT, ZONE_COLORS[DistanceZone.IDEAL]),
                             (c.TOO_CLOSE_HEIGHT, ZONE_COLORS[DistanceZone.CLOSE])):
                tx = bx + int(thr * bw)
                cv2.line(p, (tx, by - 3), (tx, by + 13), col, 2)
            mx = bx + int(min(1.0, dist.smoothed_ratio) * bw)
            cv2.circle(p, (mx, by + 5), 6, WHITE, -1, cv2.LINE_AA)
            _text(p, "far", (bx, by + 24), 0.35, GREY)
            _text(p, "close", (bx + bw - 30, by + 24), 0.35, GREY)
            y += 30

        # command
        cmd = st.decision.command
        y += 6
        cv2.line(p, (12, y - 14), (W - 12, y - 14), (70, 70, 70), 1)
        _text(p, "ROBOT COMMAND", (12, y), 0.45, GREY); y += 8
        draw_command_glyph(p, cmd, (50, y + 38), 32)
        short = cmd.label.replace("MOVE ", "")
        _text(p, short, (95, y + 36), 0.65 if len(short) <= 12 else 0.48, COMMAND_COLORS[cmd], 2)
        if st.decision.raw_command != cmd:
            _text(p, f"pending: {st.decision.raw_command.value}", (95, y + 58), 0.4, GREY)
        y += 82
        reason = st.decision.reason
        for i in range(0, min(len(reason), 92), 46):
            _text(p, reason[i:i + 46], (12, y), 0.38, (200, 200, 200)); y += 16
        y += 10

        cv2.line(p, (12, y - 10), (W - 12, y - 10), (70, 70, 70), 1)
        row("FPS", f"{fps:.1f}", (80, 210, 80) if fps >= 15 else (0, 200, 255))
        row("DETECTOR", detector_name)
        if debug:
            row("DETECT", f"{st.detect_ms:.1f} ms" + ("" if st.detector_ran else " (skipped)"))
            row("TRACK", f"{st.track_ms:.2f} ms  src={st.track.source}")
            row("PIPELINE", f"{st.total_ms:.1f} ms")
            row("PEOPLE", str(len(st.detections)))
        if segmentation_method != "off":
            seg = st.segmentation
            txt = "-" if seg is None else (
                "BLOCKED" if seg.path_blocked else ("clear" if seg.ground_visible else "no ground view"))
            row("TERRAIN", f"{segmentation_method}: {txt}",
                (60, 60, 235) if seg and seg.path_blocked else WHITE)

        # recent events (only as many as fit below the content)
        free_lines = max(0, (height - y - 8) // 16)
        events = self.event_log[-free_lines:] if free_lines else []
        y = height - 10 - 16 * (len(events) - 1)
        for e in events:
            _text(p, e[:46], (12, y), 0.36, (180, 180, 120)); y += 16
        return p

    # --------------------------------------------------------------- compose
    @staticmethod
    def compose(parts: List[Optional[np.ndarray]]) -> np.ndarray:
        parts = [x for x in parts if x is not None]
        H = max(x.shape[0] for x in parts)
        fixed = []
        for x in parts:
            if x.shape[0] != H:
                pad = np.full((H - x.shape[0], x.shape[1], 3), DARK, np.uint8)
                x = np.vstack([x, pad])
            fixed.append(x)
        return np.hstack(fixed)

    @staticmethod
    def help_overlay(img: np.ndarray, lines: List[str]) -> None:
        h, w = img.shape[:2]
        bw, bh = 380, 22 * len(lines) + 20
        x0, y0 = (w - bw) // 2, (h - bh) // 2
        sub = img[y0:y0 + bh, x0:x0 + bw]
        img[y0:y0 + bh, x0:x0 + bw] = (sub * 0.25).astype(np.uint8)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (0, 200, 255), 1)
        for i, line in enumerate(lines):
            _text(img, line, (x0 + 14, y0 + 26 + 22 * i), 0.5, WHITE)
