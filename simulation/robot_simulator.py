"""
2-D top-down differential-drive robot simulator.

It consumes EXACTLY the RobotCommand produced by the decision engine each frame
(no scripted animation). Kinematics (unicycle model):

    v     = linear  * SIM_LINEAR_SPEED          (m/s)
    omega = angular * SIM_ANGULAR_SPEED          (rad/s, + = turn left / CCW)
    x     += v * cos(theta) * dt
    y     += v * sin(theta) * dt
    theta += omega * dt

The view is centred on the robot and rotates with it (heading always up), with a
world-fixed grid and the driven trail, so turning and driving are both visible. The person the camera currently
sees is drawn where the robot PERCEIVES them: at the estimated bearing and the
rough monocular distance, relative to the robot's current pose.

Closed loop: main.py passes `heading_deg` from navigation/heading.py. That
heading already contains the commanded turns AND any measured rotation of the
laptop, so the simulated robot faces where the vision system thinks the robot
faces, and the perceived human swings to straight-ahead as the robot turns
toward them. Without `heading_deg` the commands are integrated open-loop.
Replace `step()` with motor/serial output to drive real hardware.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from navigation.decision_engine import RobotCommand

COMMAND_COLORS = {
    RobotCommand.STOP: (60, 60, 230),
    RobotCommand.FORWARD: (80, 200, 80),
    RobotCommand.BACKWARD: (200, 90, 200),
    RobotCommand.LEFT: (230, 170, 40),
    RobotCommand.RIGHT: (40, 160, 255),
    RobotCommand.FORWARD_LEFT: (170, 210, 60),
    RobotCommand.FORWARD_RIGHT: (60, 210, 200),
}


class RobotSimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.x, self.y = 0.0, 0.0
        self.theta = math.pi / 2          # facing "up" (north)
        self.v, self.omega = 0.0, 0.0
        self.odometer = 0.0
        self.command = RobotCommand.STOP
        self.trail = deque(maxlen=self.cfg.SIM_TRAIL_LENGTH)
        self.trail.append((self.x, self.y))
        self.target_rel: Optional[Tuple[float, float]] = None  # (distance m, bearing rad, + = right)
        self.command_counts = {c: 0 for c in RobotCommand}

    # ---------------------------------------------------------------- physics
    def step(self, command: RobotCommand, dt: float,
             target_rel: Optional[Tuple[float, float]] = None,
             heading_deg: Optional[float] = None) -> None:
        """heading_deg: closed-loop heading (deg, + = right of the start direction)."""
        dt = float(np.clip(dt, 0.0, 0.25))   # guard against pauses / lag spikes
        self.command = command
        self.command_counts[command] += 1
        lin, ang = command.motion
        self.v = lin * self.cfg.SIM_LINEAR_SPEED
        if heading_deg is None:
            self.omega = ang * math.radians(self.cfg.SIM_ANGULAR_SPEED_DEG)
            self.theta = (self.theta + self.omega * dt) % (2 * math.pi)
        else:
            new_theta = math.pi / 2 - math.radians(heading_deg)
            turned = (new_theta - self.theta + math.pi) % (2 * math.pi) - math.pi
            self.omega = turned / dt if dt > 0 else 0.0
            self.theta = new_theta % (2 * math.pi)
        dx = self.v * math.cos(self.theta) * dt
        dy = self.v * math.sin(self.theta) * dt
        self.x += dx
        self.y += dy
        self.odometer += math.hypot(dx, dy)
        if dx or dy:
            self.trail.append((self.x, self.y))
        self.target_rel = target_rel

    @property
    def heading_deg(self) -> float:
        """Compass-style heading: 0 = north (start direction), + = turned left."""
        return (math.degrees(self.theta) - 90.0 + 180.0) % 360.0 - 180.0

    # -------------------------------------------------------------- rendering
    # Robot-centred, HEADING-UP view (like a car's navigation screen): the robot is
    # always drawn pointing up, the world grid and the trail rotate around it when
    # it turns, and the perceived human always appears in front of it.
    def _to_px(self, wx: float, wy: float) -> Tuple[int, int]:
        W, H, s = self.cfg.SIM_WIDTH, self.cfg.SIM_HEIGHT, self.cfg.SIM_PIXELS_PER_METER
        dx, dy = wx - self.x, wy - self.y
        a = math.pi / 2 - self.theta                     # rotate world so heading points up
        rx = dx * math.cos(a) - dy * math.sin(a)
        ry = dx * math.sin(a) + dy * math.cos(a)
        return int(W / 2 + rx * s), int(H * 0.72 - ry * s)

    def render(self, target_visible: bool = True) -> np.ndarray:
        W, H, s = self.cfg.SIM_WIDTH, self.cfg.SIM_HEIGHT, self.cfg.SIM_PIXELS_PER_METER
        img = np.full((H, W, 3), (34, 30, 28), np.uint8)

        # world-fixed grid every 0.5 m (rotates with the view)
        step, reach = 0.5, (W + H) / s
        gx0 = math.floor((self.x - reach) / step) * step
        gy0 = math.floor((self.y - reach) / step) * step
        n = int(2 * reach / step) + 2
        for i in range(n):
            gx = gx0 + i * step
            col = (62, 58, 54) if abs(gx - round(gx)) < 1e-6 else (44, 41, 39)
            cv2.line(img, self._to_px(gx, self.y - reach), self._to_px(gx, self.y + reach), col, 1)
            gy = gy0 + i * step
            col = (62, 58, 54) if abs(gy - round(gy)) < 1e-6 else (44, 41, 39)
            cv2.line(img, self._to_px(self.x - reach, gy), self._to_px(self.x + reach, gy), col, 1)

        # start marker + trail
        cv2.drawMarker(img, self._to_px(0, 0), (120, 120, 120), cv2.MARKER_CROSS, 10, 1)
        if len(self.trail) > 1:
            pts = np.array([self._to_px(x, y) for x, y in self.trail], np.int32)
            cv2.polylines(img, [pts], False, (200, 150, 70), 2, cv2.LINE_AA)

        rx, ry = int(W / 2), int(H * 0.72)
        color = COMMAND_COLORS[self.command]

        # camera field-of-view wedge (always pointing up)
        half = math.radians(self.cfg.CAMERA_HFOV_DEG / 2)
        L = ry - 60
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array([(rx, ry), (int(rx - L * math.tan(half)), ry - L),
                                         (int(rx + L * math.tan(half)), ry - L)], np.int32)], (75, 75, 55))
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        # perceived target (bearing + rough distance from the vision system)
        if target_visible and self.target_rel is not None:
            d, bearing = self.target_rel
            max_d = (ry - 80) / s
            d = float(np.clip(d, 0.4, max_d))          # the metric estimate is rough anyway
            tp = (int(rx + d * math.sin(bearing) * s), int(ry - d * math.cos(bearing) * s))
            cv2.line(img, (rx, ry), tp, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.circle(img, tp, 14, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (tp[0], tp[1] - 4), 4, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.line(img, (tp[0], tp[1]), (tp[0], tp[1] + 8), (0, 200, 255), 2)
            cv2.putText(img, "HUMAN (target)", (tp[0] - 50, tp[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 200, 255), 1, cv2.LINE_AA)
        elif not target_visible:
            cv2.putText(img, "? target not visible", (rx - 78, ry - int(2.0 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1, cv2.LINE_AA)

        # robot body (always facing up)
        r = int(0.22 * s)
        for side in (-1, 1):  # wheels
            cv2.rectangle(img, (rx + side * r - 4, ry - 8), (rx + side * r + 4, ry + 8), (20, 20, 20), -1)
        cv2.circle(img, (rx, ry), r, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (rx, ry), r, color, 3, cv2.LINE_AA)
        lin, ang = self.command.motion
        if lin or ang:  # motion arrow in robot frame
            ex = int(rx - ang * 26 * (0.7 if lin else 1.0))
            ey = int(ry - lin * 30)
            cv2.arrowedLine(img, (rx, ry), (ex, ey), color, 3, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.circle(img, (rx, ry), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, "ROBOT", (rx - 22, ry + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)

        # compass: where "north" (the start direction) is relative to the robot
        cc = (W - 32, 86)
        cv2.circle(img, cc, 20, (90, 90, 90), 1, cv2.LINE_AA)
        a = math.pi / 2 - self.theta + math.pi / 2  # world north (+y) on screen
        nx, ny = int(cc[0] + 16 * math.cos(a)), int(cc[1] - 16 * math.sin(a))
        cv2.arrowedLine(img, cc, (nx, ny), (200, 200, 200), 2, cv2.LINE_AA, tipLength=0.4)
        cv2.putText(img, "start dir", (W - 66, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    (160, 160, 160), 1, cv2.LINE_AA)

        # header / footer
        cv2.rectangle(img, (0, 0), (W, 56), (20, 18, 16), -1)
        cv2.putText(img, "SIMULATED ROBOT (top view, heading up)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, self.command.label, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        foot = (f"v={self.v:+.2f} m/s  w={math.degrees(self.omega):+.0f} deg/s  "
                f"heading={self.heading_deg:+.0f}  odo={self.odometer:.1f} m")
        cv2.rectangle(img, (0, H - 22), (W, H), (20, 18, 16), -1)
        cv2.putText(img, foot, (8, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(img, (W - 10 - int(s), H - 32), (W - 10, H - 32), (200, 200, 200), 2)
        cv2.putText(img, "1 m", (W - 10 - int(s) // 2 - 10, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return img
