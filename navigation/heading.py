"""
Closed-loop robot heading, laptop rotation and target-motion read-out.

The laptop webcam cannot turn, so the robot keeps a VIRTUAL heading. All angles
are degrees, + = right (same sign as PositionInfo.bearing_deg):

    cam_yaw    physical rotation of the laptop since calibration, measured from
               background optical flow (tracking/ego_motion.py)
    robot_yaw  where the robot body points. The laptop IS the robot, so rotating
               the laptop rotates the robot by the same angle; turn commands
               rotate it further (virtually, at a speed proportional to the error)
    aim        = robot_yaw - cam_yaw: the robot's forward direction inside the
               current camera image

    target bearing relative to the robot = camera bearing - aim
                                         -> drives LEFT / CENTER / RIGHT

When the robot turns toward you, `aim` moves onto you, the relative bearing goes
to 0 and the command becomes STOP - the loop closes as it would on a real robot.
If someone rotates the laptop, robot_yaw jumps by the measured angle, you are no
longer in front of the robot, and it turns back to face you.

Turning speed is proportional to the error (TURN_GAIN, clipped between
MIN_TURN_SPEED_DEG and SIM_ANGULAR_SPEED_DEG): a constant-speed turn would
overshoot during the debouncer's minimum hold time and then oscillate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from navigation.decision_engine import RobotCommand

AIM_LIMIT_DEG = 89.0     # keep tan() finite
BUMP_SHOW_S = 3.0        # how long "LAPTOP TURNED ..." stays on screen


@dataclass
class HeadingState:
    robot_yaw: float
    cam_yaw: float
    aim: float              # robot forward direction in the current image (deg)
    turn_rate: float        # deg/s, + = turning right
    laptop_rate: float      # deg/s, + = laptop turning right
    laptop_moving: bool
    bump_deg: Optional[float]   # last completed laptop rotation (shown for a few s)


class RobotHeading:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Calibrate: the robot faces straight down the camera axis."""
        self.robot_yaw = 0.0
        self.cam_yaw = 0.0
        self.turn_rate = 0.0
        self.laptop_rate = 0.0
        self.laptop_moving = False
        self._bump_start = 0.0
        self._bump: Optional[float] = None
        self._bump_t = -1e9
        self._last_now: Optional[float] = None

    @property
    def aim(self) -> float:
        if not self.cfg.CLOSED_LOOP_HEADING:
            return 0.0
        return max(-AIM_LIMIT_DEG, min(AIM_LIMIT_DEG, self.robot_yaw - self.cam_yaw))

    def apply_camera_rotation(self, yaw_deg: float, dt: float, now: float) -> Optional[str]:
        """Laptop rotated by yaw_deg this frame. Returns an event when a rotation ends."""
        self.cam_yaw += yaw_deg
        self.robot_yaw += yaw_deg
        rate = yaw_deg / dt if dt > 0 else 0.0
        self.laptop_rate = 0.5 * rate + 0.5 * self.laptop_rate
        moving = abs(self.laptop_rate) > self.cfg.EGO_MOVING_DEG_S
        event = None
        if moving and not self.laptop_moving:
            self._bump_start = self.cam_yaw - yaw_deg
        elif self.laptop_moving and not moving:
            bump = self.cam_yaw - self._bump_start
            if abs(bump) >= 2.0:
                self._bump, self._bump_t = bump, now
                side = "right" if bump > 0 else "left"
                event = f"laptop turned {abs(bump):.0f} deg {side} - robot correcting"
        self.laptop_moving = moving
        return event

    def turn(self, command: RobotCommand, rel_bearing: Optional[float], dt: float) -> None:
        """Integrate the commanded turn (closed loop only)."""
        _, ang = command.motion          # ang +1 = turn left
        if not self.cfg.CLOSED_LOOP_HEADING or ang == 0:
            self.turn_rate = 0.0
            return
        speed = self.cfg.SIM_ANGULAR_SPEED_DEG
        if rel_bearing is not None:
            speed = min(speed, max(self.cfg.MIN_TURN_SPEED_DEG, self.cfg.TURN_GAIN * abs(rel_bearing)))
        self.turn_rate = -ang * speed
        self.robot_yaw += self.turn_rate * dt

    def state(self, now: float) -> HeadingState:
        bump = self._bump if now - self._bump_t < BUMP_SHOW_S else None
        return HeadingState(self.robot_yaw, self.cam_yaw, self.aim, self.turn_rate,
                            self.laptop_rate, self.laptop_moving, bump)


class TargetMotion:
    """Is the target walking left or right? Uses the target's bearing in the
    WORLD (camera bearing + laptop rotation), so turning the laptop does not
    look like the person moving.  Returns "LEFT" / "RIGHT" / "STILL" / ""."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[float] = None
        self.rate = 0.0
        self.label = ""

    def update(self, world_bearing: Optional[float], dt: float) -> str:
        if world_bearing is None:
            self.reset()
            return ""
        if self._prev is not None and dt > 0:
            self.rate = 0.3 * (world_bearing - self._prev) / dt + 0.7 * self.rate
        self._prev = world_bearing
        T = self.cfg.PERSON_MOVING_DEG_S
        if self.label in ("LEFT", "RIGHT"):
            # hysteresis: keep the label until the motion has clearly stopped
            if abs(self.rate) < T / 2:
                self.label = "STILL"
            else:
                self.label = "RIGHT" if self.rate > 0 else "LEFT"
        else:
            self.label = ("RIGHT" if self.rate > T else "LEFT" if self.rate < -T else "STILL")
        return self.label


def bearing_to_x(bearing_deg: float, frame_width: int, hfov_deg: float) -> float:
    """Image column of a direction (deg, + = right) for a pinhole camera."""
    fcx = frame_width / 2.0
    b = max(-AIM_LIMIT_DEG, min(AIM_LIMIT_DEG, bearing_deg))
    return fcx + fcx * math.tan(math.radians(b)) / math.tan(math.radians(hfov_deg / 2.0))
