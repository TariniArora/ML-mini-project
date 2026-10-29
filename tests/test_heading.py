"""Unit + integration tests: closed-loop heading, laptop ego-motion, target motion.

No camera or ML model needed: a fake detector reports a box and a synthetic,
textured "room" is cropped to imitate the laptop camera turning.
"""

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from detection.person_detector import Detection  # noqa: E402
from navigation.decision_engine import RobotCommand  # noqa: E402
from navigation.heading import RobotHeading, TargetMotion  # noqa: E402
from navigation.position_estimator import HorizontalPosition, PositionEstimator  # noqa: E402
from pipeline import FollowerPipeline  # noqa: E402
from simulation.robot_simulator import RobotSimulator  # noqa: E402
from tracking.ego_motion import CameraMotionEstimator  # noqa: E402

W, H, FPS = 640, 480, 30.0
ROOM_W = 1600   # wide panorama the camera "looks" into


def f_px(cfg):
    return (W / 2.0) / math.tan(math.radians(cfg.CAMERA_HFOV_DEG / 2.0))


def make_room(seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (H // 8, ROOM_W // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(noise, (ROOM_W, H), interpolation=cv2.INTER_CUBIC), (5, 5), 0)


class FakeDetector:
    """Returns whatever box the test puts in `self.box`."""
    name = "fake"
    last_latency_ms = 0.1

    def __init__(self):
        self.box = None

    def detect(self, frame):
        if self.box is None:
            return []
        return [Detection(*self.box, confidence=0.9)]


class Scene:
    """A person standing in a room, seen by a laptop camera that can be rotated.
    cam_offset (px into the room) grows when the laptop is turned right."""

    def __init__(self, cfg, person_x, person_h=0.65):
        self.cfg = cfg
        self.room = make_room()
        self.person_x = person_x          # person centre, room pixels
        self.person_h = person_h
        self.cam_offset = (ROOM_W - W) // 2
        self.det = FakeDetector()

    def frame(self):
        img = self.room[:, self.cam_offset:self.cam_offset + W].copy()
        cx = self.person_x - self.cam_offset
        bh = self.person_h * H
        box = (cx - 60, H / 2 - bh / 2, cx + 60, H / 2 + bh / 2)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (40, 40, 200), -1)
        self.det.box = box if 0 < cx < W else None
        return img

    def run(self, pipe, seconds, t0=0.0, on_frame=None):
        states = []
        for i in range(int(seconds * FPS)):
            if on_frame:
                on_frame(i)
            states.append(pipe.process(self.frame(), now=t0 + i / FPS))
        return states


def ideal_cfg():
    cfg = Config()
    cfg.TOO_FAR_HEIGHT, cfg.IDEAL_MIN_HEIGHT, cfg.IDEAL_MAX_HEIGHT, cfg.TOO_CLOSE_HEIGHT = 0.5, 0.55, 0.8, 0.9
    return cfg


class TestPositionWithAim(unittest.TestCase):
    def test_zero_aim_is_frame_centre(self):
        cfg = Config(); cfg.SMOOTHING_FACTOR = 1.0
        pe = PositionEstimator(cfg)
        info = pe.update((100, 100, 200, 400), W)          # cx = 150, clearly left
        self.assertEqual(info.position, HorizontalPosition.LEFT)
        self.assertAlmostEqual(info.error_px, 150 - W / 2)
        self.assertAlmostEqual(info.aim_x, W / 2)

    def test_aim_on_target_is_center(self):
        cfg = Config(); cfg.SMOOTHING_FACTOR = 1.0
        pe = PositionEstimator(cfg)
        cam_bearing = pe.update((100, 100, 200, 400), W).bearing_deg
        pe.reset()
        info = pe.update((100, 100, 200, 400), W, aim_deg=cam_bearing)
        self.assertEqual(info.position, HorizontalPosition.CENTER)
        self.assertAlmostEqual(info.bearing_deg, 0.0, places=6)
        self.assertAlmostEqual(info.aim_x, 150, places=3)


class TestClosedLoop(unittest.TestCase):
    def test_robot_turns_toward_person_then_stops(self):
        cfg = ideal_cfg()
        scene = Scene(cfg, person_x=ROOM_W // 2 - 180)          # ~19 deg to the camera's left
        pipe = FollowerPipeline(cfg, detector=scene.det)
        states = scene.run(pipe, 4.0)
        cmds = [s.decision.command for s in states]
        self.assertIn(RobotCommand.LEFT, cmds)                   # it turned toward the person ...
        self.assertEqual(cmds[-1], RobotCommand.STOP)            # ... and stopped facing them
        self.assertEqual(states[-1].position.position, HorizontalPosition.CENTER)
        self.assertLess(states[-1].heading.robot_yaw, -10)       # robot now points left
        self.assertNotIn(RobotCommand.RIGHT, cmds)               # no overshoot / oscillation
        self.assertLessEqual(pipe.decision.debouncer.switches, 2)

    def test_open_loop_keeps_turning(self):
        cfg = ideal_cfg(); cfg.CLOSED_LOOP_HEADING = False
        scene = Scene(cfg, person_x=ROOM_W // 2 - 180)
        pipe = FollowerPipeline(cfg, detector=scene.det)
        states = scene.run(pipe, 3.0)
        self.assertEqual(states[-1].decision.command, RobotCommand.LEFT)

    def test_laptop_rotation_is_measured_and_corrected(self):
        cfg = ideal_cfg()
        scene = Scene(cfg, person_x=ROOM_W // 2)                 # straight ahead
        pipe = FollowerPipeline(cfg, detector=scene.det)
        before = scene.run(pipe, 1.0)
        self.assertEqual(before[-1].decision.command, RobotCommand.STOP)

        # someone turns the laptop to the RIGHT by ~15 deg over half a second
        step_px = 10
        n = int(round(f_px(cfg) * math.tan(math.radians(15)) / step_px))

        def turn(i):
            if i < n:
                scene.cam_offset += step_px
        after = scene.run(pipe, 4.0, t0=1.0, on_frame=turn)
        turned = math.degrees(math.atan(n * step_px / f_px(cfg)))
        self.assertAlmostEqual(after[-1].heading.cam_yaw, turned, delta=2.0)
        self.assertTrue(any(s.heading.laptop_moving for s in after))
        self.assertTrue(any("laptop turned" in e for s in after for e in s.events))
        # the person did not move, so the motion read-out must not claim they did
        self.assertFalse(any(s.target_motion in ("LEFT", "RIGHT") for s in after[n + 5:]))
        # robot was knocked right with the laptop, turned back LEFT, and faces the person again
        self.assertIn(RobotCommand.LEFT, [s.decision.command for s in after])
        self.assertEqual(after[-1].decision.command, RobotCommand.STOP)
        self.assertAlmostEqual(after[-1].heading.robot_yaw, 0.0, delta=6.0)
        self.assertTrue(after[-1].track.has_target)

    def test_calibrate_resets_heading(self):
        cfg = ideal_cfg()
        scene = Scene(cfg, person_x=ROOM_W // 2 - 180)
        pipe = FollowerPipeline(cfg, detector=scene.det)
        st = scene.run(pipe, 3.0)[-1]
        self.assertNotAlmostEqual(pipe.heading.robot_yaw, 0.0, delta=5)
        pipe.calibrate(st)
        self.assertEqual((pipe.heading.robot_yaw, pipe.heading.cam_yaw), (0.0, 0.0))


class TestEgoMotion(unittest.TestCase):
    def test_static_camera_has_no_rotation(self):
        cfg = Config()
        room = make_room()
        ego = CameraMotionEstimator(cfg)
        for _ in range(10):
            r = ego.update(room[:, 400:400 + W])
        self.assertTrue(r.ok)
        self.assertEqual(r.yaw_deg, 0.0)

    def test_pan_sign_and_size(self):
        cfg = Config()
        room = make_room()
        ego = CameraMotionEstimator(cfg)
        ego.update(room[:, 400:400 + W])
        r = ego.update(room[:, 412:412 + W])        # camera turned right: scene slides left 12 px
        self.assertTrue(r.ok)
        self.assertAlmostEqual(r.dx_px, -12.0, delta=0.5)
        self.assertAlmostEqual(r.yaw_deg, math.degrees(math.atan(12 / f_px(cfg))), delta=0.1)


class TestTargetMotion(unittest.TestCase):
    def test_walking_and_still(self):
        tm = TargetMotion(Config())
        for i in range(30):
            label = tm.update(i * 0.5, 1 / FPS)        # 15 deg/s to the right
        self.assertEqual(label, "RIGHT")
        for _ in range(30):
            label = tm.update(15.0, 1 / FPS)
        self.assertEqual(label, "STILL")
        self.assertEqual(tm.update(None, 1 / FPS), "")


class TestHeadingAndSim(unittest.TestCase):
    def test_turn_speed_proportional(self):
        cfg = Config()
        hd = RobotHeading(cfg)
        hd.turn(RobotCommand.LEFT, -2.0, 0.1)
        self.assertAlmostEqual(hd.turn_rate, -cfg.MIN_TURN_SPEED_DEG)
        hd.turn(RobotCommand.RIGHT, 100.0, 0.1)
        self.assertAlmostEqual(hd.turn_rate, cfg.SIM_ANGULAR_SPEED_DEG)

    def test_sim_follows_heading(self):
        s = RobotSimulator(Config())
        s.step(RobotCommand.RIGHT, 0.1, None, heading_deg=20.0)
        self.assertAlmostEqual(s.heading_deg, -20.0)      # sim: + = left, heading: + = right
        self.assertLess(s.omega, 0)


if __name__ == "__main__":
    unittest.main()
