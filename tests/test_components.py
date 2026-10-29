"""Unit tests: simulator, target selection, tracker and segmentation (synthetic inputs)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from detection.person_detector import Detection  # noqa: E402
from detection.target_selector import TargetSelector  # noqa: E402
from navigation.decision_engine import RobotCommand  # noqa: E402
from simulation.robot_simulator import RobotSimulator  # noqa: E402
from tracking.tracker import TargetTracker, TrackState  # noqa: E402

W, H = 640, 480


def frame_with_person(box, color=(40, 40, 200)):
    img = np.full((H, W, 3), 128, np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box]
    img[y1:y2, x1:x2] = color
    return img


class TestSimulator(unittest.TestCase):
    def test_forward_moves_along_heading(self):
        s = RobotSimulator(Config())
        for _ in range(10):
            s.step(RobotCommand.FORWARD, 0.1)
        self.assertAlmostEqual(s.x, 0.0, places=6)
        self.assertAlmostEqual(s.y, 0.6, places=6)          # 0.6 m/s * 1 s, facing +y
        self.assertAlmostEqual(s.odometer, 0.6, places=6)

    def test_backward(self):
        s = RobotSimulator(Config())
        s.step(RobotCommand.BACKWARD, 0.2)
        self.assertLess(s.y, 0)

    def test_left_turns_ccw_right_turns_cw(self):
        s = RobotSimulator(Config())
        s.step(RobotCommand.LEFT, 0.2)
        self.assertGreater(s.heading_deg, 0)
        s.reset()
        s.step(RobotCommand.RIGHT, 0.2)
        self.assertLess(s.heading_deg, 0)
        self.assertEqual((s.x, s.y), (0.0, 0.0))            # turn in place

    def test_stop_does_not_move(self):
        s = RobotSimulator(Config())
        s.step(RobotCommand.STOP, 0.2)
        self.assertEqual((s.x, s.y, s.odometer), (0.0, 0.0, 0.0))

    def test_render(self):
        s = RobotSimulator(Config())
        s.step(RobotCommand.FORWARD_LEFT, 0.1, (2.0, math.radians(10)))
        img = s.render(True)
        self.assertEqual(img.shape, (Config().SIM_HEIGHT, Config().SIM_WIDTH, 3))


class TestTargetSelection(unittest.TestCase):
    def test_single_person(self):
        sel = TargetSelector(Config())
        d = Detection(10, 10, 100, 300, 0.9)
        self.assertIs(sel.acquire([d], (H, W))[0], d)

    def test_closest_to_center_when_multiple(self):
        sel = TargetSelector(Config())
        a = Detection(10, 100, 110, 400, 0.95)      # left, more confident
        b = Detection(280, 100, 380, 400, 0.70)     # centre
        self.assertIs(sel.acquire([a, b], (H, W))[0], b)

    def test_low_confidence_ignored(self):
        sel = TargetSelector(Config())
        self.assertIsNone(sel.acquire([Detection(280, 100, 380, 400, 0.2)], (H, W))[0])

    def test_match_prefers_previous_position(self):
        sel = TargetSelector(Config())
        prev = (100, 100, 200, 400)
        near = Detection(110, 100, 210, 400, 0.6)
        other = Detection(400, 100, 500, 400, 0.99)
        self.assertIs(sel.match([other, near], prev, (H, W)), near)

    def test_match_rejects_far_jump(self):
        sel = TargetSelector(Config())
        self.assertIsNone(sel.match([Detection(500, 100, 600, 400, 0.99)], (0, 100, 100, 400), (H, W)))


class TestTracker(unittest.TestCase):
    def test_acquire_track_coast_lose(self):
        cfg = Config()
        trk = TargetTracker(cfg)
        box = (200, 100, 300, 400)
        f = frame_with_person(box)
        r = trk.step(f, [Detection(*box, 0.9)])
        self.assertEqual(r.state, TrackState.TRACKING)
        for i in range(5):  # person moves right slowly
            b = (200 + 5 * i, 100, 300 + 5 * i, 400)
            r = trk.step(frame_with_person(b), [Detection(*b, 0.9)])
        self.assertEqual(r.state, TrackState.TRACKING)
        r = trk.step(f, [])
        self.assertEqual(r.state, TrackState.COASTING)
        self.assertLess(r.confidence, 0.9)
        for _ in range(cfg.LOST_TIMEOUT_FRAMES + 1):
            r = trk.step(f, [])
        self.assertEqual(r.state, TrackState.LOST)
        self.assertIsNone(r.box)

    def test_no_switch_to_other_person(self):
        trk = TargetTracker(Config())
        a = (100, 100, 200, 400)
        b = (450, 100, 550, 400)
        f = frame_with_person(a)
        trk.step(f, [Detection(*a, 0.8)])
        for _ in range(10):
            r = trk.step(f, [Detection(*b, 0.99), Detection(*a, 0.6)])
        self.assertLess(abs((r.box[0] + r.box[2]) / 2 - 150), 15)

    def test_invalid_detection_rejected(self):
        trk = TargetTracker(Config())
        r = trk.step(frame_with_person((0, 0, 1, 1)), [Detection(float("nan"), 0, 10, 10, 0.9)])
        self.assertEqual(r.state, TrackState.SEARCHING)


class TestSegmentation(unittest.TestCase):
    def scene(self):
        img = np.zeros((H, W, 3), np.uint8)
        img[:250] = (200, 200, 220)       # wall
        img[250:] = (60, 90, 130)         # floor
        return img

    def test_kmeans_clear_and_blocked(self):
        from segmentation.kmeans_segmentation import KMeansSegmenter
        cfg = Config()
        seg = KMeansSegmenter(cfg)
        r = seg.segment(self.scene())
        self.assertTrue(r.ground_visible)
        self.assertFalse(r.path_blocked)
        self.assertGreater(r.free_ratio["center"], 0.5)
        blocked = self.scene()
        blocked[300:400, 200:440] = (10, 200, 10)   # green obstacle on the floor ahead
        seg2 = KMeansSegmenter(cfg)
        r2 = seg2.segment(blocked)
        self.assertTrue(r2.path_blocked)
        self.assertIsNotNone(r2.vis)

    def test_felzenszwalb_optional(self):
        from segmentation import felzenszwalb_segmentation as fz
        if not fz.AVAILABLE:
            self.skipTest("scikit-image not installed")
        r = fz.FelzenszwalbSegmenter(Config()).segment(self.scene())
        self.assertTrue(r.ground_visible)
        self.assertFalse(r.path_blocked)


if __name__ == "__main__":
    unittest.main()
