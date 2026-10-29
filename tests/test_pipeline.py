"""Integration test: real ML detector + full pipeline on sample images/video (no webcam).

Skipped automatically if the detector packages are not installed.
"""

import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from detection.person_detector import DetectorUnavailableError, create_detector  # noqa: E402

SAMPLE_IMG = ROOT / "data" / "samples" / "test_frame.jpg"
SAMPLE_VID = ROOT / "data" / "samples" / "test_walk.mp4"


class TestDetectorAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.detector = create_detector(Config())
        except DetectorUnavailableError as exc:
            raise unittest.SkipTest(f"detector unavailable: {exc}")

    def test_detects_person_in_sample(self):
        if not SAMPLE_IMG.exists():
            self.skipTest("run tools/make_test_video.py first")
        img = cv2.imread(str(SAMPLE_IMG))
        dets = self.detector.detect(img)
        self.assertGreaterEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.class_name, "person")
        self.assertGreater(d.confidence, 0.45)
        self.assertTrue(d.is_valid(img.shape[1], img.shape[0]))
        self.assertGreater(self.detector.last_latency_ms, 0)

    def test_no_person_in_blank_frame(self):
        import numpy as np
        self.assertEqual(self.detector.detect(np.full((480, 640, 3), 127, np.uint8)), [])

    def test_pipeline_on_video(self):
        if not SAMPLE_VID.exists():
            self.skipTest("run tools/make_test_video.py first")
        from pipeline import FollowerPipeline
        from navigation.decision_engine import RobotCommand
        pipe = FollowerPipeline(Config(), detector=self.detector, segmentation="kmeans")
        cap = cv2.VideoCapture(str(SAMPLE_VID))
        commands = set()
        for i in range(80):
            ok, frame = cap.read()
            if not ok:
                break
            st = pipe.process(frame, now=i / 20.0)
            commands.add(st.decision.command)
        cap.release()
        self.assertIn(RobotCommand.FORWARD, commands)     # far + centred at the start
        self.assertTrue(st.target_visible)


if __name__ == "__main__":
    unittest.main()
