"""Unit tests: position, distance, smoothing and the decision engine (no camera/model needed).

Run all tests:   python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from navigation.decision_engine import DecisionEngine, DecisionInput, RobotCommand  # noqa: E402
from navigation.distance_estimator import DistanceEstimator, DistanceZone  # noqa: E402
from navigation.position_estimator import HorizontalPosition, PositionEstimator  # noqa: E402
from utils.smoothing import CommandDebouncer, ExponentialSmoother  # noqa: E402

W, H = 640, 480


def box_at(cx_ratio, h_ratio, w_px=100):
    cx, h = cx_ratio * W, h_ratio * H
    return (cx - w_px / 2, 240 - h / 2, cx + w_px / 2, 240 + h / 2)


def raw_cfg():
    cfg = Config()
    cfg.SMOOTHING_FACTOR = 1.0  # disable EMA so single calls are deterministic
    return cfg


class TestPosition(unittest.TestCase):
    def test_left_center_right(self):
        cfg = raw_cfg()
        for cx, expected in ((0.1, HorizontalPosition.LEFT), (0.5, HorizontalPosition.CENTER),
                             (0.9, HorizontalPosition.RIGHT)):
            pe = PositionEstimator(cfg)
            info = pe.update(box_at(cx, 0.5), W)
            self.assertEqual(info.position, expected, f"cx={cx}")

    def test_error_formula(self):
        pe = PositionEstimator(raw_cfg())
        info = pe.update((400, 100, 500, 300), W)   # cx = 450
        self.assertAlmostEqual(info.cx, 450)
        self.assertAlmostEqual(info.error_px, 450 - 320)
        self.assertAlmostEqual(info.error_norm, 130 / 320)

    def test_hysteresis(self):
        cfg = raw_cfg()  # T = 0.18, H = 0.06 -> leave LEFT only when e > -0.12
        pe = PositionEstimator(cfg)
        pe.update(box_at(0.5 - 0.20 / 2, 0.5), W)          # e = -0.20 -> LEFT
        self.assertEqual(pe.position, HorizontalPosition.LEFT)
        pe.update(box_at(0.5 - 0.15 / 2, 0.5), W)          # e = -0.15: inside band -> stays LEFT
        self.assertEqual(pe.position, HorizontalPosition.LEFT)
        pe.update(box_at(0.5 - 0.10 / 2, 0.5), W)          # e = -0.10 -> CENTER
        self.assertEqual(pe.position, HorizontalPosition.CENTER)

    def test_none_resets(self):
        pe = PositionEstimator(raw_cfg())
        pe.update(box_at(0.1, 0.5), W)
        self.assertIsNone(pe.update(None, W))
        self.assertEqual(pe.position, HorizontalPosition.UNKNOWN)


class TestDistance(unittest.TestCase):
    def test_zones(self):
        cfg = raw_cfg()
        for hr, expected in ((0.30, DistanceZone.FAR), (0.65, DistanceZone.IDEAL),
                             (0.95, DistanceZone.CLOSE)):
            de = DistanceEstimator(cfg)
            self.assertEqual(de.update(box_at(0.5, hr), W, H).zone, expected, f"h={hr}")

    def test_hysteresis_band_keeps_state(self):
        cfg = raw_cfg()  # FAR < 0.50, IDEAL 0.55-0.80, CLOSE > 0.90
        de = DistanceEstimator(cfg)
        de.update(box_at(0.5, 0.65), W, H)
        self.assertEqual(de.zone, DistanceZone.IDEAL)
        de.update(box_at(0.5, 0.52), W, H)   # between TOO_FAR and IDEAL_MIN -> keep IDEAL
        self.assertEqual(de.zone, DistanceZone.IDEAL)
        de.update(box_at(0.5, 0.45), W, H)   # below TOO_FAR -> FAR
        self.assertEqual(de.zone, DistanceZone.FAR)
        de.update(box_at(0.5, 0.52), W, H)   # band again -> keep FAR
        self.assertEqual(de.zone, DistanceZone.FAR)

    def test_bigger_box_means_closer_metric(self):
        de = DistanceEstimator(raw_cfg())
        far = de.approx_metric(100, W, H)
        near = de.approx_metric(300, W, H)
        self.assertGreater(far, near)

    def test_calibration(self):
        cfg = raw_cfg()
        de = DistanceEstimator(cfg)
        de.calibrate(0.40)
        self.assertEqual(de.update(box_at(0.5, 0.40), W, H).zone, DistanceZone.IDEAL)
        cfg.validate()


class TestSmoothing(unittest.TestCase):
    def test_ema(self):
        s = ExponentialSmoother(0.5)
        self.assertEqual(s.update(10), 10)
        self.assertEqual(s.update(20), 15)

    def test_debouncer_requires_persistence(self):
        d = CommandDebouncer("STOP", persistence_frames=3, min_duration=0.0)
        self.assertEqual(d.update("LEFT", 0.0), "STOP")
        self.assertEqual(d.update("LEFT", 0.1), "STOP")
        self.assertEqual(d.update("LEFT", 0.2), "LEFT")

    def test_debouncer_rejects_flicker(self):
        d = CommandDebouncer("STOP", persistence_frames=3, min_duration=0.0)
        for i, c in enumerate(["LEFT", "RIGHT"] * 10):
            out = d.update(c, i * 0.05)
        self.assertEqual(out, "STOP")
        self.assertEqual(d.switches, 0)

    def test_min_duration(self):
        d = CommandDebouncer("STOP", persistence_frames=1, min_duration=1.0)
        d.update("LEFT", 0.0)                      # first switch allowed
        self.assertEqual(d.current, "LEFT")
        self.assertEqual(d.update("RIGHT", 0.5), "LEFT")   # too soon
        self.assertEqual(d.update("RIGHT", 1.1), "RIGHT")

    def test_force_is_immediate(self):
        d = CommandDebouncer("FORWARD", persistence_frames=10, min_duration=10.0)
        d.update("FORWARD", 0.0)
        self.assertEqual(d.update("STOP", 0.01, force=True), "STOP")


class TestDecisionEngine(unittest.TestCase):
    P, Z = HorizontalPosition, DistanceZone

    def raw(self, visible=True, conf=0.9, pos=HorizontalPosition.CENTER, zone=DistanceZone.IDEAL,
            blocked=False, cfg=None):
        eng = DecisionEngine(cfg or Config())
        return eng.decide_raw(DecisionInput(visible, conf, pos, zone, blocked))[0]

    def test_rule_table(self):
        P, Z, C = self.P, self.Z, RobotCommand
        table = {
            (P.CENTER, Z.FAR): C.FORWARD, (P.LEFT, Z.FAR): C.FORWARD_LEFT,
            (P.RIGHT, Z.FAR): C.FORWARD_RIGHT, (P.LEFT, Z.IDEAL): C.LEFT,
            (P.RIGHT, Z.IDEAL): C.RIGHT, (P.CENTER, Z.IDEAL): C.STOP,
            (P.CENTER, Z.CLOSE): C.BACKWARD, (P.LEFT, Z.CLOSE): C.BACKWARD,
        }
        for (pos, zone), expected in table.items():
            self.assertEqual(self.raw(pos=pos, zone=zone), expected, f"{pos} {zone}")

    def test_safety_stops(self):
        self.assertEqual(self.raw(visible=False, zone=self.Z.FAR), RobotCommand.STOP)
        self.assertEqual(self.raw(conf=0.1, zone=self.Z.FAR), RobotCommand.STOP)
        self.assertEqual(self.raw(pos=self.P.UNKNOWN, zone=self.Z.FAR), RobotCommand.STOP)

    def test_close_can_stop_instead(self):
        cfg = Config(); cfg.BACKWARD_WHEN_CLOSE = False
        self.assertEqual(self.raw(zone=self.Z.CLOSE, cfg=cfg), RobotCommand.STOP)

    def test_terrain_veto(self):
        cfg = Config(); cfg.USE_TERRAIN_VETO = True
        self.assertEqual(self.raw(zone=self.Z.FAR, blocked=True, cfg=cfg), RobotCommand.STOP)
        self.assertEqual(self.raw(pos=self.P.LEFT, zone=self.Z.FAR, blocked=True, cfg=cfg), RobotCommand.LEFT)
        # veto disabled by default
        self.assertEqual(self.raw(zone=self.Z.FAR, blocked=True), RobotCommand.FORWARD)

    def test_smoothed_update_and_lost_target(self):
        cfg = Config()
        eng = DecisionEngine(cfg)
        far = DecisionInput(True, 0.9, self.P.CENTER, self.Z.FAR)
        t, outs = 0.0, []
        for _ in range(cfg.COMMAND_PERSISTENCE_FRAMES + 2):
            outs.append(eng.update(far, t).command); t += 0.05
        self.assertEqual(outs[0], RobotCommand.STOP)            # not on the first noisy frame
        self.assertEqual(outs[-1], RobotCommand.FORWARD)
        lost = eng.update(DecisionInput(False, 0.0), t + 0.01)  # target lost -> STOP immediately
        self.assertEqual(lost.command, RobotCommand.STOP)
        self.assertTrue(lost.safety_stop)


if __name__ == "__main__":
    unittest.main()
