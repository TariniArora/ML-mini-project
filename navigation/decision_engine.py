"""
Robot decision engine: perception -> motion command.

Inputs  : target visibility, confidence, horizontal position, distance zone,
          (optional) "path blocked" cue from the terrain module.
Outputs : one RobotCommand (+ the reason, for the UI and the logs).

Priority rules (highest first)
------------------------------
1. Target not visible / tracker LOST            -> STOP   (safety, immediate)
2. Confidence below CONFIDENCE_THRESHOLD        -> STOP   (safety, immediate)
3. Target too CLOSE                             -> BACKWARD (or STOP, configurable)
4. Target FAR  + CENTER / LEFT / RIGHT          -> FORWARD / FORWARD_LEFT / FORWARD_RIGHT
5. Target IDEAL + LEFT / RIGHT                  -> LEFT / RIGHT (turn in place)
6. Otherwise (IDEAL + CENTER)                   -> STOP   (we are where we want to be)
7. Optional terrain veto: if the path ahead looks blocked, the forward component
   is removed (FORWARD -> STOP, FORWARD_LEFT -> LEFT, FORWARD_RIGHT -> RIGHT).

The raw command is then passed through a CommandDebouncer (utils/smoothing.py)
so the robot does not react to single noisy frames. Safety STOPs skip the
debouncer - stopping must never be delayed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from navigation.distance_estimator import DistanceZone
from navigation.position_estimator import HorizontalPosition
from utils.smoothing import CommandDebouncer


class RobotCommand(str, Enum):
    STOP = "STOP"
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD_LEFT = "FORWARD_LEFT"
    FORWARD_RIGHT = "FORWARD_RIGHT"

    @property
    def label(self) -> str:
        return {
            "STOP": "STOP",
            "FORWARD": "MOVE FORWARD",
            "BACKWARD": "MOVE BACKWARD",
            "LEFT": "TURN LEFT",
            "RIGHT": "TURN RIGHT",
            "FORWARD_LEFT": "FORWARD + TURN LEFT",
            "FORWARD_RIGHT": "FORWARD + TURN RIGHT",
        }[self.value]

    @property
    def motion(self) -> Tuple[int, int]:
        """(linear, angular) unit components. linear +1 = forward,
        angular +1 = turn left (counter-clockwise)."""
        return {
            "STOP": (0, 0), "FORWARD": (1, 0), "BACKWARD": (-1, 0),
            "LEFT": (0, 1), "RIGHT": (0, -1),
            "FORWARD_LEFT": (1, 1), "FORWARD_RIGHT": (1, -1),
        }[self.value]


@dataclass
class DecisionInput:
    target_visible: bool
    confidence: float
    position: HorizontalPosition = HorizontalPosition.UNKNOWN
    distance: DistanceZone = DistanceZone.UNKNOWN
    path_blocked: bool = False


@dataclass
class Decision:
    raw_command: RobotCommand     # what this single frame suggests
    command: RobotCommand         # debounced command actually sent to the robot
    reason: str
    safety_stop: bool


class DecisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debouncer: CommandDebouncer[RobotCommand] = CommandDebouncer(
            RobotCommand.STOP, cfg.COMMAND_PERSISTENCE_FRAMES, cfg.MIN_COMMAND_DURATION)

    # Pure function of the inputs -> easy to unit-test.
    def decide_raw(self, inp: DecisionInput) -> Tuple[RobotCommand, str, bool]:
        if not inp.target_visible:
            return RobotCommand.STOP, "no target - safety stop", True
        if inp.confidence < self.cfg.CONFIDENCE_THRESHOLD:
            return RobotCommand.STOP, f"low confidence ({inp.confidence:.2f}) - safety stop", True
        if inp.position == HorizontalPosition.UNKNOWN or inp.distance == DistanceZone.UNKNOWN:
            return RobotCommand.STOP, "position/distance unknown", True

        pos, dist = inp.position, inp.distance
        if dist == DistanceZone.CLOSE:
            if self.cfg.BACKWARD_WHEN_CLOSE:
                cmd, why = RobotCommand.BACKWARD, "target too close - backing off"
            else:
                cmd, why = RobotCommand.STOP, "target too close - holding"
        elif dist == DistanceZone.FAR:
            cmd = {HorizontalPosition.CENTER: RobotCommand.FORWARD,
                   HorizontalPosition.LEFT: RobotCommand.FORWARD_LEFT,
                   HorizontalPosition.RIGHT: RobotCommand.FORWARD_RIGHT}[pos]
            why = f"target far & {pos.value.lower()} - approaching"
        else:  # IDEAL
            cmd = {HorizontalPosition.CENTER: RobotCommand.STOP,
                   HorizontalPosition.LEFT: RobotCommand.LEFT,
                   HorizontalPosition.RIGHT: RobotCommand.RIGHT}[pos]
            why = ("at ideal distance & centred - holding" if cmd == RobotCommand.STOP
                   else f"ideal distance, target {pos.value.lower()} - turning")

        if inp.path_blocked and self.cfg.USE_TERRAIN_VETO:
            vetoed = {RobotCommand.FORWARD: RobotCommand.STOP,
                      RobotCommand.FORWARD_LEFT: RobotCommand.LEFT,
                      RobotCommand.FORWARD_RIGHT: RobotCommand.RIGHT}.get(cmd)
            if vetoed is not None:
                cmd, why = vetoed, why + " | path blocked: forward vetoed"
        return cmd, why, False

    def update(self, inp: DecisionInput, now: Optional[float] = None) -> Decision:
        now = time.monotonic() if now is None else now
        raw, why, safety = self.decide_raw(inp)
        cmd = self.debouncer.update(raw, now, force=safety)
        if cmd != raw and not safety:
            why = f"{why} (pending {self.debouncer.pending_count}/{self.cfg.COMMAND_PERSISTENCE_FRAMES})"
        return Decision(raw, cmd, why, safety)

    def reset(self, now: Optional[float] = None) -> None:
        """Immediately command STOP (pause, camera failure, user reset)."""
        self.debouncer.update(RobotCommand.STOP, now, force=True)
