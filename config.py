"""
Central configuration for the Autonomous Human-Following Robot.

Every tunable number lives here so you never have to hunt through the code.
All values can be edited in this file; a few of the most useful ones can also
be overridden from the command line (see `python main.py --help`).

Units
-----
* "ratio" values are fractions of the frame size (0.0 - 1.0), which makes the
  thresholds independent of the webcam resolution.
* Times are in seconds unless the name says FRAMES.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Config:
    # ------------------------------------------------------------------ camera
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    # Horizontal field of view of the webcam. Typical laptop webcams: 55-70 deg.
    # Only used for the *approximate* bearing / metric distance read-out.
    CAMERA_HFOV_DEG: float = 60.0
    CAMERA_REOPEN_AFTER_FAILURES: int = 15   # consecutive failed reads -> reopen

    # --------------------------------------------------------------- detection
    # "yolo" = YOLOv8n (default, fastest + most accurate here)
    # "ssd"  = MobileNetV3-SSDLite (same detector family as the reference paper)
    DETECTOR: str = "yolo"
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolov8n.pt")
    SSD_WEIGHTS_PATH: str = str(MODELS_DIR / "ssdlite320_mobilenet_v3_large_coco-a79551df.pth")
    DETECTOR_IMGSZ: int = 320          # YOLO input size (320 = fast, 640 = accurate)
    DEVICE: str = "cpu"                # "cpu" or "cuda:0" if you have an NVIDIA GPU
    CONFIDENCE_THRESHOLD: float = 0.45  # detections / targets below this are ignored
    MIN_BOX_AREA_RATIO: float = 0.004  # ignore tiny boxes (< 0.4 % of the frame)
    DETECT_INTERVAL: int = 1           # run the detector every N frames (tracker in between)

    # -------------------------------------------------------- target selection
    MAX_ASSOCIATION_DIST: float = 0.20  # max centre jump between frames (fraction of frame diagonal)
    MIN_IOU_MATCH: float = 0.20         # IoU that counts as "same person" immediately
    REID_SIMILARITY_THRESHOLD: float = 0.55  # colour-histogram similarity to re-acquire the old target
    APPEARANCE_UPDATE_RATE: float = 0.05     # how fast the stored appearance adapts

    # ---------------------------------------------------------------- tracking
    # Frames the tracker may "coast" on Kalman prediction / visual tracker after the
    # detector stops seeing the target. After that the target is declared LOST.
    LOST_TIMEOUT_FRAMES: int = 15
    COAST_CONFIDENCE_DECAY: float = 0.85   # confidence multiplier per missed frame
    VISUAL_TRACKER: str = "auto"           # "auto" | "csrt" | "kcf" | "mil" | "none"

    # ----------------------------------------------------- position estimation
    # Normalised horizontal error = (cx - frame_cx) / (frame_width / 2), range [-1, 1].
    HORIZONTAL_THRESHOLD: float = 0.18   # |error| above this -> LEFT / RIGHT
    HORIZONTAL_HYSTERESIS: float = 0.06  # must come back inside (threshold - hysteresis) to be CENTER again

    # ----------------------------------------------------- distance estimation
    # Distance proxy = bounding-box height / frame height.
    #   ratio < TOO_FAR_HEIGHT                  -> FAR
    #   IDEAL_MIN_HEIGHT <= ratio <= IDEAL_MAX  -> IDEAL
    #   ratio > TOO_CLOSE_HEIGHT                -> CLOSE
    # The gaps between the bands act as hysteresis (state is kept there).
    TOO_FAR_HEIGHT: float = 0.50
    IDEAL_MIN_HEIGHT: float = 0.55
    IDEAL_MAX_HEIGHT: float = 0.80
    TOO_CLOSE_HEIGHT: float = 0.90
    CALIBRATION_BAND: float = 0.12       # half-width of the ideal band when you press 'c'
    ASSUMED_PERSON_HEIGHT_M: float = 1.70  # only for the rough metric read-out
    SHOW_APPROX_METRIC: bool = True

    # --------------------------------------------------------------- smoothing
    SMOOTHING_FACTOR: float = 0.40        # EMA alpha (1.0 = no smoothing, 0.1 = very smooth)
    COMMAND_PERSISTENCE_FRAMES: int = 4   # new command must win N consecutive frames
    MIN_COMMAND_DURATION: float = 0.40    # seconds a command is held before it may change

    # -------------------------------------------------------------- navigation
    BACKWARD_WHEN_CLOSE: bool = True      # False -> robot just STOPs when too close
    USE_TERRAIN_VETO: bool = False        # experimental: block FORWARD if path looks blocked

    # ------------------------------------------------------------ segmentation
    SEG_WIDTH: int = 96                  # segmentation runs on a tiny image (fast)
    KMEANS_K: int = 4
    KMEANS_SPATIAL_WEIGHT: float = 0.15  # weight of the (row) position feature
    KMEANS_INTERVAL: int = 5             # run every N frames
    FELZ_SCALE: float = 120.0
    FELZ_SIGMA: float = 0.8
    FELZ_MIN_SIZE: int = 40
    FELZ_INTERVAL: int = 15
    SEED_REGION: Tuple[float, float, float, float] = (0.30, 0.82, 0.70, 1.00)  # x1,y1,x2,y2 ratios
    TRAVERSABLE_SEED_SHARE: float = 0.20  # a segment covering >= 20 % of the seed area is "ground"
    GROUND_COLOR_TOLERANCE: float = 18.0  # L*a*b* distance to merge over-segmented ground regions
    FREE_SPACE_MIN: float = 0.50         # centre free-space depth below this -> "path blocked"

    # ---------------------------------------------------- closed-loop heading
    # The webcam cannot turn, so the robot keeps a VIRTUAL heading: a turn command
    # moves the robot's aim across the image until it faces the target (see
    # navigation/heading.py). False = old open-loop behaviour (aim = frame centre).
    CLOSED_LOOP_HEADING: bool = True
    TURN_GAIN: float = 2.5               # turn speed (deg/s) per degree of bearing error
    MIN_TURN_SPEED_DEG: float = 8.0      # slowest turn while a turn command is active
    PERSON_MOVING_DEG_S: float = 6.0     # target bearing changing faster -> "moving left/right"

    # -------------------------------------------------- laptop (camera) motion
    # Background optical flow measures how far the laptop itself was rotated
    # (tracking/ego_motion.py). The robot IS the laptop, so it is turned with it.
    EGO_MOTION: bool = True
    EGO_WIDTH: int = 320                 # flow runs on a downscaled grey image
    EGO_MIN_POINTS: int = 25             # fewer background features -> no estimate
    EGO_DEADBAND_PX: float = 0.35        # per-frame shifts below this are sensor noise
    EGO_MOVING_DEG_S: float = 8.0        # laptop rotating faster than this -> "LAPTOP MOVING"

    # --------------------------------------------------------------- simulator
    SIM_WIDTH: int = 400
    SIM_HEIGHT: int = 480
    SIM_PIXELS_PER_METER: float = 70.0
    SIM_LINEAR_SPEED: float = 0.6        # m/s when FORWARD/BACKWARD
    SIM_ANGULAR_SPEED_DEG: float = 50.0  # deg/s when LEFT/RIGHT
    SIM_TRAIL_LENGTH: int = 600

    # ------------------------------------------------------------------- misc
    WINDOW_NAME: str = "Human-Following Robot"
    PANEL_WIDTH: int = 300

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Raise ValueError for inconsistent settings (caught early in main)."""
        if not (0 < self.TOO_FAR_HEIGHT <= self.IDEAL_MIN_HEIGHT
                < self.IDEAL_MAX_HEIGHT <= self.TOO_CLOSE_HEIGHT):
            raise ValueError(
                "Distance thresholds must satisfy "
                "0 < TOO_FAR <= IDEAL_MIN < IDEAL_MAX <= TOO_CLOSE")
        if not 0 < self.SMOOTHING_FACTOR <= 1:
            raise ValueError("SMOOTHING_FACTOR must be in (0, 1]")
        if not 0 <= self.HORIZONTAL_HYSTERESIS < self.HORIZONTAL_THRESHOLD < 1:
            raise ValueError("Need 0 <= HORIZONTAL_HYSTERESIS < HORIZONTAL_THRESHOLD < 1")
        if self.DETECT_INTERVAL < 1:
            raise ValueError("DETECT_INTERVAL must be >= 1")


def default_config() -> Config:
    return Config()
