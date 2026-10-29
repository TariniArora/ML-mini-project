"""
Per-frame CSV logging + an end-of-run summary (FPS, latencies, command stats).

These logs are the ONLY source for performance numbers in the report - nothing
is hard-coded. Run with --log to create logs/run_<timestamp>.csv/.json.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from utils.smoothing import LatencyStats

FIELDS = ["frame", "t", "state", "confidence", "x1", "y1", "x2", "y2", "cx", "error_norm",
          "position", "height_ratio", "zone", "raw_command", "command", "people",
          "detector_ran", "detect_ms", "track_ms", "pipeline_ms", "fps", "path_blocked"]


class RunLogger:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._fh = None
        self._writer = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
            self._writer.writeheader()
        self.t_start = time.perf_counter()
        self.det_lat = LatencyStats()
        self.trk_lat = LatencyStats()
        self.pipe_lat = LatencyStats()
        self.loop_lat = LatencyStats()
        self.commands = Counter()
        self.states = Counter()
        self.frames = 0
        self.events = []

    def log(self, st, fps: float, loop_ms: Optional[float] = None) -> None:
        self.frames += 1
        if st.detector_ran:
            self.det_lat.add(st.detect_ms)
        self.trk_lat.add(st.track_ms)
        self.pipe_lat.add(st.total_ms)
        if loop_ms is not None:
            self.loop_lat.add(loop_ms)
        self.commands[st.decision.command.value] += 1
        self.states[st.track.state.value] += 1
        for e in st.events:
            self.events.append((st.frame_index, e))
        if self._writer is None:
            return
        box = st.track.box if st.track.has_target else None
        self._writer.writerow({
            "frame": st.frame_index, "t": round(time.perf_counter() - self.t_start, 4),
            "state": st.track.state.value, "confidence": round(st.track.confidence, 4),
            "x1": round(box[0], 1) if box else "", "y1": round(box[1], 1) if box else "",
            "x2": round(box[2], 1) if box else "", "y2": round(box[3], 1) if box else "",
            "cx": round(st.position.cx, 1) if st.position else "",
            "error_norm": round(st.position.smoothed_error_norm, 4) if st.position else "",
            "position": st.position.position.value if st.position else "",
            "height_ratio": round(st.distance.smoothed_ratio, 4) if st.distance else "",
            "zone": st.distance.zone.value if st.distance else "",
            "raw_command": st.decision.raw_command.value, "command": st.decision.command.value,
            "people": len(st.detections), "detector_ran": int(st.detector_ran),
            "detect_ms": round(st.detect_ms, 2), "track_ms": round(st.track_ms, 3),
            "pipeline_ms": round(st.total_ms, 2), "fps": round(fps, 2),
            "path_blocked": int(bool(st.segmentation and st.segmentation.path_blocked)),
        })

    def summary(self, extra: Optional[dict] = None, command_switches: int = 0) -> dict:
        wall = time.perf_counter() - self.t_start
        s = {
            "frames": self.frames,
            "wall_time_s": round(wall, 2),
            "average_fps_end_to_end": round(self.frames / wall, 2) if wall > 0 else 0,
            "detector_latency": self.det_lat.summary(),
            "tracker_latency": self.trk_lat.summary(),
            "pipeline_latency": self.pipe_lat.summary(),
            "full_loop_latency_incl_ui": self.loop_lat.summary(),
            "command_distribution": dict(self.commands),
            "track_state_distribution": dict(self.states),
            "debounced_command_switches": command_switches,
            "events_first_20": [f"frame {f}: {e}" for f, e in self.events[:20]],
        }
        if extra:
            s.update(extra)
        return s

    def close(self, summary: Optional[dict] = None) -> None:
        if self._fh:
            self._fh.close()
            if summary is not None:
                with open(self.path.with_suffix(".json"), "w") as f:
                    json.dump(summary, f, indent=2)
