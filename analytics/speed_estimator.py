"""
Sports Analytics CV — Speed & Distance Estimator
Estimates player speed (km/h) and total distance covered using
pixel-to-meter calibration and per-track position history.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class SpeedEstimator:
    """
    Estimates player speed and distance from frame-by-frame tracking positions.

    Uses a rolling window of positions to smooth speed estimates and reduce
    jitter caused by detection noise.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("analytics", {})
        self.fps = self.cfg.get("fps_assumption", 25)
        self.ppm = self.cfg.get("pixels_per_meter", 10.0)   # pixels per real metre
        self.window = 10  # smoothing window (frames)

        # Per-track histories
        self._positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window + 1))
        self._speeds: Dict[int, List[float]] = defaultdict(list)       # km/h per frame
        self._distances: Dict[int, float] = defaultdict(float)         # cumulative metres

    def update(self, track_id: int, center: Tuple[int, int], frame_idx: int, timestamp: Optional[float] = None) -> float:
        """
        Record a new position for a track and return current speed estimate.

        Args:
            track_id: Unique tracker ID
            center: (cx, cy) pixel position at this frame
            frame_idx: Current frame index (used for time calculation if timestamp is None)
            timestamp: Optional wall-clock timestamp (s) for real-time accurate speeds

        Returns:
            Instantaneous speed in km/h
        """
        if timestamp is None:
            # Fall back to frame_idx-based virtual timestamp
            timestamp = frame_idx / self.fps

        hist = self._positions[track_id]
        hist.append((center, timestamp))

        if len(hist) < 2:
            return 0.0

        # Compute displacement over the window
        pts = list(hist)
        start, end = pts[0], pts[-1]
        start_center, start_time = start
        end_center, end_time = end

        pixel_dist = np.sqrt((end_center[0] - start_center[0]) ** 2 + (end_center[1] - start_center[1]) ** 2)
        real_dist_m = pixel_dist / self.ppm

        # Time elapsed
        elapsed_s = end_time - start_time
        speed_ms = real_dist_m / (elapsed_s + 1e-6)
        speed_kmh = speed_ms * 3.6

        # Apply noise threshold (deadband) to filter out detection/tracking jitter
        # A player moving at less than 1.8 km/h is typically stationary or slightly shifting
        if speed_kmh < 1.8:
            speed_kmh = 0.0
            real_dist_m = 0.0

        self._speeds[track_id].append(speed_kmh)
        self._distances[track_id] += real_dist_m / (len(pts) - 1)  # per-frame increment

        return round(speed_kmh, 1)

    def get_max_speed(self, track_id: int) -> float:
        """Maximum recorded speed for a track (km/h)."""
        speeds = self._speeds.get(track_id, [])
        return round(max(speeds), 1) if speeds else 0.0

    def get_avg_speed(self, track_id: int) -> float:
        """Average speed for a track (km/h)."""
        speeds = self._speeds.get(track_id, [])
        return round(float(np.mean(speeds)), 1) if speeds else 0.0

    def get_distance(self, track_id: int) -> float:
        """Total distance covered by a track (metres)."""
        return round(self._distances.get(track_id, 0.0), 2)

    def get_all_summary(self) -> List[Dict]:
        """Return per-player speed/distance statistics as a list of dicts."""
        summary = []
        for tid in self._positions.keys():
            summary.append({
                "player_id": tid,
                "max_speed_kmh": self.get_max_speed(tid),
                "avg_speed_kmh": self.get_avg_speed(tid),
                "distance_m": self.get_distance(tid),
            })
        return sorted(summary, key=lambda x: x["max_speed_kmh"], reverse=True)

    def reset(self) -> None:
        """Clear all track histories."""
        self._positions.clear()
        self._speeds.clear()
        self._distances.clear()
        logger.debug("SpeedEstimator reset")
