"""
Sports Analytics CV — Ball Possession Tracker
Attributes ball possession to the nearest player in each frame
and accumulates possession percentages over the video.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger(__name__)


class PossessionTracker:
    """
    Tracks which player (by track_id) is in possession of the ball.

    Possession is assigned to the player whose bounding box center is
    closest to the ball center, within a configurable radius threshold.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("analytics", {})
        self.radius = self.cfg.get("possession_radius_px", 80)
        # frame_count[track_id] = number of frames in possession
        self._frame_counts: Dict[int, int] = defaultdict(int)
        self._no_possession_frames = 0
        self._total_frames = 0

    def update(
        self,
        player_positions: List[Tuple[int, int, int]],  # (track_id, cx, cy)
        ball_position: Optional[Tuple[int, int]],
    ) -> Optional[int]:
        """
        Record possession for this frame.

        Args:
            player_positions: List of (track_id, cx, cy)
            ball_position: (bx, by) ball center or None if not detected

        Returns:
            track_id of the player in possession, or None
        """
        self._total_frames += 1

        if ball_position is None or not player_positions:
            self._no_possession_frames += 1
            return None

        bx, by = ball_position
        min_dist = float("inf")
        closest_id = None

        for tid, px, py in player_positions:
            dist = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_id = tid

        if min_dist <= self.radius and closest_id is not None:
            self._frame_counts[closest_id] += 1
            return closest_id
        else:
            self._no_possession_frames += 1
            return None

    def get_possession_pct(self) -> Dict[int, float]:
        """Return possession percentage per track_id."""
        total = sum(self._frame_counts.values())
        if total == 0:
            return {}
        return {tid: round(cnt / total * 100, 1) for tid, cnt in self._frame_counts.items()}

    def get_top_possessor(self) -> Optional[int]:
        """Return track_id of player with most possession."""
        if not self._frame_counts:
            return None
        return max(self._frame_counts, key=self._frame_counts.get)

    def get_no_possession_pct(self) -> float:
        """Percentage of frames where no player had possession."""
        if self._total_frames == 0:
            return 0.0
        return round(self._no_possession_frames / self._total_frames * 100, 1)

    def summary_dict(self) -> Dict:
        """Return a structured summary of possession statistics."""
        pct = self.get_possession_pct()
        return {
            "total_frames_analyzed": self._total_frames,
            "possession_by_player": pct,
            "no_possession_pct": self.get_no_possession_pct(),
            "top_possessor_id": self.get_top_possessor(),
        }

    def reset(self) -> None:
        self._frame_counts.clear()
        self._no_possession_frames = 0
        self._total_frames = 0
