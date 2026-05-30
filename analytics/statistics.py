"""
Sports Analytics CV — Match Statistics Aggregator
Consolidates all per-frame analytics into match-level statistics.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchStatistics:
    """Aggregated match statistics."""
    # General
    video_name: str = ""
    total_frames: int = 0
    frames_analyzed: int = 0
    duration_s: float = 0.0
    fps: float = 25.0

    # Player counts
    unique_players_detected: int = 0
    avg_players_per_frame: float = 0.0
    ball_detected_frames: int = 0

    # Speed & distance
    speed_summary: List[Dict] = field(default_factory=list)
    fastest_player_id: Optional[int] = None
    fastest_speed_kmh: float = 0.0
    total_distance_coverage_m: float = 0.0

    # Possession
    possession_by_player: Dict[int, float] = field(default_factory=dict)
    no_possession_pct: float = 0.0

    # Formation
    detected_formation: str = "?"
    formation_confidence: float = 0.0

    # Analysis metadata
    analysis_start_time: str = ""
    analysis_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_dataframe(self):
        """Convert speed summary to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.speed_summary)

    def __repr__(self) -> str:
        return (
            f"MatchStatistics(players={self.unique_players_detected}, "
            f"formation={self.detected_formation}, "
            f"fastest={self.fastest_speed_kmh}km/h)"
        )


class StatisticsAggregator:
    """
    Collects per-frame data from all analytics modules and produces
    a consolidated MatchStatistics object at the end of analysis.
    """

    def __init__(self):
        self._player_frame_counts: Dict[int, int] = {}
        self._ball_frames = 0
        self._total_frames = 0
        self._start_time = time.time()

    def record_frame(
        self,
        detections: list,
        has_ball: bool,
    ) -> None:
        """
        Record per-frame detection data.

        Args:
            detections: List of Detection objects for this frame
            has_ball: Whether a ball was detected
        """
        self._total_frames += 1
        if has_ball:
            self._ball_frames += 1
        for det in detections:
            if det.class_id == 0 and det.track_id is not None:
                self._player_frame_counts[det.track_id] = (
                    self._player_frame_counts.get(det.track_id, 0) + 1
                )

    def build(
        self,
        video_name: str,
        fps: float,
        speed_estimator,
        possession_tracker,
        formation_analyzer,
    ) -> MatchStatistics:
        """
        Build the final MatchStatistics object.

        Args:
            video_name: Source video filename
            fps: Video FPS
            speed_estimator: SpeedEstimator instance
            possession_tracker: PossessionTracker instance
            formation_analyzer: FormationAnalyzer instance

        Returns:
            MatchStatistics populated with all collected data
        """
        elapsed = round(time.time() - self._start_time, 2)
        speed_summary = speed_estimator.get_all_summary()
        possession = possession_tracker.get_possession_pct()

        fastest_id = None
        fastest_kmh = 0.0
        if speed_summary:
            top = speed_summary[0]
            fastest_id = top["player_id"]
            fastest_kmh = top["max_speed_kmh"]

        total_dist = sum(
            s["distance_m"] for s in speed_summary
        )

        avg_players = (
            sum(self._player_frame_counts.values()) / max(self._total_frames, 1)
        )

        return MatchStatistics(
            video_name=video_name,
            total_frames=self._total_frames,
            frames_analyzed=self._total_frames,
            duration_s=round(self._total_frames / max(fps, 1), 2),
            fps=fps,
            unique_players_detected=len(self._player_frame_counts),
            avg_players_per_frame=round(avg_players, 1),
            ball_detected_frames=self._ball_frames,
            speed_summary=speed_summary,
            fastest_player_id=fastest_id,
            fastest_speed_kmh=fastest_kmh,
            total_distance_coverage_m=round(total_dist, 2),
            possession_by_player=possession,
            no_possession_pct=possession_tracker.get_no_possession_pct(),
            detected_formation=formation_analyzer.get_stable_formation(),
            analysis_duration_s=elapsed,
        )
