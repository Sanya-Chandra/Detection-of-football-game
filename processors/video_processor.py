"""
Sports Analytics CV — Video Processor
Frame-by-frame video analysis pipeline with tracking, heatmaps,
speed estimation, formation detection, and possession tracking.
Supports progress callbacks for Streamlit UI integration.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Generator, Tuple
from collections import defaultdict, deque

from models.yolo_detector import YOLODetector
from analytics.heatmap import HeatmapGenerator
from analytics.speed_estimator import SpeedEstimator
from analytics.formation_analyzer import FormationAnalyzer
from analytics.possession_tracker import PossessionTracker
from analytics.statistics import StatisticsAggregator, MatchStatistics
from analytics.team_classifier import TeamClassifier, TEAM_LABELS, TEAM_COLORS
from utils.drawing import (
    draw_trajectory, draw_speed_label, draw_stats_panel, draw_neon_ball_trail,
    draw_player_highlight, draw_ball_marker, get_color,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """
    Full video analytics pipeline.

    Processes a sports video frame-by-frame, running:
    - YOLOv8 player & ball detection with ByteTrack tracking
    - Player trajectory drawing
    - Speed estimation
    - Heatmap accumulation
    - Formation detection
    - Ball possession tracking
    - Statistics aggregation

    Outputs an annotated video and a MatchStatistics object.
    """

    def __init__(
        self,
        detector: YOLODetector,
        config: dict,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pose_estimator=None,
    ):
        self.detector = detector
        self.config = config
        self.progress_cb = progress_callback
        self.vid_cfg = config.get("video", {})
        self.ana_cfg = config.get("analytics", {})
        self.fast_mode = self.vid_cfg.get("fast_mode", False)
        self.live_fast_mode = self.vid_cfg.get("live_fast_mode", True)
        self.target_fps = self.vid_cfg.get("target_fps", 30)

        # Optional GPU-native pose estimator (YoloPoseEstimator)
        self.pose_est = pose_estimator
        self._pose_on_live = config.get("pose", {}).get("pose_on_live", False)

        # Team classifier (auto jersey-color clustering)
        team_cfg = config.get("team", {})
        self.team_enabled = team_cfg.get("enabled", True)
        self.team_clf = TeamClassifier(config) if self.team_enabled else None

        # Analytics modules
        self.heatmap_gen = HeatmapGenerator(config)
        self.speed_est = SpeedEstimator(config)
        self.formation_ana = FormationAnalyzer(config)
        self.possession_trk = PossessionTracker(config)
        self.stats_agg = StatisticsAggregator()

        # Trajectory buffers: track_id → deque of (cx, cy)
        self._trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.vid_cfg.get("trajectory_length", 30))
        )

        # Ball trajectory buffer for neon trail (longer trail for visual impact)
        self._ball_trail: deque = deque(maxlen=self.vid_cfg.get("trajectory_length", 30) + 15)

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Tuple[MatchStatistics, Optional[Path]]:
        """
        Process a full video file.

        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated output video
            max_frames: Override max frames from config (None = use config)

        Returns:
            Tuple of (MatchStatistics, output_video_path or None)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        writer = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or self.ana_cfg.get("fps_assumption", 25)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            limit = max_frames or self.vid_cfg.get("max_frames", 0)
            if limit and limit > 0:
                total = min(total, limit)

            logger.info(f"Video: {video_path} | {width}×{height} | {fps:.1f} FPS | {total} frames")

            # ── Setup output writer ────────────────────────────────
            out_path = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*self.vid_cfg.get("output_codec", "mp4v"))
                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                logger.info(f"Output video: {out_path}")

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or (limit > 0 and frame_idx >= limit):
                    break

                annotated = self._process_frame(frame, frame_idx, fps, width, height)

                if writer:
                    writer.write(annotated)

                frame_idx += 1
                if self.progress_cb:
                    self.progress_cb(frame_idx, total)

        finally:
            cap.release()
            if writer:
                writer.release()

        # ── Build final statistics ─────────────────────────────
        stats = self.stats_agg.build(
            video_name=Path(video_path).name,
            fps=fps,
            speed_estimator=self.speed_est,
            possession_tracker=self.possession_trk,
            formation_analyzer=self.formation_ana,
        )
        logger.info(f"Video processing complete: {stats}")
        return stats, out_path

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        width: int,
        height: int,
        timestamp: Optional[float] = None,
        is_live: bool = False,
    ) -> np.ndarray:
        """Run all analytics on a single frame and return annotated version."""
        import torch
        annotated = frame.copy()
        # Use live_fast_mode when streaming to skip heavy effects
        fast = self.live_fast_mode if is_live else self.fast_mode

        # ── Detection + Tracking ─────────────────────────────
        with torch.no_grad():
            detections = self.detector.track(frame, is_live=is_live)
        players = [d for d in detections if d.class_id == 0]
        balls = [d for d in detections if d.class_id == 32]
        has_ball = len(balls) > 0

        # ── Update analytics ───────────────────────────────────
        self.heatmap_gen.add_positions(detections, width, height)
        self.stats_agg.record_frame(detections, has_ball)

        # Player speed & trajectory
        player_positions = []
        current_speeds: Dict[int, float] = {}
        for det in players:
            if det.track_id is None:
                continue
            tid = det.track_id
            cx, cy = det.center
            self._trajectories[tid].append((cx, cy))
            speed = self.speed_est.update(tid, (cx, cy), frame_idx, timestamp=timestamp)
            current_speeds[tid] = speed
            player_positions.append((tid, cx, cy))

        # Ball possession
        ball_center = balls[0].center if has_ball else None
        self.possession_trk.update(player_positions, ball_center)

        # Track ball position for neon trail
        if has_ball:
            self._ball_trail.append(balls[0].center)

        # Formation (every 5 frames to save CPU)
        if frame_idx % 5 == 0 and player_positions:
            positions = [(p[1], p[2]) for p in player_positions]
            self.formation_ana.analyze(positions, frame_width=width)

        # ── Team classification (jersey color) ─────────────────
        team_colors: Dict[int, tuple] = {}
        team_labels_map: Dict[int, str] = {}
        if self.team_clf is not None:
            assignments = self.team_clf.update(frame, players)
            for tid, team_idx in assignments.items():
                team_colors[tid] = TEAM_COLORS[team_idx]
                team_labels_map[tid] = TEAM_LABELS[team_idx]

            # Update team possession
            if has_ball and player_positions:
                top_pid = self.possession_trk.get_top_possessor()
                if top_pid is not None and top_pid in assignments:
                    self.team_clf.update_possession(assignments[top_pid])

        # ── Annotate frame ─────────────────────────────────────

        # 1. Player trajectories (fading trails) — team-colored
        if not fast and self.vid_cfg.get("show_trajectories", True):
            for tid, traj in self._trajectories.items():
                if len(traj) > 1:
                    color = team_colors.get(tid, get_color(tid))
                    draw_trajectory(annotated, list(traj), color=color)

        # 2. Neon ball trail (glowing comet effect)
        if not fast and len(self._ball_trail) > 1:
            annotated = draw_neon_ball_trail(
                annotated,
                list(self._ball_trail),
                color=(0, 255, 255),   # neon cyan
                max_thickness=6,
                glow_radius=12,
            )

        # 3. Player highlights (glowing corner brackets + aura) — team-colored
        if not fast:
            for det in players:
                color = team_colors.get(det.track_id, get_color(det.track_id if det.track_id is not None else 0))
                annotated = draw_player_highlight(annotated, det.box, color=color)

        # 4. Draw bounding boxes with team labels + team colors
        annotated = self.detector.annotate_frame(
            annotated, detections,
            team_colors=team_colors if team_colors else None,
            team_labels=team_labels_map if team_labels_map else None,
        )

        # 5. YOLOv8-Pose skeleton overlay
        run_pose = self.pose_est is not None and (
            not is_live or self._pose_on_live
        ) and (frame_idx % 2 == 0)  # every other frame to save GPU budget
        if run_pose:
            player_boxes = [tuple(d.box) for d in players]
            try:
                poses = self.pose_est.process_frame(frame, player_boxes=player_boxes)
                annotated = self.pose_est.draw_poses(annotated, poses)
            except Exception as e:
                logger.debug(f"Pose estimation skipped: {e}")

        # 6. Ball marker (pulsing neon crosshair)
        if has_ball:
            annotated = draw_ball_marker(
                annotated,
                balls[0].center,
                radius=18,
                color=(0, 255, 255),
                frame_idx=frame_idx,
            )

        # 7. Speed labels
        for det in players:
            if det.track_id in current_speeds:
                draw_speed_label(annotated, det.center, current_speeds[det.track_id])

        # 8. Stats panel — show team possession if team classifier is active
        formation = self.formation_ana.get_stable_formation()
        panel_stats = {
            "Frame": str(frame_idx),
            "Players": str(len(players)),
            "Ball": "Yes" if has_ball else "No",
            "Formation": formation,
        }
        if self.team_clf is not None:
            pct = self.team_clf.get_team_possession_pct()
            panel_stats["Team A"] = f"{pct.get(0, 0):.0f}%"
            panel_stats["Team B"] = f"{pct.get(1, 0):.0f}%"
        elif player_positions:
            top_possessor = self.possession_trk.get_top_possessor()
            if top_possessor:
                panel_stats["Possessor"] = f"#{top_possessor}"
        annotated = draw_stats_panel(annotated, panel_stats, "top-left")

        return annotated

    def get_heatmap(self) -> np.ndarray:
        """Render and return the accumulated heatmap."""
        return self.heatmap_gen.render()

    def reset(self) -> None:
        """Reset all analytics state."""
        self.heatmap_gen.reset()
        self.speed_est.reset()
        self.formation_ana.reset()
        self.possession_trk.reset()
        if self.team_clf is not None:
            self.team_clf.reset()
        self._trajectories.clear()
        self._ball_trail.clear()
        logger.debug("VideoProcessor reset")

    def stream_frames(
        self, video_path: str, max_frames: int = 0
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields annotated frames one at a time.
        Useful for webcam/live preview in Streamlit.
        """
        import time
        cap = cv2.VideoCapture(str(video_path) if video_path != "webcam" else 0)
        if not cap.isOpened():
            logger.error(f"Cannot open stream: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret or (max_frames > 0 and frame_idx >= max_frames):
                    break
                # Pass timestamp of frame start for speed estimation; is_live enables GPU fast path
                yield self._process_frame(frame, frame_idx, fps, width, height, timestamp=start_time, is_live=True)
                frame_idx += 1
                # Enforce target FPS to keep playback smooth
                if self.target_fps > 0:
                    elapsed = time.time() - start_time
                    wait_time = max(0, (1.0 / self.target_fps) - elapsed)
                    if wait_time > 0:
                        time.sleep(wait_time)
        finally:
            cap.release()
