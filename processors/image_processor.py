"""
Sports Analytics CV — Image Processor
Runs full detection + annotation pipeline on a single sports image.
Returns annotated image, detections, and position-based insights.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

from models.yolo_detector import YOLODetector, Detection
from analytics.formation_analyzer import FormationAnalyzer
from analytics.team_classifier import TeamClassifier, TEAM_LABELS, TEAM_COLORS
from utils.drawing import draw_field_overlay, draw_stats_panel
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """
    Full image analysis pipeline.

    Steps:
    1. Load & validate image
    2. Run YOLOv8 detection
    3. (Optional) Pose estimation on player crops
    4. Formation analysis from player positions
    5. Draw annotations on output frame
    6. Generate position insights
    """

    def __init__(self, detector: YOLODetector, pose_estimator, config: dict):
        self.detector = detector
        self.pose = pose_estimator  # YoloPoseEstimator or legacy PoseEstimator
        self.config = config
        self.formation_analyzer = FormationAnalyzer(config)
        # Team classifier for single-image team separation
        team_cfg = config.get("team", {})
        self.team_clf = TeamClassifier(config) if team_cfg.get("enabled", True) else None

    def process(self, image_path: str) -> Tuple[np.ndarray, List[Detection], Dict[str, Any]]:
        """
        Analyze a sports image.

        Args:
            image_path: Path to the input image file

        Returns:
            Tuple of:
                - annotated_frame (BGR ndarray)
                - detections (list of Detection)
                - insights (dict of analytics metadata)
        """
        frame = self._load_image(image_path)
        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")

        h, w = frame.shape[:2]
        logger.info(f"Processing image: {image_path} ({w}×{h})")

        # ── Detection ──────────────────────────────────────────
        detections = self.detector.detect(frame)

        # Simple extraction: keep all persons (class 0) and balls (class 32)
        players = [d for d in detections if d.class_id == 0]
        balls   = [d for d in detections if d.class_id == 32]

        # ── Formation analysis ─────────────────────────────────
        positions = [d.center for d in players]
        formation = self.formation_analyzer.analyze(positions, frame_width=w)

        # ── Team classification ────────────────────────────────
        team_colors = {}
        team_labels_map = {}
        if self.team_clf is not None:
            assignments = self.team_clf.update(frame, players)
            for tid, team_idx in assignments.items():
                if tid is not None:
                    team_colors[tid] = TEAM_COLORS[team_idx]
                    team_labels_map[tid] = TEAM_LABELS[team_idx]

        # ── Annotate frame ─────────────────────────────────────
        annotated = draw_field_overlay(frame.copy(), alpha=0.08)
        annotated = self.detector.annotate_frame(
            annotated, players + balls,
            team_colors=team_colors if team_colors else None,
            team_labels=team_labels_map if team_labels_map else None,
        )

        # YOLOv8-Pose skeleton overlay
        if self.pose is not None and hasattr(self.pose, 'process_frame'):
            try:
                player_boxes = [tuple(d.box) for d in players]
                poses = self.pose.process_frame(frame, player_boxes=player_boxes)
                annotated = self.pose.draw_poses(annotated, poses)
            except Exception as e:
                logger.debug(f"Pose skipped on image: {e}")

        # Draw player zone zones (thirds)
        for frac in [1 / 3, 2 / 3]:
            xp = int(w * frac)
            cv2.line(annotated, (xp, 0), (xp, h), (255, 255, 255, 80), 1)

        # Stats overlay
        stats = {
            "Players": str(len(players)),
            "Ball": "Yes" if balls else "No",
            "Formation": formation.formation_str,
        }
        if team_labels_map:
            # Count players per team
            n_a = sum(1 for t in team_labels_map.values() if t == "Team A")
            n_b = sum(1 for t in team_labels_map.values() if t == "Team B")
            stats["Team A"] = str(n_a)
            stats["Team B"] = str(n_b)
        annotated = draw_stats_panel(annotated, stats, position="top-right")

        # ── Insights ───────────────────────────────────────────
        insights = self._generate_insights(players, balls, formation, w, h)

        return annotated, players + balls, insights

    def _load_image(self, path: str) -> np.ndarray | None:
        """Load image from disk or bytes."""
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"cv2.imread failed for: {path}")
        return img

    def _generate_insights(
        self,
        players: List[Detection],
        balls: List[Detection],
        formation,
        frame_w: int,
        frame_h: int,
    ) -> Dict[str, Any]:
        """Generate high-level position insights from detections."""
        third = frame_w / 3.0
        defense_count = sum(1 for p in players if p.center[0] < third)
        mid_count = sum(1 for p in players if third <= p.center[0] < 2 * third)
        attack_count = sum(1 for p in players if p.center[0] >= 2 * third)

        ball_zone = None
        if balls:
            bx = balls[0].center[0]
            if bx < third:
                ball_zone = "Defensive Third"
            elif bx < 2 * third:
                ball_zone = "Middle Third"
            else:
                ball_zone = "Attacking Third"

        return {
            "total_players": len(players),
            "ball_detected": len(balls) > 0,
            "ball_confidence": balls[0].confidence if balls else None,
            "ball_zone": ball_zone,
            "formation": formation.formation_str,
            "players_in_defense": defense_count,
            "players_in_midfield": mid_count,
            "players_in_attack": attack_count,
            "avg_detection_confidence": round(
                float(np.mean([d.confidence for d in players])), 3
            ) if players else 0.0,
        }

    def save_output(self, frame: np.ndarray, output_dir: str = "storage/images/output") -> Path:
        """Save annotated frame to the output directory."""
        from datetime import datetime
        from utils.file_utils import get_output_path
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyzed_{ts}.jpg"
        out = get_output_path("image", filename)
        cv2.imwrite(str(out), frame)
        logger.info(f"Saved annotated image: {out}")
        return out
