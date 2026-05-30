"""
Sports Analytics CV — YOLO Detector
Wraps YOLOv8 to detect players, balls, and field elements from frames.
Produces structured Detection objects for downstream analytics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single object detection result."""
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Foot position — useful for heatmap placement."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)


class YOLODetector:
    """
    YOLOv8-based object detector for sports analytics.

    Detects players (person class) and sports balls from images/frames.
    Supports both one-shot detection and tracking across video frames.
    """

    # COCO class names relevant to sports analytics
    SPORTS_CLASSES = {
        0: "Player",
        32: "Ball",
        # Court/field objects (present in some COCO variants)
        60: "Table",
        38: "Tennis Racket",
        37: "Sports Ball",
    }

    def __init__(self, model, config: dict):
        """
        Args:
            model: ultralytics YOLO model instance
            config: Full application config dict
        """
        self.model = model
        self.cfg = config.get("detection", {})
        self.conf_thresh = self.cfg.get("confidence_threshold", 0.25)
        self.ball_conf_thresh = self.cfg.get("ball_confidence_threshold", 0.15)
        self.iou_thresh = self.cfg.get("iou_threshold", 0.50)
        self.img_size = self.cfg.get("image_size", 1280)
        self.live_img_size = self.cfg.get("live_image_size", 640)
        self.target_classes = self.cfg.get("classes", {}).get("target_classes", [0, 32])
        
        # Detect if model is on CUDA for FP16 inference
        try:
            import torch
            first_param = next(iter(model.parameters()), None)
            self._use_half = (
                first_param is not None
                and first_param.dtype == torch.float16
            )
        except Exception:
            self._use_half = False
        
        # Player-specific filtering configs
        self.player_conf_thresh = self.cfg.get("player_confidence_threshold", 0.30)
        self.player_max_area = self.cfg.get("player_max_area", 180000)
        self.player_field_y_min = self.cfg.get("player_field_y_min", 0.28)
        self.player_field_y_max = self.cfg.get("player_field_y_max", 0.92)
        self.player_min_box_height = self.cfg.get("player_min_box_height", 0.06)
        
        logger.debug(f"YOLODetector init | conf={self.conf_thresh} | ball_conf={self.ball_conf_thresh} | imgsz={self.img_size} | classes={self.target_classes}")

    def detect(self, frame: np.ndarray, is_live: bool = False) -> List[Detection]:
        """
        Run inference on a single frame and return structured detections.
        Includes an automatic fallback pass for ball detection if the primary
        pass misses it (balls are small and often have low confidence).
        The fallback is skipped in live/stream mode to maximise FPS.

        Args:
            frame: BGR numpy array (H, W, 3)
            is_live: If True, use smaller imgsz and skip ball fallback for max FPS

        Returns:
            List of Detection objects
        """
        imgsz = self.live_img_size if is_live else self.img_size
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=imgsz,
            classes=self.target_classes,
            verbose=False,
            half=self._use_half,
        )
        detections = self._parse_results(results, is_live=is_live)

        # ── Fallback ball detection pass ────────────────────────
        # Skipped in live mode to maximise FPS.
        # If no ball was found in the primary pass, run a targeted
        # low-confidence pass specifically for sports ball (class 32)
        has_ball = any(d.class_id == 32 for d in detections)
        if not is_live and not has_ball and self.ball_conf_thresh < self.conf_thresh:
            ball_results = self.model.predict(
                source=frame,
                conf=self.ball_conf_thresh,
                iou=self.iou_thresh,
                imgsz=max(self.img_size, 1280),  # ensure high-res for small balls
                classes=[32],  # only sports ball
                verbose=False,
                half=self._use_half,
            )
            ball_dets = self._parse_results(ball_results, is_live=False)
            if ball_dets:
                # Take only the highest-confidence ball to avoid false positives
                best_ball = max(ball_dets, key=lambda d: d.confidence)
                detections.append(best_ball)
                logger.debug(f"Ball fallback: found ball with conf={best_ball.confidence:.3f}")

        return detections

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml", is_live: bool = False) -> List[Detection]:
        """
        Run detection + tracking on a single frame.
        Includes ball fallback detection pass (same as detect()), skipped in live mode.

        Args:
            frame: BGR numpy array
            tracker: Tracker config name (ultralytics built-in)
            is_live: If True, use smaller imgsz and skip ball fallback for max FPS

        Returns:
            List of Detection objects with track_id assigned
        """
        imgsz = self.live_img_size if is_live else self.img_size
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=imgsz,
            classes=self.target_classes,
            tracker=tracker,
            persist=True,
            verbose=False,
            half=self._use_half,
        )
        detections = self._parse_results(results, with_tracking=True, is_live=is_live)

        # ── Fallback ball detection pass ────────────────────────
        # Skipped in live mode to maximise FPS.
        has_ball = any(d.class_id == 32 for d in detections)
        if not is_live and not has_ball and self.ball_conf_thresh < self.conf_thresh:
            ball_results = self.model.predict(
                source=frame,
                conf=self.ball_conf_thresh,
                iou=self.iou_thresh,
                imgsz=max(self.img_size, 1280),
                classes=[32],
                verbose=False,
                half=self._use_half,
            )
            ball_dets = self._parse_results(ball_results, is_live=False)
            if ball_dets:
                best_ball = max(ball_dets, key=lambda d: d.confidence)
                detections.append(best_ball)
                logger.debug(f"Ball fallback (track): found ball with conf={best_ball.confidence:.3f}")

        return detections

    def _parse_results(
        self,
        results,
        with_tracking: bool = False,
        is_live: bool = False,
    ) -> List[Detection]:
        """
        Convert ultralytics Results into Detection objects.

        Args:
            results      : ultralytics prediction results
            with_tracking: Whether track IDs are present in the results
            is_live      : If True, skip y-axis zone filtering (designed for broadcast
                           pitch footage; would reject everyone in a close-up webcam feed)
        """
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            h, w = r.orig_shape
            boxes = r.boxes
            for i in range(len(boxes)):
                try:
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    track_id = None
                    if with_tracking and boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())
                    cls_name = self.SPORTS_CLASSES.get(cls_id, f"Class{cls_id}")

                    # Apply player-specific filtering to ignore stands/crowds/coaches
                    # These filters are designed for broadcast/aerial pitch footage.
                    # In live/webcam mode we skip the y-axis zone filter because
                    # a close-up camera fills the frame with the subject.
                    if cls_id == 0:
                        # 1. Player confidence threshold
                        if conf < self.player_conf_thresh:
                            continue

                        # 2. Player max area
                        area = (x2 - x1) * (y2 - y1)
                        if area > self.player_max_area:
                            continue

                        # 3. Y-axis zone filtering — SKIP in live mode
                        if not is_live:
                            y_feet_frac = y2 / h
                            if y_feet_frac < self.player_field_y_min or y_feet_frac > self.player_field_y_max:
                                continue

                        # 4. Minimum bounding box height
                        box_height_frac = (y2 - y1) / h
                        if box_height_frac < self.player_min_box_height:
                            continue

                    detections.append(Detection(
                        box=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=track_id,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse box {i}: {e}")
        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        team_labels: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on a frame copy.

        Args:
            frame: Source BGR frame
            detections: List of Detection objects
            team_colors: Optional dict of {track_id: BGR color} overrides from TeamClassifier
            team_labels: Optional dict of {track_id: team_label_str} e.g. {3: "Team A"}

        Returns:
            New annotated BGR frame
        """
        from utils.drawing import draw_bounding_box, get_color

        out = frame.copy()
        for det in detections:
            tid = det.track_id if det.track_id is not None else det.class_id
            # Use team color if available, otherwise fall back to palette
            if team_colors and tid in team_colors:
                color = team_colors[tid]
            else:
                color = get_color(tid)

            # Build label — add team tag if available
            label = det.class_name
            if team_labels and tid in team_labels:
                label = f"{team_labels[tid]}"

            out = draw_bounding_box(out, det.box, label, det.confidence, det.track_id, color)
        return out
