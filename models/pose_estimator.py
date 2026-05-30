"""
Sports Analytics CV — Pose Estimator
Provides two implementations:
  1. PoseEstimator     — legacy MediaPipe wrapper (gracefully disabled if unavailable)
  2. YoloPoseEstimator — GPU-native YOLOv8-Pose (runs on CUDA, recommended)
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PoseResult:
    """Holds pose landmarks and derived metrics for one player."""
    landmarks: Optional[List[Tuple[float, float, float]]] = None  # (x, y, z) normalized
    visible: bool = False
    joint_angles: Dict[str, float] = None

    def __post_init__(self):
        if self.joint_angles is None:
            self.joint_angles = {}


class PoseEstimator:
    """
    MediaPipe-based pose estimator.

    Processes player image crops and returns body landmark positions.
    Gracefully handles failures (returns empty PoseResult).
    """

    # MediaPipe landmark indices (a subset used for sports analysis)
    JOINTS = {
        "left_knee": (23, 25, 27),    # hip → knee → ankle
        "right_knee": (24, 26, 28),
        "left_elbow": (11, 13, 15),   # shoulder → elbow → wrist
        "right_elbow": (12, 14, 16),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
    }

    def __init__(self, config: dict):
        self.cfg = config.get("pose", {})
        self.enabled = self.cfg.get("enabled", True)
        self._mp_pose = None
        self._pose = None
        if self.enabled:
            self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Lazy-initialize MediaPipe to avoid import overhead if disabled."""
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.cfg.get("model_complexity", 1),
                min_detection_confidence=self.cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.cfg.get("min_tracking_confidence", 0.5),
            )
            logger.info("MediaPipe Pose estimator initialized")
        except ImportError:
            logger.warning("MediaPipe not installed — pose estimation disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"MediaPipe init failed: {e} — pose estimation disabled")
            self.enabled = False

    def estimate(self, frame_rgb: np.ndarray) -> PoseResult:
        """
        Estimate pose from an RGB frame crop.

        Args:
            frame_rgb: RGB numpy array of the player crop

        Returns:
            PoseResult with landmarks and joint angles
        """
        if not self.enabled or self._pose is None:
            return PoseResult()

        try:
            result = self._pose.process(frame_rgb)
            if not result.pose_landmarks:
                return PoseResult()

            h, w = frame_rgb.shape[:2]
            lms = result.pose_landmarks.landmark
            pts = [(lm.x * w, lm.y * h, lm.z * w) for lm in lms]

            angles = self._compute_angles(pts)
            return PoseResult(landmarks=pts, visible=True, joint_angles=angles)
        except Exception as e:
            logger.debug(f"Pose estimation error: {e}")
            return PoseResult()

    def _compute_angles(self, pts: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """Compute joint angles using law of cosines."""
        angles = {}
        for joint_name, (a_idx, b_idx, c_idx) in self.JOINTS.items():
            try:
                a = np.array(pts[a_idx][:2])
                b = np.array(pts[b_idx][:2])
                c = np.array(pts[c_idx][:2])
                ba = a - b
                bc = c - b
                cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                angles[joint_name] = angle
            except Exception:
                pass
        return angles

    def draw_landmarks(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """Draw pose skeleton on frame."""
        if not pose_result.visible or not self.enabled:
            return frame
        try:
            import mediapipe as mp
            mp.solutions.drawing_utils
            mp.solutions.drawing_styles
            # We need the raw MediaPipe result for drawing — skip if not stored
        except Exception:
            pass
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            logger.debug("MediaPipe Pose closed")


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8-Pose (GPU-native, recommended)
# ─────────────────────────────────────────────────────────────────────────────

# COCO 17-keypoint skeleton connections (pairs of keypoint indices)
_POSE_CONNECTIONS = [
    (5, 6),   # left shoulder – right shoulder
    (5, 7),   # left shoulder – left elbow
    (7, 9),   # left elbow – left wrist
    (6, 8),   # right shoulder – right elbow
    (8, 10),  # right elbow – right wrist
    (5, 11),  # left shoulder – left hip
    (6, 12),  # right shoulder – right hip
    (11, 12), # left hip – right hip
    (11, 13), # left hip – left knee
    (13, 15), # left knee – left ankle
    (12, 14), # right hip – right knee
    (14, 16), # right knee – right ankle
]

# Keypoint index pairs that define limb colours (BGR)
_LIMB_COLORS = [
    (255, 100, 100),  # shoulders
    (100, 200, 255),  # left arm
    (100, 200, 255),
    (255, 180, 80),   # right arm
    (255, 180, 80),
    (180, 255, 100),  # torso
    (180, 255, 100),
    (180, 255, 100),
    (100, 255, 200),  # left leg
    (100, 255, 200),
    (255, 150, 255),  # right leg
    (255, 150, 255),
]


class YoloPoseEstimator:
    """
    GPU-native pose estimator using YOLOv8-Pose (yolov8s-pose.pt).

    Runs full-frame inference on the CUDA device and draws COCO 17-keypoint
    skeletons directly onto video frames.  Each detected person is matched
    to the nearest YOLO player bounding box so poses can be associated with
    track IDs.
    """

    def __init__(self, model, config: dict):
        """
        Args:
            model  : ultralytics YOLO model loaded with yolov8s-pose.pt
            config : Full application config dict
        """
        self.model = model
        self.cfg = config.get("pose", {})
        self.conf = self.cfg.get("min_detection_confidence", 0.45)
        self.enabled = True

        # Detect FP16
        try:
            import torch
            first_param = next(iter(model.parameters()), None)
            self._half = (
                first_param is not None
                and first_param.dtype == torch.float16
            )
        except Exception:
            self._half = False

        logger.info("YoloPoseEstimator ready (GPU-native COCO 17-keypoint pose)")

    def process_frame(
        self,
        frame: np.ndarray,
        player_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict]:
        """
        Run pose estimation on a full BGR frame.

        Args:
            frame       : Full BGR video frame.
            player_boxes: Optional list of (x1,y1,x2,y2) YOLO player boxes
                          used to filter poses to detected players only.

        Returns:
            List of pose dicts, one per detected person::

                {
                    "keypoints": np.ndarray shape (17, 3),  # x, y, conf
                    "box"      : (x1, y1, x2, y2),
                }
        """
        import torch
        try:
            with torch.no_grad():
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    verbose=False,
                    half=self._half,
                )
        except Exception as e:
            logger.debug(f"YoloPoseEstimator inference error: {e}")
            return []

        poses = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            kps_data = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes_data = r.boxes.xyxy.cpu().numpy()    # (N, 4)
            for i in range(len(kps_data)):
                box = tuple(int(v) for v in boxes_data[i])
                kps = kps_data[i]  # (17, 3)  x, y, conf

                # If player boxes provided, only keep poses that overlap a known box
                if player_boxes is not None:
                    if not self._matches_any_player(box, player_boxes):
                        continue

                poses.append({"keypoints": kps, "box": box})
        return poses

    def draw_poses(
        self,
        frame: np.ndarray,
        poses: List[Dict],
        kp_radius: int = 4,
        limb_thickness: int = 2,
        conf_thresh: float = 0.4,
    ) -> np.ndarray:
        """
        Draw COCO skeleton overlays on the frame in-place.

        Args:
            frame         : BGR image array (modified in-place).
            poses         : List of pose dicts from :meth:`process_frame`.
            kp_radius     : Radius of keypoint circles.
            limb_thickness: Thickness of skeleton limb lines.
            conf_thresh   : Minimum keypoint confidence to draw.

        Returns:
            Annotated frame.
        """
        for pose in poses:
            kps = pose["keypoints"]  # (17, 3)
            h, w = frame.shape[:2]

            # Draw limbs
            for idx, (i, j) in enumerate(_POSE_CONNECTIONS):
                xi, yi, ci = kps[i]
                xj, yj, cj = kps[j]
                if ci < conf_thresh or cj < conf_thresh:
                    continue
                color = _LIMB_COLORS[idx % len(_LIMB_COLORS)]
                pt1 = (int(xi), int(yi))
                pt2 = (int(xj), int(yj))
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(frame, pt1, pt2, color, limb_thickness, cv2.LINE_AA)

            # Draw keypoints
            for xi, yi, ci in kps:
                if ci < conf_thresh:
                    continue
                pt = (int(xi), int(yi))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(frame, pt, kp_radius, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, kp_radius - 1, (0, 200, 255), -1, cv2.LINE_AA)

        return frame

    @staticmethod
    def _matches_any_player(
        pose_box: Tuple[int, int, int, int],
        player_boxes: List[Tuple[int, int, int, int]],
        iou_thresh: float = 0.25,
    ) -> bool:
        """Return True if pose_box has IoU >= iou_thresh with any player box."""
        px1, py1, px2, py2 = pose_box
        for bx1, by1, bx2, by2 in player_boxes:
            ix1 = max(px1, bx1);  iy1 = max(py1, by1)
            ix2 = min(px2, bx2);  iy2 = min(py2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            union = ((px2-px1)*(py2-py1) + (bx2-bx1)*(by2-by1) - inter)
            if union > 0 and inter / union >= iou_thresh:
                return True
        return False

    def close(self) -> None:
        pass  # GPU model lifecycle managed by ModelManager
