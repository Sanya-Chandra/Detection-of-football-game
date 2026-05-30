"""
Sports Analytics CV — Team Classifier
Automatically separates players into two teams by dominant jersey color.

Uses HSV histogram extraction from player bounding-box crops and K-Means
clustering (k=2) to find the two dominant jersey colors and assign each
tracked player to Team A or Team B.  The classifier re-clusters every
``update_interval`` frames so it adapts to lighting changes over time.
"""

from __future__ import annotations

import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# BGR colors used to draw team boxes / UI
TEAM_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 220, 255),   # Team A — vivid yellow-green (BGR)
    1: (0, 80, 255),    # Team B — vivid orange-red   (BGR)
}
TEAM_LABELS = {0: "Team A", 1: "Team B"}


def _extract_color_feature(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract a compact HSV-histogram feature vector from the upper-body
    region of a player crop (avoids grass contamination from legs).

    Returns a 1-D float32 array of length 48 (16 bins each for H, S, V),
    or None if the crop is too small to be meaningful.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h_box, w_box = y2 - y1, x2 - x1
    if h_box < 10 or w_box < 10:
        return None

    # Use the upper 55 % of the box (torso / jersey) — skip legs & feet
    y_mid = y1 + int(h_box * 0.55)
    crop = frame[y1:y_mid, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 16-bin histograms for each HSV channel, normalised to [0, 1]
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = np.linalg.norm(feat)
    return feat / (norm + 1e-6)


class TeamClassifier:
    """
    Online team classifier using K-Means jersey-color clustering.

    Parameters
    ----------
    update_interval : int
        How many frames between re-clustering (default 15).
        Lower = more adaptive but slightly more CPU cost.
    min_samples : int
        Minimum number of tracked players needed before clustering starts.
    """

    def __init__(self, config: dict, update_interval: int = 15, min_samples: int = 4):
        self.cfg = config.get("analytics", {})
        self.update_interval = update_interval
        self.min_samples = min_samples

        # track_id → team index (0 or 1); -1 = unclassified
        self._assignments: Dict[int, int] = {}
        # Accumulated features per track for the current clustering window
        self._feature_buffer: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._frame_count = 0
        self._cluster_centers: Optional[np.ndarray] = None  # (2, 48) centres

        # Possession counters per team
        self._team_frames: Dict[int, int] = defaultdict(int)
        self._total_possession_frames = 0

        logger.info("TeamClassifier initialised (auto jersey-color clustering)")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections: list,           # List[Detection] from yolo_detector
    ) -> Dict[int, int]:
        """
        Process one frame: extract features, optionally re-cluster,
        and return the current team-assignment dict {track_id: team_idx}.

        Args:
            frame   : Full BGR frame (used to crop player images).
            detections : Player Detection objects (class_id == 0 only used).

        Returns:
            Dict mapping track_id → team index (0 or 1).  Unassigned players
            are absent from the dict.
        """
        self._frame_count += 1

        # 1. Collect features for this frame
        for det in detections:
            if det.class_id != 0 or det.track_id is None:
                continue
            feat = _extract_color_feature(frame, det.box)
            if feat is not None:
                # Keep at most 5 samples per track to limit memory
                buf = self._feature_buffer[det.track_id]
                buf.append(feat)
                if len(buf) > 5:
                    buf.pop(0)

        # 2. Re-cluster every update_interval frames
        if self._frame_count % self.update_interval == 0:
            self._recluster()

        # 3. Classify any unclassified tracks using current cluster centres
        if self._cluster_centers is not None:
            for tid, feats in self._feature_buffer.items():
                if tid not in self._assignments and feats:
                    feat_avg = np.mean(feats, axis=0)
                    team = self._assign_to_cluster(feat_avg)
                    self._assignments[tid] = team

        return dict(self._assignments)

    def update_possession(self, possessing_team: Optional[int]) -> None:
        """Record which team currently has the ball."""
        if possessing_team is not None:
            self._team_frames[possessing_team] += 1
            self._total_possession_frames += 1

    def get_team_possession_pct(self) -> Dict[int, float]:
        """Return possession % per team {0: 63.2, 1: 36.8}."""
        if self._total_possession_frames == 0:
            return {0: 50.0, 1: 50.0}
        return {
            t: round(self._team_frames[t] / self._total_possession_frames * 100, 1)
            for t in [0, 1]
        }

    def get_team_for_player(self, track_id: int) -> Optional[int]:
        """Return team index (0 or 1) for a given track_id, or None."""
        return self._assignments.get(track_id)

    def get_team_color(self, track_id: int) -> Tuple[int, int, int]:
        """Return BGR color for a player's team, or default green."""
        team = self._assignments.get(track_id)
        if team is None:
            return (0, 255, 135)  # default: green
        return TEAM_COLORS[team]

    def reset(self) -> None:
        self._assignments.clear()
        self._feature_buffer.clear()
        self._cluster_centers = None
        self._frame_count = 0
        self._team_frames.clear()
        self._total_possession_frames = 0

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _recluster(self) -> None:
        """Run K-Means on all collected player features to find two team centres."""
        all_features: List[np.ndarray] = []
        all_tids: List[int] = []

        for tid, feats in self._feature_buffer.items():
            if feats:
                all_features.append(np.mean(feats, axis=0))
                all_tids.append(tid)

        if len(all_features) < self.min_samples:
            return

        X = np.stack(all_features).astype(np.float32)

        # Use OpenCV K-Means (fast, no sklearn dependency for this)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        try:
            _, labels, centres = cv2.kmeans(
                X, 2, None, criteria, attempts=5,
                flags=cv2.KMEANS_PP_CENTERS
            )
        except Exception as e:
            logger.debug(f"TeamClassifier K-Means failed: {e}")
            return

        self._cluster_centers = centres  # shape (2, 48)

        # Re-assign all known tracks using new centres
        new_assignments: Dict[int, int] = {}
        for i, tid in enumerate(all_tids):
            label = int(labels[i][0])
            new_assignments[tid] = label

        # Preserve label consistency: if majority of previously-team-0 players
        # are now labelled 1, flip the mapping so labels don't swap between calls.
        if self._assignments:
            old_0 = [tid for tid, t in self._assignments.items() if t == 0]
            if old_0:
                old_0_new = [new_assignments[tid] for tid in old_0 if tid in new_assignments]
                if old_0_new and sum(old_0_new) > len(old_0_new) / 2:
                    # Majority of team-0 players are now labelled 1 → flip
                    new_assignments = {tid: 1 - t for tid, t in new_assignments.items()}

        self._assignments = new_assignments
        logger.debug(f"TeamClassifier re-clustered: {len(all_tids)} players")

    def _assign_to_cluster(self, feat: np.ndarray) -> int:
        """Assign a single feature vector to the nearest cluster centre."""
        if self._cluster_centers is None:
            return 0
        dists = np.linalg.norm(self._cluster_centers - feat, axis=1)
        return int(np.argmin(dists))
