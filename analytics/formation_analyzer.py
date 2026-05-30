"""
Sports Analytics CV — Formation Analyzer
Detects team formations using K-Means clustering on player positions.
Labels formations using standard football notation (e.g., 4-4-2).
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FormationResult:
    """Result of formation analysis for one frame."""
    formation_str: str            # e.g., "4-3-3"
    clusters: List[np.ndarray]   # Cluster centroids [[cx, cy], ...]
    labels: np.ndarray           # Cluster label per player
    confidence: float             # Formation detection confidence (0–1)


class FormationAnalyzer:
    """
    Analyzes team formation using K-Means clustering on player centroids.

    Divides the field vertically into defensive/midfield/attacking thirds
    and counts players in each zone to infer the formation string.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("analytics", {})
        self.n_clusters = self.cfg.get("formation_clusters", 4)
        # Rolling buffer of formation strings for temporal smoothing
        self._history: List[str] = []
        self._history_len = 10

    def analyze(self, positions: List[Tuple[int, int]], frame_width: int) -> FormationResult:
        """
        Infer the formation from a list of player center positions.

        Args:
            positions: List of (cx, cy) pixel positions for all detected players
            frame_width: Frame width used for zone division

        Returns:
            FormationResult with formation string and cluster info
        """
        if len(positions) < 3:
            return FormationResult("?", [], np.array([]), 0.0)

        pts = np.array(positions, dtype=np.float32)

        # ── K-Means clustering ──────────────────────────────────
        try:
            from sklearn.cluster import KMeans
            n = min(self.n_clusters, len(pts))
            km = KMeans(n_clusters=n, n_init=5, random_state=42)
            labels = km.fit_predict(pts)
            centroids = km.cluster_centers_
        except Exception as e:
            logger.warning(f"KMeans failed: {e}")
            return FormationResult("?", [], np.array([]), 0.0)

        # ── Zone-based formation inference ──────────────────────
        formation_str = self._infer_formation(pts, frame_width)
        self._history.append(formation_str)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        # Confidence = consistency of recent history
        if self._history:
            most_common = max(set(self._history), key=self._history.count)
            confidence = self._history.count(most_common) / len(self._history)
        else:
            most_common, confidence = formation_str, 0.5

        return FormationResult(
            formation_str=most_common,
            clusters=centroids.tolist(),
            labels=labels,
            confidence=round(confidence, 2),
        )

    def _infer_formation(self, pts: np.ndarray, frame_width: int) -> str:
        """
        Divide field into 3 vertical thirds (def/mid/att) and count players.
        Returns a formation string like '4-3-3'.
        """
        x_coords = pts[:, 0]
        third = frame_width / 3.0

        defense = int(np.sum(x_coords < third))
        midfield = int(np.sum((x_coords >= third) & (x_coords < 2 * third)))
        attack = int(np.sum(x_coords >= 2 * third))

        # Normalize to typical formations (remove goalkeeper from def count)
        if defense > 0:
            defense = max(defense - 1, 0)  # subtract estimated GK

        parts = [p for p in [defense, midfield, attack] if p > 0]
        return "-".join(str(p) for p in parts) if parts else "?"

    def get_stable_formation(self) -> str:
        """Return the most frequently seen formation across history."""
        if not self._history:
            return "?"
        return max(set(self._history), key=self._history.count)

    def reset(self) -> None:
        self._history.clear()
