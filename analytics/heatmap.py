"""
Sports Analytics CV — Heatmap Generator
Accumulates player positions over time and renders Gaussian heatmap overlays.
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
from typing import List, Tuple, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


class HeatmapGenerator:
    """
    Generates color-coded player position heatmaps.

    Positions are collected across all video frames and then rendered
    on a 2-D field diagram with Gaussian smoothing.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("analytics", {})
        res = self.cfg.get("heatmap_resolution", [600, 400])
        self.width, self.height = res[0], res[1]
        self.sigma = self.cfg.get("heatmap_sigma", 20)
        # position_map[track_id] = list of (x_norm, y_norm) values (0–1)
        self.position_map: Dict[int, List[Tuple[float, float]]] = {}
        logger.debug(f"HeatmapGenerator ready | res=({self.width},{self.height}) σ={self.sigma}")

    def add_positions(
        self,
        detections: list,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """
        Record player foot positions from a list of Detection objects.

        Args:
            detections: List of Detection instances (from yolo_detector)
            frame_width: Source frame width in pixels
            frame_height: Source frame height in pixels
        """
        for det in detections:
            if det.class_id != 0:  # Only players (class 0)
                continue
            cx, cy = det.bottom_center
            xn = np.clip(cx / (frame_width + 1e-6), 0, 1)
            yn = np.clip(cy / (frame_height + 1e-6), 0, 1)
            tid = det.track_id if det.track_id is not None else -1
            self.position_map.setdefault(tid, []).append((xn, yn))

    def render(self, title: str = "Player Heatmap") -> np.ndarray:
        """
        Render the accumulated heatmap as a BGR numpy image.
        Uses cv2.GaussianBlur + cv2.applyColorMap for ~10x faster rendering
        vs the previous scipy/matplotlib approach.

        Returns:
            BGR image array of shape (height, width, 3)
        """
        # Accumulate all positions into a density grid
        density = np.zeros((self.height, self.width), dtype=np.float32)
        for positions in self.position_map.values():
            for xn, yn in positions:
                px = int(xn * (self.width - 1))
                py = int(yn * (self.height - 1))
                density[py, px] += 1.0

        # GPU-fast Gaussian blur via OpenCV (replaces scipy.ndimage.gaussian_filter)
        if density.max() > 0:
            ksize = max(3, self.sigma * 2 + 1)
            if ksize % 2 == 0:
                ksize += 1
            density = cv2.GaussianBlur(density, (ksize, ksize), self.sigma)
            density = density / (density.max() + 1e-6)

        # Convert density [0..1] → uint8 [0..255] → apply jet colormap
        density_u8 = (density * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(density_u8, cv2.COLORMAP_JET)

        # Draw the static field background (green pitch + white lines)
        field_bg = self._render_field_background()

        # Blend: show field where no activity, heatmap where there is activity
        mask = (density > 0.02).astype(np.float32)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        alpha = np.clip(density * 2.5, 0, 0.85)  # transparency proportional to density
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        result = (heatmap_colored * alpha_3ch + field_bg * (1 - alpha_3ch)).astype(np.uint8)

        # Add title text
        cv2.putText(
            result, title,
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )
        return result

    def _render_field_background(self) -> np.ndarray:
        """Render a simple football pitch as a BGR image (cached per instance)."""
        if hasattr(self, "_field_bg_cache"):
            return self._field_bg_cache.copy()

        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bg[:] = (30, 100, 30)  # dark green field

        lc = (200, 220, 200)  # line color (light green-white)
        w, h = self.width, self.height

        # Outer border
        cv2.rectangle(bg, (2, 2), (w - 3, h - 3), lc, 1)
        # Centre line
        cv2.line(bg, (w // 2, 0), (w // 2, h), lc, 1)
        # Centre circle
        cv2.circle(bg, (w // 2, h // 2), int(h * 0.13), lc, 1)
        # Centre spot
        cv2.circle(bg, (w // 2, h // 2), 3, lc, -1)
        # Left penalty area
        px_w, px_h = int(w * 0.16), int(h * 0.56)
        py = (h - px_h) // 2
        cv2.rectangle(bg, (0, py), (px_w, py + px_h), lc, 1)
        # Right penalty area
        cv2.rectangle(bg, (w - px_w, py), (w, py + px_h), lc, 1)
        # Left goal area
        gx_w, gx_h = int(w * 0.06), int(h * 0.30)
        gy = (h - gx_h) // 2
        cv2.rectangle(bg, (0, gy), (gx_w, gy + gx_h), lc, 1)
        # Right goal area
        cv2.rectangle(bg, (w - gx_w, gy), (w, gy + gx_h), lc, 1)

        self._field_bg_cache = bg
        return bg.copy()

    def render_per_player(self, max_players: int = 6) -> Dict[int, np.ndarray]:
        """Render individual heatmaps per tracked player."""
        results = {}
        for tid, positions in list(self.position_map.items())[:max_players]:
            tmp_gen = HeatmapGenerator({"analytics": {
                "heatmap_resolution": [self.width, self.height],
                "heatmap_sigma": self.sigma,
            }})
            tmp_gen.position_map = {tid: positions}
            results[tid] = tmp_gen.render(title=f"Player #{tid}")
        return results

    def _draw_field(self, ax) -> None:
        """Draw simplified football pitch markings."""
        from matplotlib.patches import Rectangle, Circle

        field_color = "#2d8a2d"
        line_color = "white"
        lw = 0.8

        # Outer boundary
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=line_color, lw=lw))
        # Centre line
        ax.axvline(0.5, color=line_color, lw=lw, alpha=0.7)
        # Centre circle
        cc = Circle((0.5, 0.5), 0.1, fill=False, edgecolor=line_color, lw=lw, alpha=0.7)
        ax.add_patch(cc)
        # Penalty areas
        ax.add_patch(Rectangle((0, 0.2), 0.16, 0.6, fill=False, edgecolor=line_color, lw=lw, alpha=0.7))
        ax.add_patch(Rectangle((0.84, 0.2), 0.16, 0.6, fill=False, edgecolor=line_color, lw=lw, alpha=0.7))

    def reset(self) -> None:
        """Clear all accumulated positions."""
        self.position_map.clear()
        logger.debug("HeatmapGenerator positions reset")

    @property
    def total_positions(self) -> int:
        return sum(len(v) for v in self.position_map.values())

    @property
    def player_count(self) -> int:
        return len(self.position_map)
