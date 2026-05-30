"""
Sports Analytics CV — Drawing Utilities
Functions to annotate frames with bounding boxes, labels, trajectories, and overlays.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Color palette for team / individual player distinction
# ──────────────────────────────────────────────────────────────
PALETTE = [
    (0, 255, 135),   # Green  — Team A
    (0, 149, 255),   # Blue   — Team B
    (255, 80, 80),   # Red    — Ball / ref
    (255, 220, 0),   # Yellow — Goalkeeper
    (200, 0, 255),   # Purple — Extra
    (255, 130, 0),   # Orange — Extra
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR color for a given tracking ID."""
    return PALETTE[track_id % len(PALETTE)]


def draw_bounding_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    track_id: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box with label and confidence on the frame.

    Args:
        frame: BGR image array
        box: (x1, y1, x2, y2) pixel coordinates
        label: Class name string
        confidence: Detection confidence (0–1)
        track_id: Optional tracking ID for color coding
        color: Optional fixed BGR color
        thickness: Box border thickness

    Returns:
        Annotated frame (in-place)
    """
    x1, y1, x2, y2 = [int(c) for c in box]
    if color is None:
        color = get_color(track_id if track_id is not None else 0)

    # Draw filled top bar for label background
    text = f"{label} {confidence:.2f}" if track_id is None else f"#{track_id} {label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 135),
    thickness: int = 2,
    fade: bool = True,
) -> np.ndarray:
    """
    Draw a fading trajectory trail on the frame.

    Args:
        frame: BGR image array
        points: List of (cx, cy) center points (oldest → newest)
        color: BGR color tuple
        thickness: Line thickness
        fade: Whether to fade older segments

    Returns:
        Annotated frame
    """
    n = len(points)
    for i in range(1, n):
        alpha = i / n if fade else 1.0
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, points[i - 1], points[i], c, thickness, cv2.LINE_AA)
    return frame


def draw_speed_label(
    frame: np.ndarray,
    position: Tuple[int, int],
    speed_kmh: float,
    color: Tuple[int, int, int] = (255, 220, 0),
) -> np.ndarray:
    """Draw a speed annotation near the player."""
    x, y = position
    text = f"{speed_kmh:.1f} km/h"
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_field_overlay(frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Draw a semi-transparent field grid overlay on the frame for tactical reference.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # Horizontal thirds
    for frac in [1 / 3, 2 / 3]:
        y = int(h * frac)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    # Vertical thirds
    for frac in [1 / 3, 2 / 3]:
        x = int(w * frac)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_stats_panel(
    frame: np.ndarray,
    stats: Dict[str, str],
    position: str = "top-right",
) -> np.ndarray:
    """
    Overlay a stats panel on the frame corner.

    Args:
        frame: BGR image
        stats: Dict of label → value strings
        position: Corner to draw at ('top-right', 'top-left', 'bottom-right', 'bottom-left')
    """
    h, w = frame.shape[:2]
    panel_w, line_h = 220, 22
    panel_h = len(stats) * line_h + 16
    padding = 10

    if position == "top-right":
        x0, y0 = w - panel_w - padding, padding
    elif position == "top-left":
        x0, y0 = padding, padding
    elif position == "bottom-right":
        x0, y0 = w - panel_w - padding, h - panel_h - padding
    else:
        x0, y0 = padding, h - panel_h - padding

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, (key, val) in enumerate(stats.items()):
        y = y0 + 14 + i * line_h
        cv2.putText(frame, f"{key}: {val}", (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 135), 1, cv2.LINE_AA)

    return frame


def draw_neon_ball_trail(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),  # neon cyan/yellow in BGR
    max_thickness: int = 6,
    glow_radius: int = 12,
) -> np.ndarray:
    """
    Draw a glowing neon comet trail behind the ball.
    Uses additive blending on a dark overlay to create a realistic glow.

    Args:
        frame: BGR image array
        points: List of (cx, cy) ball center positions (oldest → newest)
        color: BGR neon color (default: neon cyan)
        max_thickness: Thickness of the trail at the newest point
        glow_radius: Gaussian blur radius for the glow effect
    """
    n = len(points)
    if n < 2:
        return frame

    # Create a dark overlay for the neon glow
    glow_layer = np.zeros_like(frame, dtype=np.uint8)

    for i in range(1, n):
        # Progress: 0.0 (oldest) → 1.0 (newest)
        progress = i / n
        # Thickness tapers: thin at tail, thick at head
        thickness = max(1, int(max_thickness * progress))
        # Intensity fades: dim at tail, bright at head
        intensity = progress ** 0.6  # ease-in curve for smooth fade
        c = tuple(int(ch * intensity) for ch in color)

        cv2.line(glow_layer, points[i - 1], points[i], c, thickness, cv2.LINE_AA)

    # Apply Gaussian blur to create the neon glow effect
    if glow_radius > 0:
        ksize = glow_radius * 2 + 1  # must be odd
        glow_blurred = cv2.GaussianBlur(glow_layer, (ksize, ksize), 0)
    else:
        glow_blurred = glow_layer

    # Additive blend: glow on top of original frame
    frame = cv2.add(frame, glow_blurred)
    # Also add the sharp core line on top for crispness
    frame = cv2.add(frame, glow_layer)

    return frame


def draw_player_highlight(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 135),
    glow_alpha: float = 0.25,
    corner_length: int = 15,
    corner_thickness: int = 3,
) -> np.ndarray:
    """
    Draw a glowing highlight around a player with stylish corner brackets
    and a semi-transparent aura fill.

    Args:
        frame: BGR image array
        box: (x1, y1, x2, y2) bounding box
        color: BGR highlight color
        glow_alpha: Transparency of the inner glow fill
        corner_length: Length of the corner bracket lines
        corner_thickness: Thickness of the corner bracket lines
    """
    x1, y1, x2, y2 = [int(c) for c in box]

    # Semi-transparent inner glow fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)

    # Stylish corner brackets (top-left, top-right, bottom-left, bottom-right)
    cl = min(corner_length, (x2 - x1) // 3, (y2 - y1) // 3)
    ct = corner_thickness

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, ct, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, ct, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, ct, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, ct, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, ct, cv2.LINE_AA)

    return frame


def draw_ball_marker(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int = 18,
    color: Tuple[int, int, int] = (0, 255, 255),  # neon cyan
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Draw a pulsing neon circle around the ball for high visibility.

    Args:
        frame: BGR image array
        center: (cx, cy) ball center
        radius: Base circle radius
        color: BGR neon color
        frame_idx: Current frame index (for pulse animation)
    """
    import math
    # Pulsing radius animation
    pulse = int(radius + 4 * math.sin(frame_idx * 0.3))

    # Outer glow ring
    glow_layer = np.zeros_like(frame, dtype=np.uint8)
    cv2.circle(glow_layer, center, pulse + 6, color, 2, cv2.LINE_AA)
    glow_blurred = cv2.GaussianBlur(glow_layer, (15, 15), 0)
    frame = cv2.add(frame, glow_blurred)

    # Core ring
    cv2.circle(frame, center, pulse, color, 2, cv2.LINE_AA)

    # Crosshair lines
    cx, cy = center
    gap = pulse + 4
    arm = 8
    cv2.line(frame, (cx - gap - arm, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + gap + arm, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - gap - arm), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + gap + arm), color, 1, cv2.LINE_AA)

    return frame
