"""
Sports Analytics CV — Report Generator
Exports analytics results as PDF and CSV reports.
"""

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils.logger import get_logger
from analytics.statistics import MatchStatistics

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates PDF and CSV analytics reports from MatchStatistics.

    PDF generation uses ReportLab.  If not installed, a text fallback is used.
    """

    def __init__(self, output_dir: str = "storage/reports"):
        from utils.file_utils import resolve_path
        self.output_dir = resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # PDF Export
    # ──────────────────────────────────────────────────────────────

    def export_pdf(self, stats: MatchStatistics, heatmap_path: Optional[str] = None) -> Path:
        """
        Generate a PDF analytics report.

        Args:
            stats: MatchStatistics object
            heatmap_path: Optional path to heatmap image to embed

        Returns:
            Path to generated PDF file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sports_report_{timestamp}.pdf"
        out_path = self.output_dir / filename

        try:
            self._generate_pdf_reportlab(stats, heatmap_path, out_path)
        except ImportError:
            logger.warning("ReportLab not found — generating text report instead")
            out_path = out_path.with_suffix(".txt")
            self._generate_text_report(stats, out_path)

        logger.info(f"Report saved: {out_path}")
        return out_path

    def _generate_pdf_reportlab(
        self,
        stats: MatchStatistics,
        heatmap_path: Optional[str],
        out_path: Path,
    ) -> None:
        """Generate PDF using ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table,
            TableStyle, Image as RLImage, HRFlowable,
        )

        doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=1.5 * cm)
        styles = getSampleStyleSheet()
        story = []

        # ── Header ────────────────────────────────────────────
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            textColor=colors.HexColor("#00FF87"),
            fontSize=22, spaceAfter=6,
        )
        story.append(Paragraph("⚽ Sports Analytics CV Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        ))
        story.append(HRFlowable(width="100%", color=colors.HexColor("#00FF87"), thickness=1))
        story.append(Spacer(1, 0.3 * cm))

        # ── Summary Table ──────────────────────────────────────
        story.append(Paragraph("Match Overview", styles["Heading2"]))
        data = [
            ["Metric", "Value"],
            ["Video", stats.video_name or "N/A"],
            ["Duration", f"{stats.duration_s:.1f}s"],
            ["Frames Analyzed", str(stats.frames_analyzed)],
            ["FPS", f"{stats.fps:.1f}"],
            ["Players Detected", str(stats.unique_players_detected)],
            ["Ball Detection Frames", str(stats.ball_detected_frames)],
            ["Formation", stats.detected_formation],
            ["Fastest Speed", f"{stats.fastest_speed_kmh:.1f} km/h"],
            ["Total Distance", f"{stats.total_distance_coverage_m:.1f} m"],
        ]
        tbl = Table(data, colWidths=[7 * cm, 9 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00FF87")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

        # ── Speed Table ────────────────────────────────────────
        if stats.speed_summary:
            story.append(Paragraph("Player Speed & Distance", styles["Heading2"]))
            speed_data = [["Player ID", "Max Speed (km/h)", "Avg Speed (km/h)", "Distance (m)"]]
            for row in stats.speed_summary[:10]:
                speed_data.append([
                    f"#{row['player_id']}",
                    f"{row['max_speed_kmh']}",
                    f"{row['avg_speed_kmh']}",
                    f"{row['distance_m']}",
                ])
            stbl = Table(speed_data, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
            stbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0096FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(stbl)
            story.append(Spacer(1, 0.5 * cm))

        # ── Heatmap image ──────────────────────────────────────
        if heatmap_path and Path(heatmap_path).exists():
            story.append(Paragraph("Player Heatmap", styles["Heading2"]))
            img = RLImage(heatmap_path, width=14 * cm, height=9 * cm)
            story.append(img)

        doc.build(story)

    def _generate_text_report(self, stats: MatchStatistics, out_path: Path) -> None:
        """Fallback plain-text report when ReportLab is unavailable."""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== Sports Analytics CV Report ===\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            for key, val in stats.to_dict().items():
                f.write(f"{key}: {val}\n")

    # ──────────────────────────────────────────────────────────────
    # CSV Export
    # ──────────────────────────────────────────────────────────────

    def export_csv(self, stats: MatchStatistics) -> Path:
        """
        Export player speed/distance statistics as a CSV file.

        Returns:
            Path to generated CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"player_stats_{timestamp}.csv"
        out_path = self.output_dir / filename

        rows = stats.speed_summary or []
        if not rows:
            rows = [{"player_id": "N/A", "max_speed_kmh": 0, "avg_speed_kmh": 0, "distance_m": 0}]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"CSV exported: {out_path}")
        return out_path

    def get_pdf_bytes(self, stats: MatchStatistics, heatmap_path: Optional[str] = None) -> bytes:
        """Generate PDF and return as bytes (for Streamlit download)."""
        path = self.export_pdf(stats, heatmap_path)
        with open(path, "rb") as f:
            return f.read()

    def get_csv_bytes(self, stats: MatchStatistics) -> bytes:
        """Generate CSV and return as bytes (for Streamlit download)."""
        path = self.export_csv(stats)
        with open(path, "rb") as f:
            return f.read()
