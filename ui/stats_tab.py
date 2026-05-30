"""
Sports Analytics CV — Statistics & Export Tab
Full analytics dashboard with PDF/CSV export and visualization.
"""

from __future__ import annotations

import streamlit as st

from utils.logger import get_logger

logger = get_logger(__name__)


def render_stats_tab(config: dict) -> None:
    st.markdown("## 📊 Statistics & Export")

    stats = st.session_state.get("video_stats")
    heatmap_img = st.session_state.get("heatmap_img")

    if stats is None:
        st.info("📌 Run a video analysis first (Video Analytics tab) to see statistics here.")
        _render_stats_placeholder()
        return

    # ── Row 1: Core match KPIs ─────────────────────────────────
    st.markdown("### 🏆 Match Overview")
    ball_pct = f"{stats.ball_detected_frames / max(stats.frames_analyzed, 1) * 100:.1f}%"
    fastest = f"{stats.fastest_speed_kmh} km/h"
    total_dist = f"{stats.total_distance_coverage_m:.0f} m"

    from ui.components import render_stat_row
    render_stat_row(
        {
            "Players Tracked":  str(stats.unique_players_detected),
            "Fastest Speed":    fastest,
            "Ball Detected":    ball_pct,
            "Total Distance":   total_dist,
        },
        variants={
            "Fastest Speed":  "speed",
            "Ball Detected":  "ball",
            "Total Distance": "speed",
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: secondary KPIs ──────────────────────────────────
    render_stat_row(
        {
            "Frames Analyzed": str(stats.frames_analyzed),
            "Duration":        f"{stats.duration_s:.1f}s",
            "FPS":             f"{stats.fps:.1f}",
            "Avg Speed":       f"{sum(s['avg_speed_kmh'] for s in stats.speed_summary) / max(len(stats.speed_summary), 1):.1f} km/h" if stats.speed_summary else "—",
        },
        variants={"Avg Speed": "speed"}
    )

    st.markdown("---")

    # ── Charts row ─────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("""
        <div style="background:rgba(255,179,0,0.06);border:1px solid rgba(255,179,0,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.6rem;">
            <span style="color:#FFB300;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                ⚡ Player Performance — Speed &amp; Distance
            </span>
        </div>
        """, unsafe_allow_html=True)
        if stats.speed_summary:
            import pandas as pd
            df = stats.to_dataframe()
            df.columns = ["Player ID", "Max Speed (km/h)", "Avg Speed (km/h)", "Distance (m)"]
            st.bar_chart(df.set_index("Player ID")[["Max Speed (km/h)", "Avg Speed (km/h)"]])

    with col_chart2:
        st.markdown("""
        <div style="background:rgba(0,150,255,0.06);border:1px solid rgba(0,150,255,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.6rem;">
            <span style="color:#0096FF;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                ⚽ Ball Possession Distribution
            </span>
        </div>
        """, unsafe_allow_html=True)
        poss = stats.possession_by_player
        if poss:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [f"Player #{k}" for k in poss.keys()]
            sizes  = list(poss.values())
            labels.append("No Possession")
            sizes.append(stats.no_possession_pct)

            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#060b14")
            colors = ["#00FF87", "#0096FF", "#FFB300", "#FF4560", "#a855f7", "#64748b"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                colors=colors[:len(sizes)], startangle=90,
                wedgeprops=dict(width=0.55),
                textprops={"color": "#e2e8f0", "fontsize": 9},
            )
            for at in autotexts:
                at.set_color("#060b14")
                at.set_fontweight("bold")
            ax.set_facecolor("#060b14")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No possession data available")

    # ── Heatmap ────────────────────────────────────────────────
    if heatmap_img is not None:
        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(255,69,96,0.06);border:1px solid rgba(255,69,96,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.6rem;">
            <span style="color:#FF4560;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                🔥 Player Position Heatmap
            </span>
        </div>
        """, unsafe_allow_html=True)
        from ui.components import display_frame
        display_frame(heatmap_img, caption="Cumulative player position density across all analyzed frames")

    st.markdown("---")

    # ── Detailed table ─────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(0,255,135,0.05);border:1px solid rgba(0,255,135,0.15);
                border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.6rem;">
        <span style="color:#00FF87;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
            🎯 Detailed Player Statistics
        </span>
    </div>
    """, unsafe_allow_html=True)
    if stats.speed_summary:
        import pandas as pd
        df = pd.DataFrame(stats.speed_summary)
        df.columns = ["Player ID", "Max Speed (km/h)", "Avg Speed (km/h)", "Distance (m)"]
        df["Possession %"] = df["Player ID"].map(
            lambda pid: f"{stats.possession_by_player.get(pid, 0.0):.1f}%"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Export section ─────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);
                border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.8rem;">
        <span style="color:#a855f7;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
            📥 Export Reports
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_pdf, col_csv, col_json = st.columns(3)

    from analytics.report_generator import ReportGenerator
    reporter = ReportGenerator()

    with col_pdf:
        st.markdown("**PDF Report**")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    heatmap_path = None
                    if heatmap_img is not None:
                        from utils.file_utils import get_output_path
                        hp = str(get_output_path("image", "report_heatmap.png"))
                        import cv2
                        cv2.imwrite(hp, heatmap_img)
                        heatmap_path = hp

                    pdf_bytes = reporter.get_pdf_bytes(stats, heatmap_path)
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="sports_analytics_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    with col_csv:
        st.markdown("**CSV Data**")
        if st.button("📊 Export CSV"):
            with st.spinner("Exporting CSV..."):
                try:
                    csv_bytes = reporter.get_csv_bytes(stats)
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="player_statistics.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"CSV export failed: {e}")

    with col_json:
        st.markdown("**JSON Data**")
        if st.button("🔗 Export JSON"):
            import json
            try:
                json_str = json.dumps(stats.to_dict(), indent=2, default=str)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_str.encode(),
                    file_name="match_statistics.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"JSON export failed: {e}")


def _render_stats_placeholder() -> None:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1826, #0a1420);
        border: 2px dashed rgba(168,85,247,0.2);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    ">
        <div style="font-size:5rem;margin-bottom:1.2rem;filter:drop-shadow(0 0 20px rgba(168,85,247,0.5));">📊</div>
        <h3 style="color:#a855f7;font-family:'Rajdhani',sans-serif;font-size:1.6rem;margin-bottom:.6rem;">No Data Yet</h3>
        <p style="color:#7a92b2;max-width:420px;margin:0 auto;line-height:1.7;">
            Run a video analysis to populate the statistics dashboard.
            You'll be able to export PDF reports, CSV data, and visualize all player metrics.
        </p>
        <div style="margin-top:2rem;display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;">
            <span style="background:rgba(255,179,0,0.08);border:1px solid rgba(255,179,0,0.2);border-radius:20px;padding:.3rem .9rem;color:#FFB300;font-size:.75rem;">⚡ Speed Charts</span>
            <span style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:20px;padding:.3rem .9rem;color:#0096FF;font-size:.75rem;">🤝 Possession %</span>
            <span style="background:rgba(255,69,96,0.08);border:1px solid rgba(255,69,96,0.2);border-radius:20px;padding:.3rem .9rem;color:#FF4560;font-size:.75rem;">🔥 Heatmap</span>
            <span style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);border-radius:20px;padding:.3rem .9rem;color:#a855f7;font-size:.75rem;">📄 PDF Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
