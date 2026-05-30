"""
Sports Analytics CV — Video Analytics Tab
Streamlit UI for sports video processing with live preview,
heatmaps, speed/distance, possession, and ball trail visualization.
"""

from __future__ import annotations

import streamlit as st
import time
from pathlib import Path
from typing import Optional

from ui.components import display_frame, render_stat_row
from utils.logger import get_logger

logger = get_logger(__name__)


def render_video_tab(video_processor, config: dict, premium_detector=None) -> None:
    st.markdown("## 🎬 Video Analytics")
    st.markdown("Upload a sports video to track players, measure speeds, analyze possession and generate heatmaps.")

    # ── Sidebar controls ────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛 Video Settings")
        max_frames = st.slider("Max frames to analyze", 50, 1000, 200, 50, key="vid_maxframes",
                               help="Limit frames for faster processing on CPU")
        show_trajectories = st.checkbox("Draw player trajectories", value=True, key="vid_traj")
        show_heatmap = st.checkbox("Generate heatmap", value=True, key="vid_heatmap")
        webcam_mode = st.checkbox("🔴 Use webcam (live)", value=False, key="vid_webcam")

        # ── Choose between Fast Mode and Premium Quality Mode ───────
        if premium_detector is not None and not webcam_mode:
            st.markdown("---")
            st.markdown("### ⚙️ Detection Quality")
            video_mode = st.radio(
                "Model Accuracy",
                options=["🚀 Fast (High FPS, YOLOv8s)", "🏆 Premium (Best Detection, YOLOv8l)"],
                index=0,
                key="vid_mode",
                help="🚀 Fast uses a lighter model (~40 FPS) for fast processing. \n\n🏆 Premium uses a heavy-duty model that misses zero humans but takes longer to process.",
            )
            # Dynamically switch the detector inside the processor!
            if "Premium" in video_mode:
                video_processor.detector = premium_detector
            else:
                # Store the original live_detector in a private attribute first to retrieve it
                if not hasattr(video_processor, "_live_detector"):
                    video_processor._live_detector = video_processor.detector
                video_processor.detector = video_processor._live_detector

    # ── Upload ──────────────────────────────────────────────────
    sample_dir = Path("storage")
    samples = list(sample_dir.glob("*.mp4")) + list(sample_dir.glob("*.avi"))

    video_path: Optional[Path] = None

    if webcam_mode:
        st.info("📡 Webcam mode is active. Click **Start Webcam** to begin live tracking.")
        if st.button("▶ Start Webcam"):
            _run_webcam_stream(video_processor, config)
        return

    col_up, col_sample = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop a sports video (MP4 / AVI)",
            type=["mp4", "avi", "mov"],
            key="vid_uploader",
        )
    with col_sample:
        st.markdown("**Or use a sample:**")
        if samples:
            selected_sample = st.selectbox(
                "Sample videos",
                ["— Upload your own —"] + [s.name for s in samples],
                key="vid_sample",
            )
        else:
            selected_sample = "— Upload your own —"

    if uploaded:
        from utils.file_utils import save_uploaded_file
        video_path = save_uploaded_file(uploaded.read(), uploaded.name, "data/input")
    elif selected_sample and selected_sample != "— Upload your own —":
        video_path = sample_dir / selected_sample

    # ── Process video ───────────────────────────────────────────
    if video_path:
        try:
            st.markdown(f"**Source:** `{video_path.name}`")
            st.video(str(video_path))
            video_processor.config["video"]["show_trajectories"] = show_trajectories

            if st.button("🚀 Analyze Video", key="vid_analyze"):
                video_processor.reset()
                _run_video_analysis(video_processor, video_path, max_frames, show_heatmap, config)

            # Show previously computed results from session state
            if "video_stats" in st.session_state and st.session_state.video_stats:
                _render_video_results(
                    st.session_state.video_stats,
                    st.session_state.get("heatmap_img"),
                    st.session_state.get("annotated_video_path"),
                )
        except Exception as e:
            st.error(f"🚨 An unexpected error occurred during video analysis: {e}")
            logger.exception("Global Video Tab Error")
    else:
        st.info("👆 Upload a sports video or select a sample to begin analysis.")
        _render_video_placeholder()


def _run_video_analysis(video_processor, video_path: Path, max_frames: int, show_heatmap: bool, config: dict) -> None:
    """Execute video analysis and store results in session state."""
    import threading

    class ThreadState:
        def __init__(self):
            self.current = 0
            self.total = max_frames
            self.complete = False
            self.error = None
            self.stats = None
            self.out_vid = None

    state = ThreadState()

    def progress_cb(current: int, total: int):
        state.current = current
        state.total = total

    video_processor.progress_cb = progress_cb

    from utils.file_utils import get_output_path
    out_path = str(get_output_path("video", f"{video_path.stem}_analyzed.mp4"))

    def worker():
        try:
            stats, out_vid = video_processor.process_video(
                str(video_path),
                output_path=out_path,
                max_frames=max_frames,
            )
            state.stats = stats
            state.out_vid = out_vid
            state.complete = True
        except Exception as e:
            state.error = str(e)
            state.complete = True

    try:
        thread = threading.Thread(target=worker)
        thread.start()

        progress_bar = st.progress(0, text="⏳ Initializing analysis...")
        status_text = st.empty()
        start_time = time.time()

        while not state.complete:
            time.sleep(0.1)
            current = state.current
            total = state.total
            elapsed = time.time() - start_time
            pct = min(current / max(total, 1), 1.0)
            pct_100 = int(pct * 100)

            if current > 0 and elapsed > 0:
                fps = current / elapsed
                eta = (total - current) / fps
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta > 60 else f"{eta:.1f}s"
                progress_text = (
                    f"⏳ Processing: {pct_100}% | Frame {current}/{total} | "
                    f"Speed: {fps:.1f} FPS | Elapsed: {elapsed:.1f}s | ETA: {eta_str}"
                )
            else:
                progress_text = f"⏳ Processing: {pct_100}% | Frame {current}/{total} | Elapsed: {elapsed:.1f}s"

            progress_bar.progress(pct, text=progress_text)

    except Exception as e:
        st.error(f"Video monitoring loop failed: {e}")
        return

    if state.error:
        progress_bar.empty()
        st.error(f"Video processing failed: {state.error}")
    else:
        progress_bar.progress(1.0, text="✅ Analysis complete!")
        stats = state.stats
        out_vid = state.out_vid
        status_text.success(f"Processed {stats.frames_analyzed} frames in {time.time() - start_time:.1f}s")

        st.session_state.video_stats = stats
        st.session_state.annotated_video_path = str(out_vid) if out_vid else None

        if show_heatmap:
            heatmap = video_processor.get_heatmap()
            heatmap_out = str(get_output_path("image", f"{video_path.stem}_heatmap.png"))
            import cv2
            cv2.imwrite(heatmap_out, heatmap)
            st.session_state.heatmap_img = heatmap
        else:
            st.session_state.heatmap_img = None

        st.rerun()


def _run_webcam_stream(video_processor, config: dict) -> None:
    """
    Live webcam analytics stream.

    Fixes applied vs the old version:
    - Passes the fully-configured VideoProcessor (with GPU models, team classifier,
      pose estimator) directly into StreamProcessor — no bare new instance.
    - Stop button uses st.session_state so it works reliably mid-stream.
    - Shows real FPS, frame index, and detection count as a live overlay.
    - BGR→RGB conversion done correctly before display.
    """
    import cv2
    import time

    # ── Stop control via session state (button re-render won't interrupt generator) ─
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("⏹ Stop Webcam", key="stop_webcam_btn"):
            st.session_state.webcam_running = False
            st.info("Webcam stopped.")
            return

    # ── Live stats display slots ─────────────────────────────────
    frame_slot  = st.empty()
    metrics_row = st.empty()
    status_slot = st.empty()

    status_slot.info("📡 Opening camera… (this may take 1–2 seconds)")

    try:
        from processors.stream_processor import StreamProcessor

        # Pass the FULL video_processor — not a bare new one.
        # This means team classifier, YOLOv8-Pose, GPU FP16 model all work live.
        stream = StreamProcessor(video_processor, config)
        st.session_state.webcam_running = True

        frame_idx = 0
        t_start = time.time()
        t_last_fps = t_start
        fps_display = 0.0
        fps_window = 0

        for annotated_bgr in stream.start_webcam(camera_index=0):
            # ── Check stop flag ──────────────────────────────────
            if not st.session_state.get("webcam_running", True):
                stream.stop()
                status_slot.success("✅ Webcam stream stopped.")
                break

            frame_idx += 1
            now = time.time()

            # FPS calculated over a 1-second rolling window
            fps_window += 1
            elapsed_window = now - t_last_fps
            if elapsed_window >= 1.0:
                fps_display = fps_window / elapsed_window
                fps_window = 0
                t_last_fps = now

            # ── Burn FPS + frame counter onto the frame ──────────
            h, w = annotated_bgr.shape[:2]
            overlay_text = f"Frame: {frame_idx}  |  {fps_display:.1f} FPS  |  {w}x{h}"
            cv2.putText(
                annotated_bgr, overlay_text,
                (w - 340, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 135), 1, cv2.LINE_AA
            )

            # ── Display frame (BGR → RGB for Streamlit) ──────────
            frame_rgb = annotated_bgr[:, :, ::-1]
            frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            # ── Live metric row ───────────────────────────────────
            total_elapsed = now - t_start
            metrics_row.markdown(
                f"<div style='display:flex;gap:2rem;padding:.3rem 0;font-family:monospace;font-size:.85rem;color:#7a92b2;'>"
                f"<span>🎞 <b style='color:#00FF87'>Frame {frame_idx}</b></span>"
                f"<span>⚡ <b style='color:#FFB300'>{fps_display:.1f} FPS</b></span>"
                f"<span>⏱ <b style='color:#00D4FF'>{total_elapsed:.0f}s</b></span>"
                f"<span>📐 <b style='color:#a855f7'>{w}×{h}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # First frame: clear the "opening camera" status
            if frame_idx == 1:
                status_slot.empty()

    except Exception as e:
        status_slot.error(f"❌ Webcam error: {e}")
        logger.exception("Webcam stream error")
    finally:
        st.session_state.webcam_running = False


def _render_video_results(stats, heatmap_img, video_path: Optional[str]) -> None:
    """Render the 6 showcase analytics panels after video processing."""
    st.markdown("---")
    st.markdown("## 📊 Analytics Results")

    # ── Top KPI row: 4 primary metrics ─────────────────────────
    ball_pct = f"{stats.ball_detected_frames / max(stats.frames_analyzed, 1) * 100:.1f}%"
    render_stat_row(
        {
            "Players Tracked":   str(stats.unique_players_detected),
            "Fastest Speed":     f"{stats.fastest_speed_kmh} km/h",
            "Ball Detected":     ball_pct,
            "Total Distance":    f"{stats.total_distance_coverage_m:.0f} m",
        },
        variants={
            "Fastest Speed":  "speed",
            "Ball Detected":  "ball",
            "Total Distance": "speed",
        }
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Speed table  +  Possession donut ─────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # 1. ⚡ Speed & Distance
        st.markdown("""
        <div style="background:rgba(255,179,0,0.06);border:1px solid rgba(255,179,0,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.3rem;">
            <span style="color:#FFB300;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                ⚡ Speed &amp; Distance per Player
            </span>
        </div>
        """, unsafe_allow_html=True)
        if stats.speed_summary:
            import pandas as pd
            df = pd.DataFrame(stats.speed_summary)
            df.columns = ["Player ID", "Max Speed (km/h)", "Avg Speed (km/h)", "Distance (m)"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No speed data available")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. 🤝 Ball Possession
        st.markdown("""
        <div style="background:rgba(0,150,255,0.06);border:1px solid rgba(0,150,255,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.3rem;">
            <span style="color:#0096FF;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                🤝 Ball Possession %
            </span>
        </div>
        """, unsafe_allow_html=True)
        poss = stats.possession_by_player
        if poss:
            import pandas as pd
            poss_df = pd.DataFrame([
                {"Player": f"#{k}", "Possession %": f"{v:.1f}%"}
                for k, v in sorted(poss.items(), key=lambda x: -x[1])
            ])
            st.dataframe(poss_df, use_container_width=True, hide_index=True)
            st.markdown(
                f'<div style="color:#7a92b2;font-size:.8rem;margin-top:.4rem;">'
                f'No clear possession: <strong style="color:#e2e8f0;">{stats.no_possession_pct:.1f}%</strong></div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No possession data")

    with col_right:
        # 3. 🔥 Heatmap
        st.markdown("""
        <div style="background:rgba(255,69,96,0.06);border:1px solid rgba(255,69,96,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.3rem;">
            <span style="color:#FF4560;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                🔥 Player Position Heatmap
            </span>
        </div>
        """, unsafe_allow_html=True)
        if heatmap_img is not None:
            display_frame(heatmap_img, caption="Cumulative player movement density")
        else:
            st.info("Enable heatmap in sidebar to see this")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. 📈 Speed distribution chart
        st.markdown("""
        <div style="background:rgba(255,179,0,0.06);border:1px solid rgba(255,179,0,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.3rem;">
            <span style="color:#FFB300;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                📈 Max Speed per Player
            </span>
        </div>
        """, unsafe_allow_html=True)
        if stats.speed_summary:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            speeds = [s["max_speed_kmh"] for s in stats.speed_summary]
            ids    = [f"#{s['player_id']}" for s in stats.speed_summary]

            fig, ax = plt.subplots(figsize=(6, max(2.5, len(ids) * 0.38)), facecolor="#060b14")
            ax.set_facecolor("#0d1826")
            colors = ["#FFB300" if s == max(speeds) else "#1e3a5f" for s in speeds]
            bars = ax.barh(ids, speeds, color=colors, height=0.6)
            ax.set_xlabel("Max Speed (km/h)", color="#7a92b2", fontsize=9)
            ax.tick_params(colors="#7a92b2", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#0d1826")
            # Annotate bars
            for bar, val in zip(bars, speeds):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}", va="center", color="#e2e8f0", fontsize=8)
            ax.set_title("", color="#00FF87")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Possession donut chart (full width) ────────────────────
    if poss:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,150,255,0.06);border:1px solid rgba(0,150,255,0.2);
                    border-radius:14px;padding:.8rem 1rem .4rem;margin-bottom:.6rem;">
            <span style="color:#0096FF;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;">
                ⚽ Possession Distribution
            </span>
        </div>
        """, unsafe_allow_html=True)
        import pandas as pd, matplotlib
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
            wedgeprops=dict(width=0.55),   # donut style
            textprops={"color": "#e2e8f0", "fontsize": 9},
        )
        for at in autotexts:
            at.set_color("#060b14")
            at.set_fontweight("bold")
        ax.set_facecolor("#060b14")
        plt.tight_layout()
        col_c, col_d, col_e = st.columns([1, 2, 1])
        with col_d:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Display and Download annotated video ────────────────────────────────
    if video_path and Path(video_path).exists():
        st.markdown("---")
        st.markdown("### 📥 Annotated Video")
        with open(video_path, "rb") as f:
            st.download_button(
                "⬇️ Download Annotated Video (MP4)",
                data=f.read(),
                file_name="sports_analyzed.mp4",
                mime="video/mp4",
            )

def _render_video_placeholder() -> None:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1826, #0a1420);
        border: 2px dashed rgba(0,150,255,0.2);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    ">
        <div style="font-size:5rem;margin-bottom:1.2rem;filter:drop-shadow(0 0 20px rgba(0,150,255,0.5));">🎬</div>
        <h3 style="color:#00D4FF;font-family:'Rajdhani',sans-serif;font-size:1.6rem;margin-bottom:.6rem;">Ready for Video Analysis</h3>
        <p style="color:#7a92b2;max-width:480px;margin:0 auto;line-height:1.7;">
            Upload a sports video to enable frame-by-frame player tracking,
            speed & distance measurement, heatmap generation, and ball possession analysis.
        </p>
        <div style="margin-top:2rem;display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;">
            <span style="background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.2);border-radius:20px;padding:.3rem .9rem;color:#00FF87;font-size:.75rem;">🎯 Player Tracking</span>
            <span style="background:rgba(255,179,0,0.08);border:1px solid rgba(255,179,0,0.2);border-radius:20px;padding:.3rem .9rem;color:#FFB300;font-size:.75rem;">⚡ Speed & Distance</span>
            <span style="background:rgba(255,69,96,0.08);border:1px solid rgba(255,69,96,0.2);border-radius:20px;padding:.3rem .9rem;color:#FF4560;font-size:.75rem;">🔥 Heatmaps</span>
            <span style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:20px;padding:.3rem .9rem;color:#0096FF;font-size:.75rem;">🤝 Possession</span>
        </div>
        <p style="color:#2a4a6a;font-size:0.75rem;margin-top:2rem;">Powered by YOLOv8l + ByteTrack · OpenCV · Matplotlib</p>
    </div>
    """, unsafe_allow_html=True)
