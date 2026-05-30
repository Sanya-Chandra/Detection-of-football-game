"""
Sports Analytics CV — Image Analysis Tab
Streamlit UI for static sports image analysis.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Optional

from ui.components import render_stat_row, loading_spinner
import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


def render_image_tab(image_processor, config: dict) -> None:
    st.markdown("## 📸 Image Analysis")
    st.markdown("Upload a sports image to detect players, track the ball, and get real-time analytics.")

    # ── Sidebar: sample images only (dataset removed) ──────────
    with st.sidebar:
        st.markdown("### 🖼 Sample Images")
        sample_dir = Path("storage")
        samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.webp"))
        if samples:
            selected = st.selectbox(
                "Use a sample image",
                ["— Upload your own —"] + [s.name for s in samples],
            )
        else:
            selected = "— Upload your own —"

    # ── File upload + settings ──────────────────────────────────
    col_up, col_opts = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop a sports image (JPG / PNG / WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_uploader",
            help="Supports JPEG, PNG and WEBP formats",
        )
    with col_opts:
        st.markdown("**Detection Settings**")
        conf_thresh = st.slider(
            "Confidence threshold", 0.05, 0.9, 0.20, 0.05, key="img_conf",
            help="Lower = detect more objects (including faint balls). Default: 0.20"
        )
        show_grid = st.checkbox("Show field grid", value=True, key="img_grid")

    # Determine source image
    image_path: Optional[Path] = None
    if uploaded:
        from utils.file_utils import save_uploaded_file
        image_path = save_uploaded_file(uploaded.read(), uploaded.name)
    elif selected and selected != "— Upload your own —":
        image_path = sample_dir / selected

    # ── Analyze ─────────────────────────────────────────────────
    if image_path:
        try:
            image_processor.detector.conf_thresh = conf_thresh

            # Initialize display size in session state if not set
            if "img_display_size" not in st.session_state:
                st.session_state.img_display_size = 80

            # Premium zoom controls with + / - buttons
            st.markdown("### 🔍 Image Zoom")
            col_zoom, _ = st.columns([2, 3])
            with col_zoom:
                col_dec, col_val, col_inc = st.columns([1, 2, 1])
                with col_dec:
                    if st.button("➖ Zoom Out", key="zoom_dec", use_container_width=True):
                        st.session_state.img_display_size = max(20, st.session_state.img_display_size - 10)
                        st.rerun()
                with col_val:
                    st.markdown(
                        f"<div style='text-align:center;font-size:1.1rem;font-weight:700;"
                        f"background:rgba(0,255,135,0.06);border:1px solid rgba(0,255,135,0.15);"
                        f"border-radius:10px;padding:0.35rem 0;color:#00FF87;'>{st.session_state.img_display_size}%</div>",
                        unsafe_allow_html=True
                    )
                with col_inc:
                    if st.button("➕ Zoom In", key="zoom_inc", use_container_width=True):
                        st.session_state.img_display_size = min(200, st.session_state.img_display_size + 10)
                        st.rerun()

            size_percent = st.session_state.img_display_size

            with loading_spinner("🔍 Detecting players and ball..."):
                annotated, detections, insights = image_processor.process(str(image_path))
                # ── Save annotated image automatically to storage/images/output ──
                if annotated is not None:
                    from utils.file_utils import get_output_path
                    out_name = f"{Path(image_path).stem}_annotated.jpg"
                    out_path = get_output_path("annotated_images", out_name)
                    cv2.imwrite(str(out_path), annotated)

            col_preview, col_annotated = st.columns(2)

            with col_preview:
                st.markdown("**Original Image**")
                img = cv2.imread(str(image_path))
                if img is not None:
                    h = int(img.shape[0] * size_percent / 100)
                    w = int(img.shape[1] * size_percent / 100)
                    st.image(cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB), caption="Original")
                else:
                    st.image(str(image_path), use_container_width=True)

            with col_annotated:
                st.markdown("**Detected Players & Ball**")
                if annotated is not None:
                    h_a = int(annotated.shape[0] * size_percent / 100)
                    w_a = int(annotated.shape[1] * size_percent / 100)
                    st.image(cv2.cvtColor(cv2.resize(annotated, (w_a, h_a)), cv2.COLOR_BGR2RGB), caption="AI Annotated")

            st.markdown("---")

            # ── Key Metrics Row ────────────────────────────────────
            st.markdown("### 🏆 Detection Results")
            ball_status = "✅ YES" if insights["ball_detected"] else "❌ NO"
            ball_conf = f"{insights['ball_confidence']:.0%}" if insights.get("ball_confidence") else "—"
            render_stat_row(
                {
                    "Players Detected":    str(insights["total_players"]),
                    "Ball Detected":       ball_status,
                    "Ball Confidence":     ball_conf,
                    "Avg Player Conf":     f"{insights['avg_detection_confidence']:.0%}",
                },
                variants={
                    "Ball Detected":   "ball",
                    "Ball Confidence": "ball",
                }
            )

            st.markdown("")

            # ── Zone breakdown + Ball info ─────────────────────────
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 🏃 Player Zone Distribution")
                st.markdown(f"""
                <div style="background:#0d1826;border:1px solid rgba(0,255,135,0.15);border-radius:14px;padding:1.2rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:.6rem;">
                        <span style="color:#7a92b2;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;">Defensive Third</span>
                        <span style="color:#00FF87;font-weight:700;">{insights['players_in_defense']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:.6rem;">
                        <span style="color:#7a92b2;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;">Middle Third</span>
                        <span style="color:#FFB300;font-weight:700;">{insights['players_in_midfield']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7a92b2;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;">Attacking Third</span>
                        <span style="color:#FF4560;font-weight:700;">{insights['players_in_attack']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown("#### ⚽ Ball Intelligence")
                if insights["ball_detected"]:
                    zone = insights.get("ball_zone", "Unknown")
                    zone_color = {"Defensive Third": "#00FF87", "Middle Third": "#FFB300", "Attacking Third": "#FF4560"}.get(zone, "#7a92b2")
                    st.markdown(f"""
                    <div style="background:#0d1826;border:1px solid rgba(255,69,96,0.25);border-radius:14px;padding:1.2rem;">
                        <div style="color:#FF4560;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem;">Ball Location</div>
                        <div style="font-size:1.4rem;font-weight:700;color:{zone_color};font-family:'Rajdhani',sans-serif;">{zone}</div>
                        <div style="margin-top:.8rem;color:#7a92b2;font-size:.8rem;">
                            Confidence: <span style="color:#e2e8f0;font-weight:600;">{insights['ball_confidence']:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#0d1826;border:1px solid rgba(255,69,96,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
                        <div style="font-size:2rem;margin-bottom:.5rem;">⚽</div>
                        <div style="color:#7a92b2;font-size:.9rem;">Ball not detected in this frame.<br>Try lowering the confidence threshold.</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Download ───────────────────────────────────────────
            st.markdown("")
            ret, buf = cv2.imencode(".jpg", annotated)
            if ret:
                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=buf.tobytes(),
                    file_name="sports_annotated.jpg",
                    mime="image/jpeg",
                )
        except Exception as e:
            st.error(f"🚨 An unexpected error occurred during image analysis: {e}")
            logger.exception("Global Image Tab Error")
    else:
        st.info("👆 Upload a sports image or select a sample from the sidebar to begin analysis.")
        _render_demo_placeholder()


def _render_demo_placeholder() -> None:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1826, #0a1420);
        border: 2px dashed rgba(0,255,135,0.2);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    ">
        <div style="font-size:5rem;margin-bottom:1.2rem;filter:drop-shadow(0 0 20px rgba(0,255,135,0.4));">⚽</div>
        <h3 style="color:#00FF87;font-family:'Rajdhani',sans-serif;font-size:1.6rem;margin-bottom:.6rem;">Ready for Analysis</h3>
        <p style="color:#7a92b2;max-width:420px;margin:0 auto;line-height:1.7;">
            Upload any sports image to instantly detect players, locate the ball,
            and get zone-by-zone player distribution analytics.
        </p>
        <div style="margin-top:2rem;display:flex;gap:.8rem;justify-content:center;flex-wrap:wrap;">
            <span style="background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.2);border-radius:20px;padding:.3rem .9rem;color:#00FF87;font-size:.75rem;">🎯 Player Detection</span>
            <span style="background:rgba(255,69,96,0.08);border:1px solid rgba(255,69,96,0.2);border-radius:20px;padding:.3rem .9rem;color:#FF4560;font-size:.75rem;">⚽ Ball Tracking</span>
            <span style="background:rgba(255,179,0,0.08);border:1px solid rgba(255,179,0,0.2);border-radius:20px;padding:.3rem .9rem;color:#FFB300;font-size:.75rem;">🏃 Zone Analysis</span>
        </div>
        <p style="color:#2a4a6a;font-size:0.75rem;margin-top:2rem;">Powered by YOLOv8l · OpenCV · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
