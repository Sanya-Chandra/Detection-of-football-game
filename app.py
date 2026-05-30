"""
Sports Analytics CV — Main Streamlit Application
=================================================
Entry point for the sports analytics dashboard.

Run with:
    streamlit run app.py

Or via:
    python run.py
"""

import sys
from pathlib import Path

# ── Ensure project root is on Python path ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.logger import get_logger
from utils.file_utils import load_config, ensure_dirs
from models.model_manager import ModelManager
from models.yolo_detector import YOLODetector
from models.pose_estimator import PoseEstimator, YoloPoseEstimator
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from ui.components import apply_custom_css, hero_banner, render_sidebar
from ui.image_tab import render_image_tab
from ui.video_tab import render_video_tab
from ui.stats_tab import render_stats_tab

logger = get_logger(__name__)

# ── Page configuration (must be first Streamlit call) ──────────
st.set_page_config(
    page_title="Sports Analytics CV",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/sports-analytics-cv",
        "Report a bug": None,
        "About": "# Sports Analytics CV\nAI-powered sports analysis using YOLOv8 and Computer Vision.",
    },
)


# ── Cached resource initialization ─────────────────────────────

@st.cache_resource(show_spinner="⚙️ Loading AI models (first run may take a moment)...")
def init_models(config: dict):
    """
    Initialize all AI models.  Cached across Streamlit sessions.
    Downloads YOLOv8 weights automatically on first run.
    """
    logger.info("Initializing AI models...")
    manager = ModelManager(config)
    yolo = manager.get_yolo_model()         # yolov8l — full accuracy for file analysis
    detector = YOLODetector(yolo, config)
    pose = PoseEstimator(config)
    logger.info("Models ready")
    return manager, detector, pose


@st.cache_resource(show_spinner="⚡ Loading live-stream model (yolov8s + FP16)...")
def init_live_models(config: dict):
    """
    Initialize a smaller, faster YOLO model for real-time streaming.
    Uses yolov8s with FP16 half-precision on CUDA for maximum throughput.
    Cached separately so it doesn't block the main model initialization.
    """
    logger.info("Initializing live-stream model...")
    manager = ModelManager(config)
    live_yolo = manager.get_live_yolo_model()  # yolov8s — optimised for live stream
    live_detector = YOLODetector(live_yolo, config)
    logger.info("Live-stream model ready")
    return live_detector


@st.cache_resource(show_spinner="🦴 Loading pose model (yolov8s-pose + FP16)...")
def init_pose_models(config: dict):
    """
    Initialize YOLOv8-Pose for GPU-native skeleton estimation.
    Loads yolov8s-pose.pt with FP16 on CUDA.
    Returns None gracefully if loading fails.
    """
    try:
        logger.info("Initializing YOLOv8-Pose model...")
        manager = ModelManager(config)
        pose_model = manager.get_pose_model()
        pose_est = YoloPoseEstimator(pose_model, config)
        logger.info("YOLOv8-Pose model ready")
        return pose_est
    except Exception as e:
        logger.warning(f"YOLOv8-Pose init failed (skipping pose): {e}")
        return None


@st.cache_data(show_spinner=False)
def _load_config() -> dict:
    return load_config()


# ── Main Application ────────────────────────────────────────────

def main() -> None:
    # Apply global CSS theme
    apply_custom_css()

    # Load configuration
    config = _load_config()
    ensure_dirs(config)

    # Hero banner
    hero_banner()

    # Premium sidebar
    render_sidebar(config)


    # ── Initialize models (cached) ─────────────────────────────

    try:
        manager, detector, pose = init_models(config)
        live_detector = init_live_models(config)
        pose_est = init_pose_models(config)
    except Exception as e:
        st.error(
            f"❌ Failed to load AI models: {e}\n\n"
            "Please run `python scripts/download_models.py` to pre-download weights, "
            "then restart the app."
        )
        logger.exception("Model initialization failed")
        st.stop()

    # ── Build processors ────────────────────────────────────────────────
    image_processor = ImageProcessor(detector, pose_est, config)
    video_processor = VideoProcessor(live_detector, config, pose_estimator=pose_est)  # yolov8s + FP16 + pose

    # ── Tabs ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Image Analysis",
        "🎬 Video Analytics",
        "📊 Statistics & Export",
        "ℹ️ About",
    ])

    with tab1:
        render_image_tab(image_processor, config)

    with tab2:
        render_video_tab(video_processor, config, premium_detector=detector)

    with tab3:
        render_stats_tab(config)

    with tab4:
        _render_about_tab(manager)


def _render_about_tab(manager) -> None:
    """Render the About / System Info tab."""
    st.markdown("## ℹ️ About Sports Analytics CV")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        Sports Analytics CV is an end-to-end AI Computer Vision system designed
        to analyze player movements, formations, and tactics from sports media.

        #### Core Features
        - **Player Detection** — YOLOv8 multi-class detection
        - **Multi-Object Tracking** — ByteTrack persistent IDs
        - **Speed Estimation** — Pixel-to-meter calibrated tracking
        - **Heatmaps** — Gaussian-smoothed position density maps
        - **Formation Analysis** — K-Means clustering based
        - **Possession Tracking** — Proximity-based attribution
        - **Pose Estimation** — MediaPipe body landmarks
        - **PDF/CSV Export** — Professional analytics reports

        #### Supported Sports
        Football ⚽ · Basketball 🏀 · Tennis 🎾 · Rugby 🏉 · Hockey 🏑
        """)

    with col2:
        st.markdown("### 🔧 System Information")
        try:
            import torch
            import cv2
            import ultralytics

            st.markdown(f"""
            | Component | Version/Status |
            |---|---|
            | Python | {sys.version.split()[0]} |
            | PyTorch | {torch.__version__} |
            | CUDA | {'✅ ' + torch.version.cuda if torch.cuda.is_available() else '❌ N/A'} |
            | OpenCV | {cv2.__version__} |
            | Ultralytics | {ultralytics.__version__} |
            | Device | {manager.device} |
            | Weights Dir | `{manager.weights_dir}` |
            """)
        except Exception as e:
            st.warning(f"Could not retrieve full system info: {e}")

        st.markdown("### 🏗️ Architecture")
        st.code("""
sports_analytics_cv/
├── app.py              # Streamlit entry point
├── models/             # YOLOv8, MediaPipe wrappers
├── analytics/          # Heatmap, speed, formation, possession
├── processors/         # Image, video, stream pipelines
├── ui/                 # Streamlit components & tabs
├── utils/              # Logging, drawing, file helpers
├── scripts/            # Setup & benchmark tools
└── storage/            # Sample images & videos
        """, language="text")


if __name__ == "__main__":
    main()
