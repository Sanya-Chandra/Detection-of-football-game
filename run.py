"""
Sports Analytics CV — CLI Runner
=================================
Command-line launcher that handles environment checks,
dependency verification, and launches the Streamlit app.

Usage:
    python run.py                  # Start Streamlit UI
    python run.py --check          # Environment health check only
    python run.py --download       # Pre-download model weights
    python run.py --image path     # CLI image analysis
    python run.py --video path     # CLI video analysis
"""

import sys
import argparse
import subprocess
from pathlib import Path

# ── Project root setup ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> bool:
    """Verify all required packages are installed."""
    required = [
        ("streamlit", "streamlit"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python-headless"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    print("\n--- Environment Check ---")
    print("=" * 50)
    for module, pkg in required:
        try:
            __import__(module)
            print(f"  [ OK ] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg}  <- run: pip install {pkg}")
            all_ok = False

    # Optional packages
    optional = [
        ("mediapipe", "mediapipe"),
        ("reportlab", "reportlab"),
    ]
    print("\n--- Optional Packages ---")
    for module, pkg in optional:
        try:
            __import__(module)
            print(f"  [ OK ] {pkg}")
        except ImportError:
            print(f"  [WARN] {pkg}  (optional - install for full features)")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: ON - {torch.cuda.get_device_name(0)}")
        else:
            print("\n  GPU: OFF - Not available (CPU mode)")
    except Exception:
        pass

    print("=" * 50)
    if all_ok:
        print("[ OK ] All required packages installed!\n")
    else:
        print("[FAIL] Some packages missing. Run: pip install -r requirements.txt\n")
    return all_ok


def download_models() -> None:
    """Run the model download script."""
    script = PROJECT_ROOT / "scripts" / "download_models.py"
    subprocess.run([sys.executable, str(script)], check=True)


def run_streamlit() -> None:
    """Launch the Streamlit UI and open it in the default web browser without showing a console popup."""
    import subprocess, time, webbrowser, sys

    url = "http://localhost:8501"
    print("\nStarting Sports Analytics CV...")
    print(f"   Opening browser at: {url}\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.maxUploadSize", "500",
        "--server.headless", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#00FF87",
        "--theme.backgroundColor", "#0a0e1a",
        "--theme.secondaryBackgroundColor", "#111827",
        "--theme.textColor", "#e2e8f0",
    ]

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(cmd, creationflags=creationflags)

    # Give the server a moment to start before opening the browser.
    time.sleep(3)
    webbrowser.open(url)

    process.wait()


def analyze_image_cli(image_path: str) -> None:
    """Run image analysis from CLI and save output."""
    from utils.file_utils import load_config, ensure_dirs
    from models.model_manager import ModelManager
    from models.yolo_detector import YOLODetector
    from models.pose_estimator import PoseEstimator
    from processors.image_processor import ImageProcessor

    print(f"\n📸 Analyzing image: {image_path}")
    config = load_config()
    ensure_dirs(config)

    manager = ModelManager(config)
    yolo = manager.get_yolo_model()
    detector = YOLODetector(yolo, config)
    pose = PoseEstimator(config)
    processor = ImageProcessor(detector, pose, config)

    annotated, detections, insights = processor.process(image_path)
    out_path = processor.save_output(annotated)

    print(f"\n📊 Results:")
    for k, v in insights.items():
        print(f"   {k}: {v}")
    print(f"\n✅ Annotated image saved: {out_path}")


def analyze_video_cli(video_path: str) -> None:
    """Run video analysis from CLI."""
    from utils.file_utils import load_config, ensure_dirs
    from models.model_manager import ModelManager
    from models.yolo_detector import YOLODetector
    from processors.video_processor import VideoProcessor

    print(f"\n🎬 Analyzing video: {video_path}")
    config = load_config()
    ensure_dirs(config)

    manager = ModelManager(config)
    yolo = manager.get_yolo_model()
    detector = YOLODetector(yolo, config)

    def progress(cur, total):
        pct = cur / max(total, 1) * 100
        print(f"\r  Progress: {pct:.1f}%  [{cur}/{total}]", end="", flush=True)

    processor = VideoProcessor(detector, config, progress_callback=progress)
    out_path = str(PROJECT_ROOT / "storage" / "videos" / "output" / Path(video_path).name)
    stats, out = processor.process_video(video_path, output_path=out_path)

    print(f"\n\n📊 Results:")
    print(f"   Players: {stats.unique_players_detected}")
    print(f"   Formation: {stats.detected_formation}")
    print(f"   Fastest: {stats.fastest_speed_kmh} km/h")
    if out:
        print(f"   Output: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Sports Analytics CV — CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--check", action="store_true", help="Run environment health check")
    parser.add_argument("--download", action="store_true", help="Pre-download model weights")
    parser.add_argument("--image", type=str, metavar="PATH", help="Analyze a single image")
    parser.add_argument("--video", type=str, metavar="PATH", help="Analyze a video file")
    args = parser.parse_args()

    print("""
    +------------------------------------------+
    |      * Sports Analytics CV v1.0 *        |
    |  AI-powered Player Tracking & Analysis   |
    +------------------------------------------+
    """)

    if args.check:
        check_environment()
        return

    if args.download:
        download_models()
        return

    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        analyze_image_cli(args.image)
        return

    if args.video:
        if not Path(args.video).exists():
            print(f"❌ Video not found: {args.video}")
            sys.exit(1)
        analyze_video_cli(args.video)
        return

    # Default: launch Streamlit UI
    if not check_environment():
        print("⚠️  Some dependencies missing. The app may not function correctly.")
        print("   Install with: pip install -r requirements.txt\n")

    run_streamlit()


if __name__ == "__main__":
    main()
