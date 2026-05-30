"""
Sports Analytics CV — Model Download Script
Downloads all required model weights into models/weights/.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model yolov8s
"""

import sys
import argparse
from pathlib import Path

# ── Ensure project root is on path ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.file_utils import load_config

logger = get_logger(__name__)


MODELS = {
    "yolov8n": {"file": "yolov8n.pt", "size": "6MB",  "desc": "Nano — fastest, least accurate"},
    "yolov8s": {"file": "yolov8s.pt", "size": "22MB", "desc": "Small — balanced speed/accuracy"},
    "yolov8m": {"file": "yolov8m.pt", "size": "52MB", "desc": "Medium — high accuracy"},
    "yolov8l": {"file": "yolov8l.pt", "size": "87MB", "desc": "Large — best accuracy (recommended with GPU)"},
}


def download_yolo(model_name: str, weights_dir: Path) -> bool:
    """Download a YOLO model weight file."""
    try:
        from ultralytics import YOLO
        import shutil

        local = weights_dir / MODELS[model_name]["file"]
        if local.exists():
            print(f"  [ OK ] {model_name} already downloaded at {local}")
            return True

        print(f"  [DOWN] Downloading {model_name} ({MODELS[model_name]['size']})...")
        model = YOLO(MODELS[model_name]["file"])

        # Try to copy from ultralytics cache to our weights dir
        try:
            ckpt = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
            if ckpt and ckpt.exists():
                shutil.copy2(ckpt, local)
                print(f"  [ OK ] Saved to {local}")
        except Exception:
            print(f"  [ OK ] {model_name} ready (cached by ultralytics)")
        return True

    except Exception as e:
        print(f"  [FAIL] Failed to download {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Specific model to download (default: download yolov8n)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    config = load_config()
    weights_dir = PROJECT_ROOT / config.get("paths", {}).get("models", "models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Sports Analytics CV - Model Downloader ---")
    print("=" * 50)
    print(f"  Weights directory: {weights_dir}\n")

    models_to_download = []
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.model:
        models_to_download = [args.model]
    else:
        models_to_download = ["yolov8l"]  # Default — best accuracy

    success = 0
    for m in models_to_download:
        print(f"\n  [INFO] {m}: {MODELS[m]['desc']}")
        if download_yolo(m, weights_dir):
            success += 1

    print(f"\n{'=' * 50}")
    print(f"  [ OK ] Downloaded {success}/{len(models_to_download)} models")
    print(f"  Ready to run: python run.py\n")


if __name__ == "__main__":
    main()
