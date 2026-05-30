"""
Sports Analytics CV — File Utilities
Helpers for path resolution, file management, and config loading.
"""

import shutil
import yaml
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Root of the project — resolved relative to this file's parent's parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to config.yaml relative to project root

    Returns:
        Parsed configuration dictionary
    """
    cfg_file = resolve_path(config_path)
    if not cfg_file.exists():
        logger.warning(f"Config file not found at {cfg_file}. Using empty config.")
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Config loaded from {cfg_file}")
    return cfg or {}


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create all required output/upload directories from config."""
    path_keys = config.get("paths", {})
    for key, rel_path in path_keys.items():
        full = resolve_path(rel_path)
        full.mkdir(parents=True, exist_ok=True)


def get_file_category(filename: str) -> str:
    """
    Determine the category of a file based on its extension.
    Maps to 'image', 'video', 'pdf', or 'csv'.
    """
    ext = Path(filename).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        return "image"
    elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return "video"
    elif ext in [".pdf"]:
        return "pdf"
    elif ext in [".csv"]:
        return "csv"
    else:
        return "other"


def save_uploaded_file(data: bytes, filename: str, upload_dir: Optional[str] = None) -> Path:
    """
    Save raw bytes from a Streamlit upload to the workspace storage folders:
    - storage/images/input/ for images
    - storage/videos/input/ for videos
    """
    category = get_file_category(filename)
    if category == "image":
        dest_dir = resolve_path("storage/images/input")
    elif category == "video":
        dest_dir = resolve_path("storage/videos/input")
    else:
        dest_dir = resolve_path("storage/other/input")

    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    logger.info(f"Saved upload: {dest} ({len(data)/1024:.1f} KB)")
    return dest


def get_output_path(subdir: str, filename: str) -> Path:
    """
    Generate output path under our workspace storage folders:
    - storage/images/output/ for images and heatmaps
    - storage/videos/output/ for videos
    - storage/reports/ for pdf and csv exports
    """
    category = get_file_category(filename)

    if category == "image" or subdir in ["annotated_images", "heatmaps", "image"]:
        out = resolve_path("storage/images/output")
    elif category == "video" or subdir in ["annotated_videos", "video"]:
        out = resolve_path("storage/videos/output")
    elif category == "pdf" or category == "csv" or subdir in ["reports", "pdf", "csv"]:
        out = resolve_path("storage/reports")
    else:
        out = resolve_path("storage/other/output")

    out.mkdir(parents=True, exist_ok=True)
    return out / filename


def file_md5(path: str) -> str:
    """Return the MD5 hash of a file (for integrity checks)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clear_directory(dir_path: str) -> None:
    """Remove all files inside a directory without deleting the directory itself."""
    p = resolve_path(dir_path)
    if p.exists():
        for item in p.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    logger.debug(f"Cleared directory: {p}")
