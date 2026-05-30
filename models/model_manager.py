"""
Sports Analytics CV — Model Manager
Auto-downloads and caches YOLOv8 model weights into models/weights/.
Provides GPU/CPU device detection and model lifecycle management.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from utils.logger import get_logger
from utils.file_utils import load_config, resolve_path

logger = get_logger(__name__)


class ModelManager:
    """
    Handles model download, caching, and device management.

    On first run the YOLOv8 weights are automatically downloaded by the
    ultralytics library.  We redirect the cache directory to our local
    models/weights/ folder so everything stays inside the project.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.weights_dir = resolve_path(
            self.config.get("paths", {}).get("models", "models/weights")
        )
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # Redirect ultralytics / torch hub cache into our project folder
        os.environ["YOLO_CONFIG_DIR"] = str(self.weights_dir)
        os.environ["TORCH_HOME"] = str(self.weights_dir)

        self.device = self._select_device()
        logger.info(f"ModelManager ready | device={self.device} | weights_dir={self.weights_dir}")

    # ──────────────────────────────────────────────────────────
    # Device selection
    # ──────────────────────────────────────────────────────────

    def _select_device(self) -> str:
        """
        Auto-select the best available compute device.
        Priority: CUDA → MPS (Apple Silicon) → CPU
        """
        preference = self.config.get("detection", {}).get("device", "auto")
        if preference != "auto":
            return preference

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device detected: {gpu}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected")
            return "mps"
        logger.warning("No GPU detected — using CPU. Install CUDA PyTorch for GPU acceleration.")
        return "cpu"

    @property
    def use_fp16(self) -> bool:
        """True when FP16 half-precision should be used (CUDA only)."""
        cfg_fp16 = self.config.get("detection", {}).get("fp16", True)
        return cfg_fp16 and self.device == "cuda"

    # ──────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────

    def get_yolo_model(self, model_name: Optional[str] = None):
        """
        Load and return a YOLOv8 model, downloading weights if necessary.

        Args:
            model_name: e.g. 'yolov8n.pt', 'yolov8s.pt'. Defaults to config value.

        Returns:
            ultralytics.YOLO instance
        """
        from ultralytics import YOLO  # Import here to avoid slow startup if not needed

        if model_name is None:
            model_name = self.config.get("detection", {}).get("model", "yolov8n.pt")

        local_path = self.weights_dir / model_name

        if local_path.exists():
            logger.info(f"Loading cached YOLO model: {local_path}")
            model = YOLO(str(local_path))
        else:
            logger.info(f"Downloading YOLO model: {model_name} → {local_path}")
            # ultralytics downloads to its default cache; we then copy to our dir
            model = YOLO(model_name)
            try:
                import shutil
                src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
                if src and src.exists() and src != local_path:
                    shutil.copy2(src, local_path)
                    logger.info(f"Copied weights to {local_path}")
            except Exception as e:
                logger.warning(f"Could not copy weights to local dir: {e}")

        model.to(self.device)
        if self.use_fp16:
            model.half()
            logger.info(f"YOLO model ready on {self.device} (FP16 enabled)")
        else:
            logger.info(f"YOLO model ready on {self.device}")
        return model

    def get_live_yolo_model(self):
        """
        Load a smaller, faster YOLO model optimised for real-time streaming.
        Uses yolov8s by default (configurable via detection.live_model).
        FP16 half-precision is enabled automatically on CUDA for max throughput.
        """
        from ultralytics import YOLO

        model_name = self.config.get("detection", {}).get("live_model", "yolov8s.pt")
        local_path = self.weights_dir / model_name

        if local_path.exists():
            logger.info(f"Loading cached live YOLO model: {local_path}")
            model = YOLO(str(local_path))
        else:
            logger.info(f"Downloading live YOLO model: {model_name} → {local_path}")
            model = YOLO(model_name)
            try:
                import shutil
                src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
                if src and src.exists() and src != local_path:
                    shutil.copy2(src, local_path)
                    logger.info(f"Copied weights to {local_path}")
            except Exception as e:
                logger.warning(f"Could not copy weights to local dir: {e}")

        model.to(self.device)
        if self.use_fp16:
            model.half()
            logger.info(f"Live YOLO model ready on {self.device} (FP16 enabled — max throughput)")
        else:
            logger.info(f"Live YOLO model ready on {self.device}")
        return model

    def get_pose_model(self):
        """
        Load YOLOv8-Pose model for GPU-native skeleton estimation.
        Uses yolov8s-pose.pt — good balance of speed and accuracy on RTX 4060.
        FP16 is enabled automatically on CUDA for maximum throughput.
        """
        from ultralytics import YOLO

        model_name = self.config.get("pose", {}).get("yolo_model", "yolov8s-pose.pt")
        local_path = self.weights_dir / model_name

        if local_path.exists():
            logger.info(f"Loading cached pose model: {local_path}")
            model = YOLO(str(local_path))
        else:
            logger.info(f"Downloading pose model: {model_name} → {local_path}")
            model = YOLO(model_name)
            try:
                import shutil
                src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
                if src and src.exists() and src != local_path:
                    shutil.copy2(src, local_path)
                    logger.info(f"Copied pose weights to {local_path}")
            except Exception as e:
                logger.warning(f"Could not copy pose weights to local dir: {e}")

        model.to(self.device)
        if self.use_fp16:
            model.half()
            logger.info(f"Pose model ready on {self.device} (FP16 enabled)")
        else:
            logger.info(f"Pose model ready on {self.device}")
        return model

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")

    def __repr__(self) -> str:
        return f"ModelManager(device={self.device}, weights_dir={self.weights_dir})"
