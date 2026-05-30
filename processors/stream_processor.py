import cv2
import numpy as np
import threading
import time
from typing import Generator, Optional

from processors.video_processor import VideoProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class LiveVideoCapture:
    """
    Threaded OpenCV VideoCapture to eliminate buffer latency and camera lag.
    Always reads the absolute latest frame from the webcam.
    """
    def __init__(self, camera_index: int = 0):
        self.cap = cv2.VideoCapture(camera_index)
        self.ret = False
        self.frame = None
        self.is_running = True
        self.lock = threading.Lock()
        
        # Start a daemon thread to keep reading frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.1)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, propId: int) -> float:
        return self.cap.get(propId)

    def release(self):
        self.is_running = False
        # Give the thread a moment to finish gracefully
        self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()


class StreamProcessor:
    """
    Real-time sports analytics on webcam or RTSP streams.
    Accepts a fully-configured VideoProcessor (with pose, team classifier, GPU model)
    and drives its per-frame pipeline from a live camera.
    """

    def __init__(self, video_processor: VideoProcessor, config: dict):
        """
        Args:
            video_processor: A fully-initialised VideoProcessor (from app.py, with
                             pose estimator + team classifier already attached).
            config          : Application config dict.
        """
        self.processor = video_processor
        self.config = config
        self.is_running = False

    def start_webcam(self, camera_index: int = 0) -> Generator[np.ndarray, None, None]:
        """
        Start processing webcam frames.

        Args:
            camera_index: OpenCV camera index (0 = default webcam)

        Yields:
            Annotated BGR frames
        """
        self.is_running = True
        cap = LiveVideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open webcam at index {camera_index}")
            self.is_running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        # Downsample resolution if it's too large to prevent lagging
        max_dim = 640
        scale = 1.0

        # Save original ppm to avoid cumulative scaling across restarts
        if not hasattr(self, "_original_ppm") and hasattr(self.processor, "speed_est"):
            self._original_ppm = self.processor.speed_est.ppm

        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            target_width = int(width * scale)
            target_height = int(height * scale)
            logger.info(f"Downsampling webcam from {width}x{height} to {target_width}x{target_height} (scale={scale:.2f}) for zero-lag performance.")
            
            # Reset and scale the speed estimator pixels_per_meter to remain accurate!
            if hasattr(self.processor, "speed_est") and hasattr(self, "_original_ppm"):
                self.processor.speed_est.ppm = self._original_ppm * scale
        else:
            target_width = width
            target_height = height
            if hasattr(self.processor, "speed_est") and hasattr(self, "_original_ppm"):
                self.processor.speed_est.ppm = self._original_ppm

        logger.info(f"Webcam stream started: {target_width}×{target_height} @ {fps:.1f} FPS")

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Give the thread a tiny bit of time to capture the first frame
                    time.sleep(0.01)
                    continue
                
                # Perform fast resizing to maintain smooth FPS
                if scale < 1.0:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # Pass current wall-clock timestamp for perfectly accurate speeds; is_live enables GPU fast path
                yield self.processor._process_frame(frame, frame_idx, fps, target_width, target_height, timestamp=time.time(), is_live=True)
                frame_idx += 1
        finally:
            cap.release()
            self.is_running = False
            logger.info("Webcam stream stopped")

    def stop(self) -> None:
        """Signal the stream to stop."""
        self.is_running = False
