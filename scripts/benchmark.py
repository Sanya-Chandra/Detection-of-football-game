"""
Sports Analytics CV — Performance Benchmark
Measures detection speed (FPS) and memory usage.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --frames 100 --model yolov8n
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.file_utils import load_config
from models.model_manager import ModelManager
from models.yolo_detector import YOLODetector


def benchmark(model_name: str = "yolov8n.pt", num_frames: int = 50) -> None:
    config = load_config()
    config["detection"]["model"] = model_name

    print(f"\n⏱  Sports Analytics CV — Benchmark")
    print("=" * 50)
    print(f"  Model: {model_name}")
    print(f"  Frames: {num_frames}")

    manager = ModelManager(config)
    yolo = manager.get_yolo_model(model_name)
    detector = YOLODetector(yolo, config)

    print(f"  Device: {manager.device}")

    # Warm up
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        detector.detect(dummy)

    # Benchmark detection
    times = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_frames}] avg: {1/np.mean(times[-10:]):.1f} FPS")

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    fps = 1.0 / np.mean(times)

    print("\n📊 Results:")
    print(f"  Average latency:  {avg_ms:.1f}ms")
    print(f"  P95 latency:      {p95_ms:.1f}ms")
    print(f"  Throughput:       {fps:.1f} FPS")

    # Memory (if psutil available)
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory usage:     {mem_mb:.0f}MB")
    except ImportError:
        pass

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark detection speed")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.model, args.frames)
