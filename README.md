# Sports Analytics CV

<div align="center">

![Sports Analytics CV](https://img.shields.io/badge/AI-Computer%20Vision-00FF87?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-0096FF?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF6B35?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-FFD700?style=for-the-badge&logo=python)

**An end-to-end AI Computer Vision system for real-time sports analytics.**  
Detect players, track movements, analyze formations, and generate tactical insights from sports images and videos.

</div>

---

## ✨ Features

| Feature | Technology | Description |
|---|---|---|
| 🎯 Player Detection | YOLOv8 | Detect players and ball with bounding boxes |
| 📡 Multi-Object Tracking | ByteTrack | Persistent player IDs across video frames |
| 🔥 Heatmaps | OpenCV + Matplotlib | Player position density visualization |
| 🏃 Speed Estimation | Pixel Calibration | Real-time speed in km/h per player |
| 📐 Formation Analysis | K-Means Clustering | Detect 4-4-2, 4-3-3, etc. formations |
| ⚽ Possession Tracking | Proximity Analysis | Ball possession % per player |
| 🦴 Pose Estimation | MediaPipe | Body landmark detection |
| 📊 Analytics Dashboard | Streamlit | Live stats, charts, and metrics |
| 📄 Export Reports | ReportLab | PDF + CSV report generation |
| 🔴 Live Webcam | OpenCV | Real-time webcam analytics |

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Linux / Mac:**
```bash
chmod +x setup.sh && ./setup.sh
```

Then start the app:
```bash
python run.py or run_gui
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data
python scripts/generate_sample_data.py

# 4. Download AI models (auto on first run, or pre-download)
python scripts/download_models.py

# 5. Launch the app
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🗂️ Project Structure

```
sports_analytics_cv/
│
├── app.py                    # Main Streamlit application
├── run.py                    # CLI launcher & environment checker
├── config.yaml               # Global configuration
├── requirements.txt          # Python dependencies
├── setup.bat / setup.sh      # Automated setup scripts
├── .env.example              # Environment variables template
├── Dockerfile                # Docker containerization
│
├── models/                   # AI model wrappers
│   ├── model_manager.py      # Auto-download & device management
│   ├── yolo_detector.py      # YOLOv8 detection + tracking
│   ├── pose_estimator.py     # MediaPipe pose estimation
│   └── weights/              # Downloaded model files (auto-created)
│
├── analytics/                # Sports analytics engine
│   ├── heatmap.py            # Player position heatmaps
│   ├── speed_estimator.py    # Speed & distance estimation
│   ├── formation_analyzer.py # K-Means formation detection
│   ├── possession_tracker.py # Ball possession analysis
│   ├── statistics.py         # Match statistics aggregation
│   └── report_generator.py   # PDF & CSV export
│
├── processors/               # Media processing pipelines
│   ├── image_processor.py    # Static image analysis
│   ├── video_processor.py    # Video frame-by-frame analysis
│   └── stream_processor.py   # Live webcam/stream processing
│
├── ui/                       # Streamlit UI components
│   ├── components.py         # Shared widgets & CSS theme
│   ├── image_tab.py          # Image analysis tab
│   ├── video_tab.py          # Video analytics tab
│   └── stats_tab.py          # Statistics & export tab
│
├── utils/                    # Utilities
│   ├── logger.py             # Centralized logging
│   ├── drawing.py            # Frame annotation functions
│   └── file_utils.py         # File I/O & config loading
│
├── scripts/                  # Standalone scripts
│   ├── download_models.py    # Pre-download model weights
│   ├── generate_sample_data.py # Create test images & videos
│   └── benchmark.py          # Performance benchmarking
│
├── dataset/                  # Sample data
│   ├── sample_images/        # Test sports images
│   └── sample_videos/        # Test sports video
│
├── uploads/                  # User-uploaded files (runtime)
├── outputs/                  # Processed outputs (runtime)
│   ├── annotated_images/
│   ├── annotated_videos/
│   ├── heatmaps/
│   └── reports/
└── logs/                     # Application logs
```

---

## 🎮 Usage Guide

### Image Analysis
1. Go to **📸 Image Analysis** tab
2. Upload a sports image (JPG/PNG) or select a sample
3. Adjust confidence threshold in the sidebar
4. View detected players, ball, formation inference
5. Download the annotated image

### Video Analytics
1. Go to **🎬 Video Analytics** tab
2. Upload a sports video (MP4/AVI) or select a sample
3. Set max frames to analyze (higher = slower but more complete)
4. Click **Analyze Video**
5. View player trajectories, heatmap, speed charts

### Live Webcam
1. Go to **🎬 Video Analytics** tab
2. Enable **🔴 Use webcam** checkbox
3. Click **Start Webcam** for real-time tracking

### Export Reports
1. Run video analysis first
2. Go to **📊 Statistics & Export** tab
3. Download **PDF Report**, **CSV Data**, or **JSON**

---

## ⚙️ Configuration

Edit `config.yaml` to customize:
```yaml
detection:
  model: "yolov8n.pt"          # nano (fast) or yolov8s.pt (accurate)
  confidence_threshold: 0.45

analytics:
  pixels_per_meter: 10.0       # Calibrate for speed accuracy
  heatmap_sigma: 20            # Heatmap smoothing

video:
  max_frames: 500              # Limit frames processed
  show_trajectories: true
```

---

## 🚢 Deployment

### Local
```bash
python run.py
```

### Streamlit Cloud
1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set entry point: `app.py`

### HuggingFace Spaces
1. Create a new Space (SDK: Streamlit)
2. Upload project files
3. App auto-deploys

### Docker
```bash
docker build -t sports-analytics-cv .
docker run -p 8501:8501 sports-analytics-cv
```

### Render / Railway
- Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Build command: `pip install -r requirements.txt && python scripts/generate_sample_data.py`

### Google Colab
```python
!pip install -r requirements.txt
!python scripts/generate_sample_data.py
!streamlit run app.py &
# Then use ngrok or localtunnel to expose port 8501
```

---

## 🔧 CLI Reference

```bash
python run.py                       # Launch Streamlit UI
python run.py --check               # Environment health check
python run.py --download            # Pre-download model weights
python run.py --image path/to/img   # CLI image analysis
python run.py --video path/to/vid   # CLI video analysis
python scripts/benchmark.py         # Performance benchmark
```

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | ≥8.1.0 | YOLOv8 + ByteTrack |
| `torch` | ≥2.0.0 | Deep learning backend |
| `opencv-python-headless` | ≥4.9 | Image/video processing |
| `mediapipe` | ≥0.10 | Pose estimation |
| `streamlit` | ≥1.32 | Web UI |
| `scikit-learn` | ≥1.3 | K-Means formation |
| `scipy` | ≥1.11 | Gaussian heatmaps |
| `reportlab` | ≥4.0 | PDF reports |
| `pandas` | ≥2.0 | Data analysis |

---

## 🧪 Environment Check

```bash
python run.py --check
```

Expected output:
```
✅ streamlit
✅ ultralytics
✅ cv2
✅ torch
✅ numpy
...
✅ All required packages installed!
```

---

## 📜 License

MIT License — Free for academic, research, and portfolio use.

---

## 🏗️ Built With

- **YOLOv8** by Ultralytics — State-of-the-art object detection
- **ByteTrack** — Multi-object tracking algorithm
- **MediaPipe** by Google — Human pose estimation
- **Streamlit** — Interactive data science UI
- **OpenCV** — Computer vision primitives
- **ReportLab** — PDF generation
- **PyTorch** — Neural network inference

---

<div align="center">
⚽ Built for Final Year Projects · Research · Portfolio · Deployment
</div>
