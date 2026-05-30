# Dockerfile for Sports Analytics CV
# Build:  docker build -t sports-analytics-cv .
# Run:    docker run -p 8501:8501 sports-analytics-cv

FROM python:3.11-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories and generate sample data
RUN mkdir -p uploads outputs/annotated_images outputs/annotated_videos \
             outputs/heatmaps outputs/reports models/weights logs && \
    python scripts/generate_sample_data.py && \
    python scripts/download_models.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.maxUploadSize=200"]
