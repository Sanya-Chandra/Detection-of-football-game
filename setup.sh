#!/bin/bash
# ============================================================
# Sports Analytics CV — Linux/Mac Setup Script
# ============================================================
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo ""
echo " ╔═══════════════════════════════════════════╗"
echo " ║     ⚽ Sports Analytics CV — Setup        ║"
echo " ╚═══════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo " ❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi
echo " ✅ $(python3 --version) detected"

# ── Create virtual environment ────────────────────────────────
if [ ! -f "venv/bin/python3" ]; then
    echo " 📦 Creating fresh virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo " ❌ Failed to create virtual environment."
        echo "    If 'venv' folder exists, please delete it manually and re-run."
        exit 1
    fi
    echo " ✅ Virtual environment created"
else
    echo " ✅ Virtual environment exists"
fi

# ── Activate ──────────────────────────────────────────────────
source venv/bin/activate

# ── Upgrade pip ───────────────────────────────────────────────
echo " 📦 Upgrading pip..."
pip install --upgrade pip --quiet

# ── Install dependencies ──────────────────────────────────────
echo " 📦 Installing dependencies..."
pip install -r requirements.txt

# ── Install PyTorch ───────────────────────────────────────────
echo " 📦 Setting up PyTorch..."
pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

if command -v nvidia-smi &> /dev/null; then
    echo " 🟩 NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo " ⬜ No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── Directories ───────────────────────────────────────────────
echo " 📁 Creating output directories..."
mkdir -p storage/images/input storage/images/output storage/videos/input storage/videos/output \
         storage/heatmaps storage/reports models/weights logs
echo " ✅ Directories ready"

# ── Verify PyTorch / CUDA ─────────────────────────────────────
echo " 🔍 Verifying PyTorch and GPU access..."
python -c "import torch; print(f'  [INFO] PyTorch version: {torch.__version__}'); print(f'  [INFO] CUDA Available: {torch.cuda.is_available()}')"

# ── Download models ───────────────────────────────────────────
echo " 🤖 Downloading AI model weights..."
python scripts/download_models.py

echo ""
echo " ╔═══════════════════════════════════════════╗"
echo " ║  ✅ Setup Complete!                       ║"
echo " ║                                           ║"
echo " ║  Start the app:  python run.py            ║"
echo " ║  Or directly:    streamlit run app.py     ║"
echo " ╚═══════════════════════════════════════════╝"
echo ""
