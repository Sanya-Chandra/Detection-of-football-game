@echo off
REM ============================================================
REM Sports Analytics CV — Windows Setup Script
REM ============================================================
REM Run this script once to set up the project environment.
REM Usage: setup.bat

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║     ⚽ Sports Analytics CV — Setup        ║
echo  ╚═══════════════════════════════════════════╝
echo.

REM ── Check Python ─────────────────────────────────────────
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  ❌ Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
echo  ✅ Python detected

REM ── Check / Create virtual environment ────────────────────
IF NOT EXIST "venv\Scripts\python.exe" (
    echo  📦 Creating fresh virtual environment...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo  ❌ Failed to create virtual environment.
        echo     If "venv" folder exists, please delete it manually and re-run.
        pause
        exit /b 1
    )
    echo  ✅ Virtual environment created
) ELSE (
    echo  ✅ Virtual environment exists
)

REM ── Activate virtual environment ──────────────────────────
call venv\Scripts\activate.bat

REM ── Upgrade pip ───────────────────────────────────────────
echo  📦 Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── Install dependencies ─────────────────────────────────
echo  📦 Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

IF ERRORLEVEL 1 (
    echo  ❌ Dependency installation failed
    pause
    exit /b 1
)
echo  ✅ All dependencies installed

REM ── Install PyTorch (CUDA/CPU) ────────────────────────────
echo  📦 Setting up PyTorch...
pip uninstall -y torch torchvision torchaudio >nul 2>&1

nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo  🟩 NVIDIA GPU detected. Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) ELSE (
    echo  ⬜ No NVIDIA GPU detected. Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM ── Create required directories ───────────────────────────
echo  📁 Creating output directories...
mkdir storage\images\input 2>nul
mkdir storage\images\output 2>nul
mkdir storage\videos\input 2>nul
mkdir storage\videos\output 2>nul
mkdir storage\heatmaps 2>nul
mkdir storage\reports 2>nul
mkdir models\weights 2>nul
mkdir logs 2>nul
echo  ✅ Directories ready

REM ── Verify PyTorch / CUDA ─────────────────────────────────
echo  🔍 Verifying PyTorch and GPU access...
python -c "import torch; print(f'  [INFO] PyTorch version: {torch.__version__}'); print(f'  [INFO] CUDA Available: {torch.cuda.is_available()}')"

REM ── Download models ───────────────────────────────────────
echo  🤖 Downloading AI model weights...
python scripts\download_models.py

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║  ✅ Setup Complete!                       ║
echo  ║                                           ║
echo  ║  Start the app:  python run.py            ║
echo  ║  Or directly:    streamlit run app.py     ║
echo  ╚═══════════════════════════════════════════╝
echo.
pause
