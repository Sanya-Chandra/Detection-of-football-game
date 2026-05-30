@echo off
REM Silent launcher for the Streamlit app using the virtual environment
IF EXIST "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" run.py
) ELSE (
    echo [ERROR] Virtual environment not found! Please run setup.bat first.
    pause
)
