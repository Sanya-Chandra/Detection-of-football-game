@echo off
echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║     🗑️ Sports Analytics CV — Reset         ║
echo  ╚═══════════════════════════════════════════╝
echo.
echo  ⚠️  WARNING: This will delete the virtual environment (venv)
echo      and all downloaded AI models. You will need to run 
echo      setup.bat again before you can use the application.
echo.
set /p confirm="Are you sure you want to reset? (Y/N): "
if /i "%confirm%" NEQ "Y" (
    echo  ❌ Reset cancelled.
    pause
    exit /b
)

echo.
echo  [1/4] Force-closing any running Python processes to unlock files...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo  [2/4] Deleting virtual environment (venv)...
if exist "venv" (
    rmdir /s /q "venv"
    echo    ✅ venv deleted
) else (
    echo    ✅ venv already missing
)

echo  [3/4] Deleting downloaded AI models...
if exist "models\weights" (
    del /q "models\weights\*.pt" 2>nul
    echo    ✅ AI models deleted
) else (
    echo    ✅ No AI models found
)

echo  [4/4] Cleaning up Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1
echo    ✅ Cache cleared

echo.
echo  ✅ Reset Complete!
echo     Run setup.bat to reinstall everything.
echo.
pause
