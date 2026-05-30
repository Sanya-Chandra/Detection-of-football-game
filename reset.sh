#!/bin/bash
set -e

echo ""
echo " ╔═══════════════════════════════════════════╗"
echo " ║     🗑️ Sports Analytics CV — Reset         ║"
echo " ╚═══════════════════════════════════════════╝"
echo ""
echo " ⚠️  WARNING: This will delete the virtual environment (venv)"
echo "     and all downloaded AI models. You will need to run "
echo "     setup.sh again before you can use the application."
echo ""
read -p "Are you sure you want to reset? (Y/N): " confirm
if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
    echo " ❌ Reset cancelled."
    exit 0
fi

echo ""
echo " [1/4] Force-closing any running Python processes..."
pkill -f python3 || true
sleep 1

echo " [2/4] Deleting virtual environment (venv)..."
if [ -d "venv" ]; then
    rm -rf venv
    echo "   ✅ venv deleted"
else
    echo "   ✅ venv already missing"
fi

echo " [3/4] Deleting downloaded AI models..."
if [ -d "models/weights" ]; then
    rm -f models/weights/*.pt
    echo "   ✅ AI models deleted"
else
    echo "   ✅ No AI models found"
fi

echo " [4/4] Cleaning up Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✅ Cache cleared"

echo ""
echo " ✅ Reset Complete!"
echo "    Run ./setup.sh to reinstall everything."
echo ""
