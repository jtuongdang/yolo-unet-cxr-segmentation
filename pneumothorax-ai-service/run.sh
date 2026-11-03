#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  PneumoAI CADe/CADx Diagnostic PACS Service"
echo "  Decoupled U-Net++ & YOLOv11 Multi-Lesion Workstation"
echo "=========================================================="

if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment 'venv' not found in $DIR!"
    exit 1
fi

if [ ! -f "weights/best_unet_plus_plus.pth" ]; then
    echo "[WARNING] U-Net++ weights 'weights/best_unet_plus_plus.pth' not found!"
fi

if [ ! -f "weights/best_B_m1280.pt" ]; then
    echo "[WARNING] YOLO weights 'weights/best_B_m1280.pt' not found!"
fi

echo "[INFO] Starting Web Server on http://0.0.0.0:8000 ..."
echo "[INFO] Open http://localhost:8000 in your browser."
echo "=========================================================="

./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
