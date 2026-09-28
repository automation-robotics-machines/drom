#!/usr/bin/env bash

# -----------------------------
# Setup DROM workspace
# -----------------------------

echo "[1/4] Creating virtual environment..."
python -m venv .env
source .env/bin/activate

sleep 1

echo "[2/4] Installing package dependencies..."
pip install -e .

sleep 1

echo "[3/4] Installing PyTorch, TorchVision, and TorchAudio dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

sleep 1

echo "[4/4] Finished DROM setup. Enjoy!"