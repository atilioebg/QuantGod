#!/bin/bash

# Cloud Setup Script for QuantGod ETL
echo "------------------------------------------------------------"
echo "🚀 Preparing Cloud Environment for QuantGod ETL"
echo "------------------------------------------------------------"

# 1. Update and install basic dependencies
sudo apt update && sudo apt install -y rclone python3-pip python3-venv

# 2. Create Virtual Environment
if [ ! -d "venv_cloud" ]; then
    echo "[INFO] Creating virtual environment 'venv_cloud'..."
    python3 -m venv venv_cloud
fi

# 3. Install Python requirements
echo "[INFO] Installing Python dependencies..."
source venv_cloud/bin/activate
pip install --upgrade pip
pip install -r cloud/requirements_cloud.txt

# 4. Create directory structure
echo "[INFO] Creating data directories..."
mkdir -p data/processed
mkdir -p data/models
mkdir -p logs

# 5. Check Rclone configuration
if [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
    echo "⚠️  WARNING: Rclone config not found at ~/.config/rclone/rclone.conf"
    echo "Please configure rclone using 'rclone config' before running the pipeline."
else
    echo "✅ Rclone config found."
fi

echo "------------------------------------------------------------"
echo "✅ Setup complete. To start the pipeline:"
echo "1. Activate venv: source venv_cloud/bin/activate"
echo "2. Run pipeline: python -m cloud.orchestration.run_pipeline"
echo "------------------------------------------------------------"
