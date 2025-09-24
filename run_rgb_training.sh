#!/bin/bash
# Script to run RGB training with proper logging

echo "Starting RGB Training with Fixed Logging"
echo "========================================"

# Set environment variables
export PYTHONPATH="/home/huong.nguyenthi2/SLR_PROJECT:$PYTHONPATH"
export WANDB_PROJECT="slr-rgb-training"
export WANDB_SILENT="false"  # Enable wandb output for debugging

# Create necessary directories
mkdir -p /home/huong.nguyenthi2/SLR_PROJECT/checkpoints/rgb
mkdir -p /home/huong.nguyenthi2/SLR_PROJECT/src/lightning_logs

# Change to source directory
cd /home/huong.nguyenthi2/SLR_PROJECT/src/

echo "Environment setup:"
echo "PYTHONPATH: $PYTHONPATH"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "Current directory: $(pwd)"
echo "========================================"

# Run training
echo "Starting training..."
python -m src.script.train_rgb

echo "Training completed!"
echo "Check logs in:"
echo "  - Lightning logs: /home/huong.nguyenthi2/SLR_PROJECT/src/lightning_logs/"
echo "  - Checkpoints: /home/huong.nguyenthi2/SLR_PROJECT/checkpoints/rgb/"
echo "  - Wandb: https://wandb.ai/projects/slr-rgb-training"
