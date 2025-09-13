#!/bin/bash
#SBATCH --job-name=rgb_multiclip
#SBATCH --output=/home/huong.nguyenthi2/SLR_PROJECT/src/script/logs/rgb_multiclip_%j.log
#SBATCH --error=/home/huong.nguyenthi2/SLR_PROJECT/src/script/logs/rgb_multiclip_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "Starting RGB Multi-clip Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Date: $(date)"
echo "=================================="

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kv_env

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd /home/huong.nguyenthi2/SLR_PROJECT/src/
export PYTHONPATH="/home/huong.nguyenthi2/SLR_PROJECT:$PYTHONPATH"
export WANDB_PROJECT="slr-rgb-training"
export WANDB_SILENT="true"

echo "Environment Variables:"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "=================================="

echo "Starting Training..."
python -m src.script.train_rgb
exit_code=$?

echo "=================================="
echo "Training Summary"
echo "Exit code: $exit_code | End time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully"
    echo "W&B dashboard: https://wandb.ai/projects/slr-rgb-training"
    echo "Checkpoints: /home/huong.nguyenthi2/SLR_PROJECT/checkpoints/rgb"
else
    echo "Training failed"
fi
echo "=================================="
