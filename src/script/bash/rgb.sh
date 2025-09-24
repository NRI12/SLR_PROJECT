#!/bin/bash
#SBATCH --job-name=rgb_multiclip
#SBATCH --output=/home/huong.nguyenthi2/SLR_PROJECT/src/script/logs/rgb_multiclip_%j.log
#SBATCH --error=/home/huong.nguyenthi2/SLR_PROJECT/src/script/logs/rgb_multiclip_%j.err
#SBATCH --partition=dgx-small
#SBATCH --time=24:00:00
#SBATCH --nodelist=hpc-dgx01

echo "Starting RGB Multi-clip Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Date: $(date)"
echo "=================================="

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kv_env


echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd /home/huong.nguyenthi2/SLR_PROJECT/
export PYTHONPATH="/home/huong.nguyenthi2/SLR_PROJECT:$PYTHONPATH"


echo "Starting Training..."
python -m src.script.train_rgb
exit_code=$?

echo "=================================="
echo "Training Summary"
echo "Exit code: $exit_code | End time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully"
    echo "Checkpoints: /home/huong.nguyenthi2/SLR_PROJECT/checkpoints/rgb"
else
    echo "Training failed"
fi
echo "=================================="
