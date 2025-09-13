#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=/home/huong.nguyenthi2/SLR_PROJECT/src/script/logs/avi2npy_%j.log
#SBATCH --partition=dgx-small
module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kv_env

echo "Running in conda env: $CONDA_DEFAULT_ENV"
which python

cd /home/huong.nguyenthi2/SLR_PROJECT/src/
# python -m preprocessing.keypoint_continue
python -m preprocessing.optical_flow