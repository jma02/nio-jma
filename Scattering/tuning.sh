#!/bin/bash
#SBATCH --job-name=bcnn_tune
#SBATCH --output=logs/bcnn_tune_%A_%a.out
#SBATCH --error=logs/bcnn_tune_%A_%a.err
#SBATCH --array=0-10
#SBATCH --gpus=2
#SBATCH --partition=gpu-t4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G

vpkg_require anaconda/2024.02:python3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fnoenv

echo "HOST: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
which python
python --version
nvidia-smi

python bcnn_tune.py --trial_id ${SLURM_ARRAY_TASK_ID}
