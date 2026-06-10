#!/bin/bash
#SBATCH --job-name=main_sebra_train
#SBATCH --output=logs/main_sebra_train_%j.out
#SBATCH --error=logs/main_sebra_train_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00


# ----------------------------
# Fail fast
# ----------------------------
set -e


# ----------------------------
# Setup logs early
# ----------------------------
mkdir -p logs


# ----------------------------
# Environment
# ----------------------------
source .venv/bin/activate


# ----------------------------
# CPU / CUDA stability
# ----------------------------
export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


# ----------------------------
# Debug info
# ----------------------------
echo "Running on: $(hostname)"
nvidia-smi || true


# ----------------------------
# Run training
# ----------------------------
python Zebra_Identifier/models/main_sebra_train.py