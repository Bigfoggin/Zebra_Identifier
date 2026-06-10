#!/bin/bash
#SBATCH --job-name=sebra_analysis
#SBATCH --output=logs/sebra_analysis_%j.out
#SBATCH --error=logs/sebra_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00


set -e

mkdir -p logs

source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "Running analysis..."
python Zebra_Identifier/models/analyze_sebra_model.py