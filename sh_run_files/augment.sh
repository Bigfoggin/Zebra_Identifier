#!/bin/bash
#SBATCH --job-name=zebra_augment
#SBATCH --partition=students
#SBATCH --output=logs/augment_%j.out
#SBATCH --error=logs/augment_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Directory for logs
mkdir -p logs

# Activate Virtual Environment
source .venv/bin/activate

echo "Data Augmentation started at: $(date)"

# Execution
python Zebra_Identifier/data_augmentation/augment_data.py

echo "Data Augmentation finished at: $(date)"