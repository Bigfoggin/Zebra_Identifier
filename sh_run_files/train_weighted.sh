#!/bin/bash
#SBATCH --job-name=zebra_weighted
#SBATCH --output=logs/weighted_train_%j.out
#SBATCH --error=logs/weighted_train_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=students          
#SBATCH --mem=200G                    
#SBATCH --gres=gpu:a100:1             
#SBATCH --cpus-per-task=32            
#SBATCH --time=1-00:00:00             

# Load CUDA to resolve the driver warning
module load cuda/12.1

# Create log directory
mkdir -p logs

# Activate environment
source .venv/bin/activate

# Execute
python Zebra_Identifier/models/sebra_weighted.py