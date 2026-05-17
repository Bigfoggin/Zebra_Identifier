#!/bin/bash
#SBATCH --job-name=zebra_validate
#SBATCH --output=logs/validate_weighted.out
#SBATCH --error=logs/validate_weighted.err
#SBATCH --ntasks=1

# --- Your Specific Settings ---
#SBATCH --partition=students          
#SBATCH --mem=32G                     
#SBATCH --gres=gpu:a100:1             
#SBATCH --cpus-per-task=32            
#SBATCH --time=1-00:00:00             

# Create logs directory if it doesn't exist
mkdir -p logs

# Load CUDA (optional for CPU mode, but good practice for consistency)
module load cuda/12.1

# Activate environment
source .venv/bin/activate

# Run code
# Pointing to the new weighted validation script
python Zebra_Identifier/models/validate_weighted.py