#!/bin/bash
#SBATCH --job-name=zebra_error_analysis
#SBATCH --output=logs/error_analysis.out
#SBATCH --error=logs/error_analysis.err
#SBATCH --ntasks=1
#SBATCH --partition=students          
#SBATCH --mem=32G                     
#SBATCH --cpus-per-task=4            
#SBATCH --time=02:00:00             

# Activate environment
source .venv/bin/activate

# Execute
python Zebra_Identifier/models/wrong_images_weighted.py