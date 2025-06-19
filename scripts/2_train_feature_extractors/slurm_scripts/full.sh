#!/usr/bin/env bash
#SBATCH --job-name=full
#SBATCH --output=logs/full.out
#SBATCH --error=logs/full.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

source ~/.bashrc

module load anaconda3/2024.10-1
module load cuda
conda activate /jet/projects/bio200037p/sml161/conda/opso

cd /jet/home/sammlapp/song25_oven_aiid/oven_aiid/develop_and_evaluate_aiid/4_train_aiid

python train.py train_configs/full.yml

python ~/scripts/jobstats.py

# List the Slurm script as completed
echo "$0" >> completed_slurm_jobs.txt
