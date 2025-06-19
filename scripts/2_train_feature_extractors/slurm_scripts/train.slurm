#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=%j_%t.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared,GPU-small
#SBATCH --gpus=1 #8 per core, max=4 on GPU-shared

source ~/.bashrc

module load anaconda3/2024.10-1
module load cuda
conda activate /jet/projects/bio200037p/sml161/conda/opso

cd /jet/home/sammlapp/song25_oven_aiid/oven_aiid/develop_and_evaluate_aiid/4_train_aiid
python train.py train_configs/arcface.yml

python ~/scripts/jobstats.py