#!/usr/bin/env bash

# make logs dir if doesnt exist yet
mkdir -p logs

# Launch Slurm jobs for training feature extractors
sbatch slurm_scripts/job_000.sh
sbatch slurm_scripts/job_001.sh
sbatch slurm_scripts/job_002.sh
sbatch slurm_scripts/job_003.sh
sbatch slurm_scripts/job_004.sh
sbatch slurm_scripts/job_005.sh
sbatch slurm_scripts/job_006.sh
sbatch slurm_scripts/job_008_hawkears.sh
sbatch slurm_scripts/job_009_arcface_8.sh
sbatch slurm_scripts/job_010_full.sh
