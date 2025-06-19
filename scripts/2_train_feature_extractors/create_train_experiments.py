import itertools
import yaml
import os
import copy

# will use just 10_000 of ~94,000 training clips for experiments

experiments = []

loss_experiments = [
    {"name":"BCE Loss"},  # default config with BCE loss
    {
        "name":"CE loss",
        "training": {
            "loss_fn": "cross_entropy_loss",
        }
    },
    {
         "name":"Arcface loss 1",
        "training": {
            "loss_fn": "subcenter_arcface_loss",
            "arcface_loss_subcenters": 1,
        }
    },
    {
        "name":"Arcface loss 2",
        "training": {
            "loss_fn": "subcenter_arcface_loss",
            "arcface_loss_subcenters": 2,
        }
    },
    {
        "name":"Arcface loss 4",
        "training": {
            "loss_fn": "subcenter_arcface_loss",
            "arcface_loss_subcenters": 4,
        }
    },
        {
        "name":"Arcface loss 8",
        "training": {
            "loss_fn": "subcenter_arcface_loss",
            "arcface_loss_subcenters": 8,
        }
    },
    {
        "name":"Contrastive loss",
        "training": {
            "loss_fn": "contrastive_point_loss",
        }
    },
    {
        "name":"Contrastive loss 32ppb",
        "training": {
            "loss_fn": "contrastive_point_loss",
            "n_points_per_batch": 32,
        }
    },
    {
        "name":"SSL Contrastive",
        "training": {
            "loss_fn": "ssl_contrastive_point_loss",
        }
    },
    {
        "name":"SSL Contrastive 32ppb",
        "training": {
            "loss_fn": "ssl_contrastive_point_loss",
            "n_points_per_batch": 32,
        }
    },
]
experiments.extend(loss_experiments)

backbone_experiments = [{"name":"resnet50","training": {"backbone": "resnet50"}}]
experiments.extend(backbone_experiments)

preprocessing_experiments = [
    {"name":f"{n}s clips", "preprocessing": {"clip_duration": n}} for n in (1, 2, 4)
]
preprocessing_experiments.append({"name":"reduce noise", "preprocessing": {"reduce_noise": True}})
preprocessing_experiments.append({"name":"overlay", "preprocessing": {"use_overlay": True}})
experiments.extend(preprocessing_experiments)

n_points_experiments = [{"name":f"{n} points", "data": {"max_points": n}} for n in (8, 16, 32, 64, 128)]
experiments.extend(n_points_experiments)

training_set_size_experiments = [
    {"name": f"{n} samples", "data": {"n_train_samples": n}} for n in (100, 500, 1000, 5000, 10_000, 50_000, None)
]
experiments.extend(training_set_size_experiments)

# Load base config
with open("./train_configs/base.yml", "r") as f:
    base_config = yaml.safe_load(f)

os.makedirs("slurm_scripts", exist_ok=True)

def deep_update(orig, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(orig.get(k), dict):
            deep_update(orig[k], v)
        else:
            orig[k] = v

# Number of experiments per Slurm job
experiments_per_job = 5

from itertools import islice

experiment_id = 0
# Iterate over experiments in chunks of 'experiments_per_job'
for job_index, start in enumerate(range(0, len(experiments), experiments_per_job)):
    # Create a list to hold the configuration file paths for this job
    config_paths = []

    # Process a chunk of experiments
    for i, experiment_cfg in enumerate(islice(experiments, start, start + experiments_per_job)):
        config = copy.deepcopy(base_config)
        deep_update(config, experiment_cfg)

        # Save to config file
        config_filename = f"config_{experiment_id:03}.yml"
        config_path = os.path.join("train_configs", config_filename)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        config_paths.append(config_path)
        experiment_id+=1


    # Write SLURM script for this set of experiments
    slurm_script = f"""#!/usr/bin/env bash
#SBATCH --job-name=exp_set_{job_index:03}
#SBATCH --output=logs/exp_set_{job_index:03}.out
#SBATCH --error=logs/exp_set_{job_index:03}.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

source ~/.bashrc

module load anaconda3/2024.10-1
module load cuda
conda activate /jet/projects/bio200037p/sml161/conda/opso

cd /jet/home/sammlapp/song25_oven_aiid/oven_aiid/develop_and_evaluate_aiid/4_train_aiid

"""

    # Add commands to run each experiment sequentially
    for config_path in config_paths:
        slurm_script += f"python train.py {config_path}\n"

    slurm_script += """
python ~/scripts/jobstats.py

# List the Slurm script as completed
echo "$0" >> completed_slurm_jobs.txt
"""

    # Save the Slurm script
    slurm_script_path = os.path.join("slurm_scripts", f"job_{job_index:03}.sh")
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script)

# submit:
# for script in slurm_scripts/job_*.sh; do
#     sbatch "$script"
# done
