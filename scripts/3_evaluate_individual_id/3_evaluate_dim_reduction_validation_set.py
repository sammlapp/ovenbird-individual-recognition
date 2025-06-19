"""Evaluate validation set performance of full model across dimensionality reduction hyperparameters (algorithm, n dimensions)

we evluate UMAP with 2,3,5,10,20,30 dimensions
and TSNE with 2,3 dimensions

because they are stochastic, we repeat 30 times

each time, we use HDBSCAN to cluster, then evaluate clustering performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
import torch
import random

# local tool imports
import sys

sys.path.append(f"../../src/")
from preprocessor import OvenbirdPreprocessor
from model import Resnet18_Classifier
import evaluation

random.seed(2024)
np.random.seed(2024)

# load annotations
# path to annotated individual Ovenbird dataset (download from EDI))
labels_dir = "../../../localization_dataset_labeled_clips"
labels = pd.read_csv(f"{labels_dir}/labeled_clips.csv")

# prepare for prediction
labels["file"] = labels["rel_path"].apply(lambda x: Path(labels_dir) / x)
labels["song_center_time"] = 5  # we have 10s audio clip centered on the annotated song

val_labels = labels[labels["data_split"] == "val"]
test_labels = labels[labels["data_split"] == "test"]
# Generate test set embeddings with each model

## Supervised point=label
ckpt_path = "../../checkpoints/full_2025-04-10T11:02:36.028451_best.pth"
m = Resnet18_Classifier(num_classes=234)
m.load_state_dict(torch.load(ckpt_path))
m.device = torch.device("cuda:0")
m.to(m.device)

pre = OvenbirdPreprocessor()
pre.pipeline.load_audio.set(load_metadata=False)

# create embeddings once, this is a deterministic step
print("embedding samples")
val_embeddings = evaluation.embed(val_labels, m, pre, batch_size=128, num_workers=8)

# reduce dimensionality of the features and perform clustering
# 30 repeats for each number of reduced dimensions, with UMAP and TSNE algorithms
# also try
all_scores = []
nreps = 30
for alg in (None, "umap", "tsne"):
    if alg == "umap":
        ndim_options = [2, 3, 5, 10, 20, 30]
    elif alg == "tsne":
        ndim_options = [2, 3]
    else:
        ndim_options = [None]
    for ndim in ndim_options:
        for _ in tqdm(range(nreps)):
            labels, features = evaluation.cluster(
                val_labels,
                val_embeddings,
                reduction_algorithm=alg,
                reduced_n_dimensions=ndim,
            )
            scores = evaluation.evaluate(val_labels["aiid_label"].values, labels)
            scores["ndim"] = ndim
            scores["reduction_algorithm"] = alg or "None"
            all_scores.append(scores)
dim_red_performance = pd.DataFrame(all_scores)
dim_red_performance.to_csv(
    "../../results/dimemsionality_reduction_val_performance.csv", index=False
)
summary = (
    dim_red_performance.groupby(["reduction_algorithm", "ndim"], dropna=False)
    .mean()
    .round(2)
)
summary.to_csv("../../results/dimemsionality_reduction_val_performance_means.csv")
