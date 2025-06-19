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
labels_dir = "../../../localization_dataset_labeled_clips/"
labels = pd.read_csv(f"{labels_dir}/labeled_clips.csv")

# prepare for prediction
labels["file"] = labels["rel_path"].apply(lambda x: Path(labels_dir) / x)
labels["song_center_time"] = 5  # we have 10s audio clip centered on the annotated song

val_labels = labels[labels["data_split"] == "val"]
test_labels = labels[labels["data_split"] == "test"]
# Generate test set embeddings with each model

## Supervised point=label
ckpt_path = "../../checkpoints/full_best.pth"
m = Resnet18_Classifier(num_classes=234)
m.load_state_dict(torch.load(ckpt_path))
m.device = torch.device("cuda:0")
m.to(m.device)

pre = OvenbirdPreprocessor()
pre.pipeline.load_audio.set(load_metadata=False)

# create embeddings once, this is a deterministic step
print("embedding samples")
test_embeddings = evaluation.embed(test_labels, m, pre, batch_size=128, num_workers=8)


all_scores = []
nreps = 30
for alg in ("umap", "tsne"):
    if alg == "umap":
        ndim_options = [30]
    elif alg == "tsne":
        ndim_options = [2, 3]
    for ndim in ndim_options:
        for _ in tqdm(range(nreps)):
            labels, features = evaluation.cluster(
                test_labels,
                test_embeddings,
                reduction_algorithm=alg,
                reduced_n_dimensions=ndim,
            )
            scores = evaluation.evaluate(test_labels["aiid_label"].values, labels)
            scores["ndim"] = ndim
            scores["reduction_algorithm"] = alg or "None"
            all_scores.append(scores)
dim_red_performance = pd.DataFrame(all_scores)
dim_red_performance.to_csv(
    "../../results/dimemsionality_reduction_test_performance.csv", index=False
)
summary = (
    dim_red_performance.groupby(["reduction_algorithm", "ndim"], dropna=False)
    .mean()
    .round(2)
)
summary.to_csv("../../results/dimemsionality_reduction_test_performance_means.csv")

import seaborn as sns
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from matplotlib import pyplot as plt


def figsize(w, h):
    plt.rcParams["figure.figsize"] = [w, h]


figsize(5, 3)
# %config InlineBackend.figure_format = 'retina'
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


sns.boxplot(
    dim_red_performance.reset_index(),
    x="ndims",
    y="accuracy",
    hue="reduction_algorithm",
)
plt.savefig("../../figures/dimensionality_reduction_val_accuracy.pdf")
