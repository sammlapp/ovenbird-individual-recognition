# evaluation utils and helpers
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score, homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score, accuracy_score
import numpy as np
from pathlib import Path

from dataset import AIIDLocalizedClipDataset
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch
import scipy


def identity(x):
    return x


import pandas as pd

from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE

from sklearn.cluster import HDBSCAN

from preprocessor import OvenbirdPreprocessor
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def embed(
    aiid_df, model, preprocessor, batch_size=64, num_workers=0,
):
    # embed the samples with the current network
    dataloader = DataLoader(
        dataset=AIIDLocalizedClipDataset(
            aiid_df, preprocessor=preprocessor, bypass_augmentations=True
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity,
        shuffle=False,
    )
    embeddings = []
    for batch in tqdm(dataloader):
        batch_data = torch.vstack([s.data[None, :, :] for s in batch]).to(model.device)
        model.eval()
        with torch.no_grad():
            outputs = model(batch_data)  # might return (embedding,logit)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # discard projections or class outputs
        embeddings.append(outputs)
    return torch.cat(embeddings).detach().cpu().numpy()


def cluster_points(
    clip_df,
    embeddings,
    use_spatial_data=False,
    cluster_grouping_col=None,
    reduced_n_dimensions=5,
    min_cluster_size=5,
    random_state=None,
    reduction_algorithm='umap',
):
    """dimensionally reduce everything, then cluster one group at a time"""
    # dimensionality reduction on embeddings
    all_features = reduce_dims(
        embeddings,
        reduction_algorithm=reduction_algorithm,
        reduced_n_dimensions=reduced_n_dimensions,
        random_state=random_state,
    )

    if use_spatial_data:
        # add x and y columns to the UMAP feature array
        x = clip_df["x"].values  # 1d array
        y = clip_df["y"].values  # 1d array
        all_features = np.concatenate([all_features, x[:, None], y[:, None]], axis=1)

    # rescale to mean=0, variance=1
    all_features = StandardScaler().fit_transform(all_features)

    all_features = pd.DataFrame(all_features, index=clip_df.index)

    if cluster_grouping_col is not None:

        # be careful to retain index (make list of df) so that we can order properly later
        all_labels = []

        label_offset = 0
        print(f"clustering each value of {cluster_grouping_col} separately")
        # treat each grid or point or other category separately for clustering
        for group_id, group_df in tqdm(clip_df.groupby(cluster_grouping_col)):

            # cluster with HDBSCAN
            if len(group_df) <= min_cluster_size:
                # cannot cluster, too few samples
                labels = np.array([-1] * len(group_df))
            else:
                grid_features = all_features[clip_df[cluster_grouping_col] == group_id]
                hdb = HDBSCAN(min_cluster_size=min_cluster_size)
                hdb.fit(grid_features)
                labels = hdb.labels_

            # use unique labels rather than duplicating from other grids
            # eg first grid uses [0,1,2], second [3,4] etc
            # but -1 retained across all grids as un-clustered noise class
            global_labels = [l if l == -1 else l + label_offset for l in labels]
            all_labels.append(
                pd.DataFrame(index=group_df.index, data={"label": global_labels})
            )

            label_offset += (
                max(labels) + 1
            )  # shift labels for next group to not re-use labels

        all_labels = pd.concat(all_labels).loc[clip_df.index].values[:, 0]

    else:
        # cluster all samples simultaneously with HDBSCAN
        print("clustering all samples")
        hdb = HDBSCAN(min_cluster_size=min_cluster_size)
        hdb.fit(all_features)
        all_labels = hdb.labels_

    return all_labels, all_features


def reduce_dims(
    features, reduction_algorithm, reduced_n_dimensions, random_state=None
):
    """apply dimensionality reduction on an array where each row represents a feature vector

    Args:
        features: floating point array where rows are feature vectors
        reduction_algorithm: 'umap', 'tsne', or None for no reduction
        reduced_n_dimensions: how many dimensions should be output
            - note that TSNE only supports up to 3 in the default algorithm

    Returns:
        np.array of dimensionality-reduced features
    """

    if reduction_algorithm is None or reduced_n_dimensions is None:
        return np.array(features)
    
    # standardize features on each dimension before reduction
    scaled_features = StandardScaler().fit_transform(features)

    # choose reduction algorithm
    if reduction_algorithm == "umap":
        reducer = umap.UMAP(
            n_components=reduced_n_dimensions, random_state=random_state
        )
    elif reduction_algorithm == "tsne":
        reducer = TSNE(n_components=reduced_n_dimensions)
    else:
        raise ValueError(
            f"unrecognized reduction_algorithm {reduction_algorithm}, can be 'umap' or 'tsne' or None"
        )
    # fit reduction algorithm and reduce dimensionality
    return reducer.fit_transform(scaled_features)


def cluster(
    aiid_df,
    embeddings,
    use_spatial_data=False,
    cluster_grouping_col=None,
    reduced_n_dimensions=5,
    min_cluster_size=5,
    reduction_algorithm="umap",
    random_state=None,
):
    """reduce dimensions and cluster both happen within group samples only"""
    if cluster_grouping_col is not None:

        # be careful to retain index (make list of df) so that we can order properly later
        all_labels = []
        all_features = []

        label_offset = 0
        print(f"clustering each value of {cluster_grouping_col} separately")
        # treat each grid or point or other category separately for clustering
        for group_id, group_df in tqdm(aiid_df.groupby(cluster_grouping_col)):
            grid_embeddings = embeddings[aiid_df[cluster_grouping_col] == group_id]

            # dimensionality reduction on embeddings
            features = reduce_dims(
                grid_embeddings,
                reduction_algorithm=reduction_algorithm,
                reduced_n_dimensions=reduced_n_dimensions,
                random_state=random_state,
            )

            if use_spatial_data:
                # add x and y columns to the dim-reduced feature array
                x = group_df["x"].values  # 1d array
                y = group_df["y"].values  # 1d array
                features = np.concatenate([features, x[:, None], y[:, None]], axis=1)

            # standard scaler
            features = StandardScaler().fit_transform(features)

            # cluster with HDBSCAN
            hdb = HDBSCAN(min_cluster_size=min_cluster_size)
            hdb.fit(features)

            # use unique labels rather than duplicating from other grids
            all_labels.append(
                pd.DataFrame(
                    index=group_df.index, data={"label": hdb.labels_ + label_offset}
                )
            )
            all_features.append(pd.DataFrame(index=group_df.index, data=features))
            label_offset += hdb.labels_.max() + 1

        all_labels = pd.concat(all_labels).loc[aiid_df.index].values[:, 0]
        all_features = pd.concat(all_features).loc[aiid_df.index].values

    else: 
        # print("reducing dimensions")
        all_features = reduce_dims(
                embeddings,
                reduction_algorithm=reduction_algorithm,
                reduced_n_dimensions=reduced_n_dimensions,
                random_state=random_state,
            )
        

        if use_spatial_data:
            # add x and y columns to the dim-reduced feature array
            x = aiid_df["x"].values  # 1d array
            y = aiid_df["y"].values  # 1d array
            all_features = np.concatenate(
                [all_features, x[:, None], y[:, None]], axis=1
            )

        # cluster with HDBSCAN
        # print("clustering")
        all_features = StandardScaler().fit_transform(all_features)
        hdb = HDBSCAN(min_cluster_size=min_cluster_size)
        hdb.fit(all_features)
        all_labels = hdb.labels_

    return all_labels, all_features


## Embed, reduce dims, and cluster in one function ##
def make_pseudolabels(
    aiid_df,
    model,
    preprocessor,
    use_spatial_data=False,
    use_grid_id=False,  # True sets cluster_grouping_col='grid_id'
    cluster_grouping_col=None,
    batch_size=64,
    num_workers=0,
    reduced_n_dimensions=5,
    min_cluster_size=5,
    random_state=None,
    global_dim_reduction=False,  # if True, does dimensionality reduction on all data before clustering on subsets
    reduction_algorithm='umap',
):
    # print("embedding samples")
    embeddings = embed(aiid_df, model, preprocessor, batch_size, num_workers)

    if use_grid_id:
        assert cluster_grouping_col is None, "double-specified cluster grouping column"
        cluster_grouping_col = "grid_id"

    clustering_fn = cluster_points if global_dim_reduction else cluster
    all_labels, all_features = clustering_fn(
        aiid_df,
        embeddings=embeddings,
        use_spatial_data=use_spatial_data,
        cluster_grouping_col=cluster_grouping_col,
        reduced_n_dimensions=reduced_n_dimensions,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        reduction_algorithm=reduction_algorithm,
    )

    return all_labels, all_features, embeddings


## EVAL utils ##
def bipartite_hungarian_matching_accuracy(true_labels, predicted_labels):
    """
    Compute the accuracy of predicted labels against true labels using
    bipartite matching via the Hungarian algorithm.
    """

    # Create a confusion matrix
    # automatically adds cols/rows if there are missing classes in either true or predicted labels
    # ie if there are more total classes in either true or predicted labels
    cm = confusion_matrix(true_labels, predicted_labels)

    # Apply the Hungarian algorithm to find the optimal assignment
    # the "cost" to minimize is the negative of the confusion matrix, since we want to maximize the matches
    # and the cm counts the number of matches between true and predicted labels
    # for a specific label correspondence
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Calculate the accuracy based on the optimal assignment
    total_correct = cm[row_ind, col_ind].sum()
    total_samples = len(true_labels)

    accuracy = total_correct / total_samples
    return accuracy


# Cluster Purity
def cluster_purity(true_labels, predicted_labels):
    cluster_labels = np.unique(predicted_labels)
    majority_labels = []
    true_labels_reordered = []
    for label in cluster_labels:
        indices = np.where(predicted_labels == label)
        # Select the most common true label in the predicted cluster
        majority_label = scipy.stats.mode(np.array(true_labels)[indices])[0]
        majority_labels.extend([majority_label] * len(indices[0]))
        true_labels_reordered.extend(true_labels[indices])
    return accuracy_score(true_labels_reordered, majority_labels)


def evaluate(true_labels, predicted_labels, verbose=False):
    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # Fowlkes-Mallows Index (FMI)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    # Homogeneity, Completeness, and V-Measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, predicted_labels
    )

    purity = cluster_purity(true_labels, predicted_labels)

    accuracy = bipartite_hungarian_matching_accuracy(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
    )

    if verbose:
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.3f}")
        print(f"Fowlkes-Mallows Index (FMI): {fmi:.3f}")
        print(f"Homogeneity: {homogeneity:.3f}")
        print(f"Completeness: {completeness:.3f}")
        print(f"V-Measure: {v_measure:.3f}")
        print(f"Cluster Purity: {purity:.3f}")
        print(f"Bipartite Hungarian Matching Accuracy: {accuracy:.3f}")

    return {
        "ari": ari,
        "nmi": nmi,
        "fmi": fmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "purity": purity,
        "accuracy": accuracy,
    }


def run_eval(
    model,
    labels_dir,
    split="val",
    pre=None,
    batch_size=256,
    num_workers=8,
    min_cluster_size=7,
    reduced_n_dimensions=5,
    reduction_algorithm='umap',
):

    labels = pd.read_csv(f"{labels_dir}/labeled_clips.csv")

    # prepare for prediction
    labels["file"] = labels["rel_path"].apply(lambda x: Path(labels_dir) / x)
    labels["song_center_time"] = (
        5  # we have 10s audio clip centered on the annotated song
    )

    # choose validation or test set
    labels = labels[labels["data_split"] == split]

    if pre is None:
        pre = OvenbirdPreprocessor()
        pre.pipeline.load_audio.set(load_metadata=False)
        # pre.pipeline.to_tensor.set(range=[-70, -20])

    predicted_labels, _, _ = make_pseudolabels(
        labels,
        model=model,
        preprocessor=pre,
        batch_size=batch_size,
        num_workers=num_workers,
        min_cluster_size=min_cluster_size,
        reduced_n_dimensions=reduced_n_dimensions,
        reduction_algorithm=reduction_algorithm,

    )
    return evaluate(labels["aiid_label"].values, predicted_labels)


# utility to create summary df with mean and range of performance metrics
def mean_range_table(df, grouping_col, value_cols):
    summary = []
    for name, group in df.groupby(grouping_col):
        row = {}
        for col in value_cols:
            values = group[col].dropna()
            if not values.empty:
                mean = values.mean()
                min_val = values.min()
                max_val = values.max()
                row[col] = f"{mean:.2f} ({min_val:.2f}-{max_val:.2f})"
            else:
                row[col] = "NaN"
        row[grouping_col] = name
        summary.append(row)

    # Create DataFrame from summary
    summary_df = pd.DataFrame(summary).set_index(grouping_col)
    return summary_df