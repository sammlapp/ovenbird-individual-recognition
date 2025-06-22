# cluster detected songs on per-point basis:
# load t-SNE features from AIID feature extractor, reduced to 3D
# cluster with HDBSCAN using various min_samples values
# clustering is performed by considering songs from one point at a time
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
import random
from tqdm.autonotebook import tqdm
from sklearn.cluster import AgglomerativeClustering

# specify local paths
weights_path = "../../checkpoints/full_best.pth"
pam_dataset_path = "../../../pam_dataset_v4/"
features_path = f"{pam_dataset_path}/oven_clips_aiid_embeddings_tsne-3d.npy"


# Random Seed
random_seed = 20250411
np.random.seed(random_seed)
random.seed(random_seed)

# load embeddings of strict detections from AIID model
ovenbird_clips = pd.read_csv(f"{pam_dataset_path}/pam_dataset_clips.csv")
features = np.load(features_path)
ovenbird_clips["features3d"] = features.tolist()

# chose distance threshold based on evaluation set
agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5)

# reduce dimensions and cluster one point at a time

for point_code, point_dets in tqdm(ovenbird_clips.groupby("point_code")):

    feat = np.vstack(point_dets["features3d"].values)
    for min_samples in (10, 30, 50, 100, 300):
        if len(point_dets) < min_samples:
            # use -1 noise clss as the label since there aren't enough points to form any clusters
            ovenbird_clips.loc[point_dets.index, f"cluster_{min_samples}"] = -1
            ovenbird_clips.loc[point_dets.index, f"clustered_{min_samples}"] = False
            continue
        # print(f"clustering with min_samples={min_samples}")
        hdb = HDBSCAN(
            min_samples=min_samples, allow_single_cluster=True, store_centers="medoid"
        )
        hdb.fit(feat)
        labels = np.array([f"{point_code}_{l}" for l in hdb.labels_])
        ovenbird_clips.loc[point_dets.index, f"clustered_{min_samples}"] = (
            hdb.labels_ >= 0
        )
        ovenbird_clips.loc[point_dets.index, f"cluster_{min_samples}"] = labels
        ovenbird_clips.loc[point_dets.index, f"cluster_{min_samples}_prob"] = (
            hdb.probabilities_
        )

        # store which were mediods
        for medoid in hdb.medoids_:
            m_idx = point_dets.iloc[
                np.linalg.norm((feat - medoid), axis=1).argmin()
            ].name
            ovenbird_clips.at[m_idx, f"cluster_{min_samples}_medoid"] = True

        ovenbird_clips[f"cluster_{min_samples}_medoid"] = ovenbird_clips[
            f"cluster_{min_samples}_medoid"
        ].fillna(False)

    # agglomerative clustering
    # aggcl_labels = agg_clustering.fit_predict(feat)
    # ovenbird_clips.loc[point_dets.index, f"aggcl_label"] = np.array(
    #     [f"{point_code}_{l}" for l in aggcl_labels]
    # )


ovenbird_clips.to_csv(f"{pam_dataset_path}/per_point_clusters.csv", index=False)

# select random 10 samples per cluster
dets_10_per_cluster = ovenbird_clips.groupby("cluster_30", group_keys=False).apply(
    lambda x: x.sample(10) if len(x) > 10 else x
)
dets_10_per_cluster.to_csv(
    f"{pam_dataset_path}/per_point_clusters_sample10.csv", index=False
)

# select medoid of each cluster (not used in the paper)
# ovenbird_clips[ovenbird_clips["cluster_30_medoid"] == True].to_csv(
#     f"{pam_dataset_path}/cluster_medoids.csv", index=False
# )

cluster_summary = ovenbird_clips.groupby("cluster_30")["point_code"].min()
cluster_summary.to_csv(f"{pam_dataset_path}/cluster_summary.csv")
