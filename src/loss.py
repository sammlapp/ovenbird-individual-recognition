import torch
import numpy as np
import torch.nn.functional as F


def ssl_location_loss(
    features,
    labels,
    clip_ids,
    point_ids,
    temperature=0.1,
    same_clip_weight=1,
    different_location_weight=1,
    same_location_weight=1,
    pseudo_label_weight=1,
    device=torch.device("cpu"),
):
    """contrastive loss pushing different locations (points) apart and pulling augmented variations of same sample (clip_d) together

    computes feature similarity and evaluates 3 incentives:
        - same clip augmented in different ways should have similar feature vectors
        - clips from different locations (points) should have not have similar feature vectors
        - clips from same location should have similar feature vectors
        - (optional, only if labels are provided) clips from same pseudo-label should have similar feature vectors to eachother,
            but should not have similar feature vectors to clips from other pseudo-labels

    Args:
        features: projection head outputs, (batch_size,projection_size)
        labels: integer pseudo-labels for membership to clusters (batch_size,)
            - label -1 indicates no membership to a cluster
            - if labels=None, computes the loss without considering pseudo-label cluster similarity
        clip_ids: unique identifier for audio clips; same value when a single clip was augmented in different wayss
        point_ids: unique identifier for the location of the recording
        temperature: "sharpens" the similarity matrix
        same_clip_weight: contribution of loss term for "same clip, different augmentation, incentivise feature similarity" to total loss
        different_location_weight: contribution of loss term for "clips from different locations (points) should not have similar features" to total loss
        pseudo_label_weight: contribution of loss term for "clips from the same pseudo-label should have similar features" to total loss
        device: torch device on which to compute the loss
    """
    features = features.to(device)
    if labels is not None:
        labels = labels.to(device)

    # compute pair-wise feature vector similarity for each sample
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # 1. Invariance to Augmentation: Same clip with different augmentation should be close
    # define pairwise matrix for batch where 1 means same clip with different augmentation
    same_clip_different_aug = (
        torch.tensor(np.array(clip_ids)[:, None] == np.array(clip_ids)[None, :])
        .float()
        .to(device)
    )
    # don't need to reward similarity of the same augmentation to itself
    # so same clip, same aug gets a 0 in this matrix
    same_clip_different_aug.fill_diagonal_(0)

    # we might not have any clips of the same clip_id. In this case, drop this loss term
    if same_clip_different_aug.max() < 1:
        loss_same_clip = 0
        same_clip_weight = 0
    else:
        # Pairwise feature similarity of augmented variations of the same original clip:
        # negative loss term to incentivise similarity
        exp_logits_same_clip = torch.exp(similarity_matrix) * same_clip_different_aug
        loss_same_clip = -torch.log(exp_logits_same_clip.sum(1) + 1e-8).mean()
        # normalize by rate of pairwise instances of "same clip" in this batch
        #loss_same_clip /= same_clip_different_aug.mean()

    # 2. Inter-Point Discrepancy: Songs from different ARUs should be far apart
    # make 2d array of pairwise clips where 1 indicates clips are from different points
    # 0 indicates clips are from the same point
    different_location = (
        torch.tensor(np.array(point_ids)[:, None] != np.array(point_ids)[None, :])
        .float()
        .to(device)
    )
    # Pairwise similarity of features from clips originating from different locations:
    exp_logits_diff_location = torch.exp(similarity_matrix) * different_location
    # positive loss term to penalize similarity
    loss_different_location = torch.log(exp_logits_diff_location.sum(1) + 1e-8).mean()
    # normalize by rate of pairwise instances of "different grid" in this batch
    loss_different_location /= different_location.sum()

    # 3. Intra-Point Similarity: Songs from same ARUs should have similar features
    # Pairwise similarity of features from clips originating from same locations:
    same_location = 1 - different_location
    exp_logits_same_location = torch.exp(similarity_matrix) * same_location
    # negative loss term to incentivise similarity
    loss_same_location = -torch.log(exp_logits_same_location.sum(1) + 1e-8).mean()
    # normalize by rate of pairwise instances of "different grid" in this batch
    loss_same_location /= same_location.sum()

    # 4. Pseudo-Label Consistency: clips from the same point with the same pseudo-label
    if labels is None:  # no labels provided, don't compute the pseudo-label loss term
        loss_same_pseudolabel = 0
        pseudo_label_weight = 0
    else:

        # should have similar feature vectors
        same_pseudo_label = (
            torch.eq(labels[:, None], labels[None, :]).float().to(device)
        )
        # don't consider samples with cluster -1 to have the same pseudo-label
        # _except_ for the diagonal (i.e. the same sample)
        # -1 label is given by HDBSCAN for samples that are not assigned to any cluster
        # give -1 clips a "0"
        same_pseudo_label = same_pseudo_label * (labels[:, None] != -1).float()

        # Term for same-pseudolabel pairs:
        exp_logits_same_pseudolabel = torch.exp(similarity_matrix) * same_pseudo_label
        # negative term to incentivise similarity
        loss_same_pseudolabel = -torch.log(
            exp_logits_same_pseudolabel.sum(1) + 1e-8
        ).mean()
        # normalize by rate of same psuedo-label
        loss_same_pseudolabel /= same_pseudo_label.sum()

    combined_loss = (
        loss_same_clip * same_clip_weight
        + loss_different_location * different_location_weight
        + loss_same_location * same_location_weight
        + loss_same_pseudolabel * pseudo_label_weight
    ) / (
        same_clip_weight
        + different_location_weight
        + same_location_weight
        + pseudo_label_weight
    )

    return combined_loss
