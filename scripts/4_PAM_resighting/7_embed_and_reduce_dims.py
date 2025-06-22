# embed sgl34 and Ohiopyple strict Ovenbird detections from 2021-2024 with AIID model
# then reduce dimensions to 3d with TSNE
# save reduced dim embeddings to file in numpy format

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
from sklearn.manifold import TSNE

# import classes from local modules
sys.path.append("../../src/")
from preprocessor import OvenbirdPreprocessor
from model import Resnet18_Classifier
from dataset import PointCodeDataset

# specify local paths
weights_path = "../../checkpoints/full_2025-04-10T11:02:36.028451_best.pth"  # feature extractor checkpoint
pam_dataset_path = "../../../pam_dataset_v4/"  # path to publicly abailable PAM dataset
save_path = f"{pam_dataset_path}/oven_clips_aiid_embeddings_tsne-3d.npy"

# embedding parameters
batch_size = 256
num_workers = 12

# load trained contrastive OVEN AIID model

model = Resnet18_Classifier(num_classes=234)
model.load_state_dict(torch.load(weights_path))
model.device = "cuda:0"
model.to(model.device)

preprocessor = OvenbirdPreprocessor()

# dataset contains 3s audio clips in which a species classification method detected Ovenbird
ovenbird_clips = pd.read_csv(f"{pam_dataset_path}/pam_dataset_clips.csv")
ovenbird_clips["file"] = ovenbird_clips["clip_name"].apply(
    lambda clip: f"{pam_dataset_path}/audio/{clip}"
)
ovenbird_clips["start_time"] = 0
ovenbird_clips["end_time"] = ovenbird_clips["start_time"] + 3
ovenbird_clips = ovenbird_clips.set_index(["file", "start_time", "end_time"])


def identity(x):
    return x


dataloader = DataLoader(
    dataset=PointCodeDataset(
        ovenbird_clips, preprocessor=preprocessor, bypass_augmentations=True
    ),
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=identity,
    shuffle=False,
)

# embed all clips by iterating the dataloader and running the model
print("embedding")
embeddings = []
for batch in tqdm(dataloader):
    batch_data = torch.vstack([s.data[None, :, :] for s in batch]).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch_data)  # might return (embedding,logit)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # discard projections or class outputs
    embeddings.append(outputs.detach().cpu())
embeddings = torch.cat(embeddings).numpy()


# rescale features before dimensionality reduction
print("rescaling")
scaled_emb = StandardScaler().fit_transform(embeddings)

# reduce dimensionality of features from 512 to 3 using TSNE
print("reducing with tsne to 3D")
reducer = TSNE(n_components=3, random_state=20250410)
all_features = reducer.fit_transform(scaled_emb)

# rescale features again afater dimensionality reduction
print("rescaling after tsne")
all_features = StandardScaler().fit_transform(all_features)

# save 3-dimensional features corresponding to rows of pam_dataset_clips.csv
np.save(save_path, all_features)

print(f"Finished. Saved features to {save_path}")
