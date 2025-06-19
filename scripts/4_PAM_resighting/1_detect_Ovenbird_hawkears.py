"""
Sample HawkEars + Bioacoustics Model zoo perdiction script

this script runs HawkEars CNN species classifier on audio files

for the sake of example and reproducibility, we run the species classification
on all clips that were retained and included in the AIID analysis
the model was originally run on the full PAM dataset

This script was originally run on the full PAM dataset, which is
available from the authors upon reasonable request.
"""

import pandas as pd
from pathlib import Path
import bioacoustics_model_zoo as bmz

# download the provided dataset of audio clips, then specify the path to the dataset here
pam_dataset_path = "../../../pam_dataset_v4/"

# inference ML parameters
batch_size = 1024  # use arge values when running on GPU, set to 1 if using CPU
num_workers = 8  # cpu parallelization of pre-processing

# this dataset contains 3s audio clips ultimately used for the AIID analysis
ovenbird_clips = pd.read_csv(f"{pam_dataset_path}/pam_dataset_clips.csv")
ovenbird_clips["file"] = ovenbird_clips["clip_name"].apply(
    lambda clip: f"{pam_dataset_path}/audio/{clip}"
)
ovenbird_clips["start_time"] = 0
ovenbird_clips["end_time"] = 3
ovenbird_clips = ovenbird_clips.set_index(["file", "start_time", "end_time"])

# optionally, subset to a random selection of clips to run this script quickly
print("running HawkEars on 30 random clips for quick example")
ovenbird_clips = ovenbird_clips.sample(30)

# Set directory for predictions to be saved to
save_dir = "../resources/species_classifier_outputs"
Path(save_dir).mkdir(exist_ok=True)

# Load HawkEars from Bioacoustics Model Zoo
# if this is the first time running this script, the model will be downloaded
model = bmz.HawkEars()

# Use the model to predict the presence of species in 3s audio clips
print("Beginning prediction. \n")
scores = model.predict(
    ovenbird_clips,
    batch_size=batch_size,
    num_workers=num_workers,
)

# save predictions to csv:
# HawkEars predicts the presence of >300 species.
# Here we retain only the confidence scores for Ovenbird.
# scores[["Ovenbird"]].to_csv(f"{save_dir}/hawkears_preds.csv")
# print(f"Prediction complete. Outputs are saved to {save_dir}")
