# create version of Hawkears where off-center Ovenbirds are ignored:
# preprocessing mutes the first and third seconds of the 3 second clip
# detections with score > 0 for Ovenbird are retained for AIID recapture analysis
# Note: the original dataset on which this script was run is not included in this repo
# This script is provided for reference only.
# Please contact the authors if you wish to obtain the original dataset.

import sys
from glob import glob
from pathlib import Path
from time import time as timer

import bioacoustics_model_zoo as bmz
import numpy as np
import opensoundscape as opso
import pandas as pd
from opensoundscape import Audio
from tqdm.autonotebook import tqdm

sys.path.append("../../src/")
from preprocessor import mute_and_normalize

np.random.seed(2024)
t0 = timer()

# parameters
device = "cuda:0"
batch_size = 512
num_workers = 12
threshold = 0  # raising this often doesn't help - HawkEars gives distant, blurred ovenbird songs high scores

# find clean predictions with no other species detected
preds_dir = "REDACTED"
project_dir = "REDACTED"
det_paths = [
    f"{preds_dir}/hawkears_dets_v2/oven_dets_appl2021e.csv",
    f"{preds_dir}/hawkears_dets_v2/oven_dets_earl2022b.csv",
    f"{preds_dir}/hawkears_dets_v2/oven_dets_earl2023b.csv",
    f"{preds_dir}/hawkears_dets_v2/oven_dets_earl2024a.csv",
]


for det_path in det_paths:
    deployment = Path(det_path).stem.split("_")[-1]
    print(f"deployment: {deployment}")
    save_path = f"{project_dir}/centered_dets/oven_dets_centered_{deployment}.csv"
    if Path(save_path).exists():
        print(f"{deployment} already completed")
        continue  # don't repeat if completed

    dets = pd.read_csv(det_path)
    dets["card"] = dets["file"].apply(lambda x: x.split("/")[-2])

    dets["end_time"] = dets["start_time"] + 3
    dets = dets.set_index(["file", "start_time", "end_time"])[[]]

    # this preprocessing mutes all but the central 1 second of the 3 second clip
    # we don't add noise, as this didn't have the desired effect of filtering to foreground sounds
    hawkears_strict = bmz.HawkEars()
    hawkears_strict.preprocessor.insert_action(
        "mute_and_normalize",
        opso.Action(mute_and_normalize, is_augmentation=False),
        after_key="trim_audio",
    )
    hawkears_strict.device = device
    strict_scores = hawkears_strict.predict(
        dets, batch_size=batch_size, num_workers=num_workers
    )[["Ovenbird"]]

    # discard clips below threshold
    strict_scores = strict_scores[strict_scores["Ovenbird"] > threshold]
    strict_scores.to_csv(save_path)

print(
    f"finished center-only prediction in {timer()-t0/60:0.0f} minutes. Detections are saved to {save_path}"
)
