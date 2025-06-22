# extract ~1.1M 3s clips as MP3 into a publishable dataset
# this scrript was used to create the contents of the /audio directory in the
# publicly available 4-year passive acoustic monitoring dataset (Dryad: add link once public)
# Note: the inputs to this script are not included in this repository. The
# script is included for reference only. The dataset created by this script is a
# public dataset of Ovenbird detections in the selected date and time window.
# That dataset can be downloaded and used to reproduce downstream analyses and
# results.

import pandas as pd
from opensoundscape import Audio
from pathlib import Path

pam_dataset_dir = "../../../pam_dataset_v4/"

clip_df = pd.read_csv(f"{pam_dataset_dir}/pam_dataset_clips.csv")
print(f"extracting {len(clip_df)} clips to {pam_dataset_dir}")

Path(Path(pam_dataset_dir) / "audio").mkdir(exist_ok=True, parents=True)


def extract_clip(row):
    clip_path = f"{pam_dataset_dir}/audio/{row.clip_name}"
    if Path(clip_path).exists():
        return
    Audio.from_file(
        row.file, load_metadata=False, offset=row.start_time, duration=3
    ).save(clip_path, compression_level=0.1)


from joblib import Parallel, delayed

num_cores = 10
Parallel(n_jobs=num_cores)(delayed(extract_clip)(clip_df.loc[i]) for i in clip_df.index)
print(f"finished extraction to {pam_dataset_dir}")
