"""run HawkEars inference on 14 SBT localization arrays, ~50 recorders each

1.7Tb of audio across all the arrays

requirements: bioacoustics-model-zoo (github)

Note: the full audio data used by this script is not provided in this repository, the script is provided for reference only
"""

import bioacoustics_model_zoo as bmz
from glob import glob
from pathlib import Path
from tqdm.autonotebook import tqdm

data_root = "REDACTED"  # path to folder containing audio from all localization arrays (not included in GitHub repo)
out_dir = "REDACTED"  # matches score_files_dir of 2_select_detections_to_localize.ipynb

arrays = glob(f"{data_root}/SBT-*")
m = bmz.HawkEars()
m.device = "cuda:0"

print(f"Found {len(arrays)} arrays from SBT dataset")
for audio_dir in tqdm(arrays):
    folders = glob(f"{audio_dir}/SBT*")
    if len(folders) < 1:
        continue
    array_name = Path(audio_dir).name
    preds_save_dir = f"{out_dir}/{array_name}/"
    Path(preds_save_dir).mkdir(exist_ok=True)

    folders = glob(f"{audio_dir}/SBT*")
    print(f"Found {len(folders)} audio folders for array {Path(audio_dir).name}")

    for f in tqdm(folders):
        files = glob(f"{f}/*.wav")
        if len(files) < 1:
            continue
        audio_folder_name = Path(f).name
        # print(f"Running hawkears on f{audio_folder_name}")

        save_dir = f"{preds_save_dir}/{audio_folder_name}"
        Path(save_dir).mkdir(exist_ok=True)

        preds_save_path = f"{save_dir}/{audio_folder_name}_hawkears_preds.csv"
        if Path(preds_save_path).exists():
            # already done
            continue
        print(f"running HawkEars prediction on {len(files)} files from {f}")

        preds = m.predict(files, num_workers=8, batch_size=256)

        # change from full path to just file name (much smaller file results)
        preds = preds.reset_index()
        preds["file"] = preds["file"].apply(lambda x: Path(x).name)
        # save smaller float format, 3 decimals
        preds.to_csv(
            f"{save_dir}/{Path(f).name}_hawkears_preds.csv",
            float_format="%.3f",
            index=False,
        )
