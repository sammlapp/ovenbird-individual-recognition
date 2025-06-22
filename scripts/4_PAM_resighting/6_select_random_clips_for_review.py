# Select 1000 random 3s clips to annotate for presence of Ovenbird song
import pandas as pd
from opensoundscape import Audio
from aru_metadata_parser.parse import audiomoth_start_time
from glob import glob
import datetime
import numpy as np
from tqdm.autonotebook import tqdm
from pathlib import Path
import bioacoustics_model_zoo as bmz
from sklearn.metrics import roc_auc_score, precision_score, recall_score

np.random.seed(2025)

datasets = {
    # REDACTED
}


metadata = {
    # REDACTED
}

# export re-usable set of clips for analysis
dir = "REDACTED""


points = pd.read_csv(f"{dir}/point_list.csv")["point"].values

all_clips = []
for dataset in metadata.keys():
    dpl = pd.read_csv(metadata[dataset])
    dpl = dpl.set_index("card_code")
    if "dropoff:Point ID" in dpl.columns:
        dpl = dpl.rename(columns={"dropoff:Point ID": "point_code"})

    files = pd.DataFrame({"file": glob(f"{datasets[dataset]}/*/*.WAV")})

    files["datetime"] = files["file"].apply(audiomoth_start_time)

    files["date"] = files["datetime"].apply(lambda x: x.date())
    files["time"] = files["datetime"].apply(lambda x: x.time())
    files["date_no_year"] = files["date"].apply(lambda x: x.replace(year=1000))

    files = files[files.date_no_year.apply(lambda x: x >= datetime.date(1000, 5, 15))]
    files = files[files.date_no_year.apply(lambda x: x <= datetime.date(1000, 5, 24))]
    files = files[
        files.time.apply(
            lambda x: x
            in (datetime.time(hour=10, minute=0), datetime.time(hour=9, minute=30))
        )
    ]

    if files.time.values[0] == datetime.time(hour=9, minute=30):
        earliest = 30 * 60
    else:
        earliest = 0

    files["card"] = files["file"].apply(lambda x: Path(x).parent.stem)
    files["point_code"] = files["card"].map(dpl["point_code"])
    files = files[files["point_code"].isin(points)]

    print(dataset)
    print(f"{len(files)} total files in period")

    clips = files.sample(300)[["file", "datetime", "date", "time", "point_code"]]
    clips["start_time"] = np.random.uniform(
        earliest, earliest + 60 * 90, size=len(clips)
    )
    clips["dataset"] = dataset
    clips["year"] = clips["date"].apply(lambda x: x.year)
    all_clips.append(clips)
all_clips = pd.concat(all_clips)

save_dir = "REDACTED"
# modify file and start_time to refer to the clips (for annotation tool)
all_clips["full_file"] = all_clips["file"].copy()
all_clips["full_file_offset"] = all_clips["start_time"].copy()
all_clips["file"] = None
all_clips["start_time"] = 0
for i, row in tqdm(all_clips.iterrows()):
    clip_name = f"random_{i:04n}_{row['point_code']}_{row['year']}.mp3"
    Audio.from_file(row.full_file, offset=row.start_time, duration=3).save(
        f"{save_dir}/{clip_name}"
    )
    all_clips.at[i, "file"] = clip_name
all_clips.to_csv(f"{save_dir}/random_clips_to_review.csv", index=False)

for year, clips in all_clips.groupby("year"):
    clips.to_csv(f"{save_dir}/{year}_clips.csv")

reviewed = pd.read_csv("./random_reviewed.csv")
reviewed.groupby("year")["annotation"].value_counts()

h = bmz.HawkEars()
df = pd.DataFrame(
    {
        "file": all_clips["full_file"],
        "start_time": all_clips["full_file_offset"],
        "end_time": all_clips["full_file_offset"] + 3,
    }
).set_index(["file", "start_time", "end_time"])
p = h.predict(df, batch_size=64, num_workers=0)
reviewed["hawkears_OVEN_score"] = p["Ovenbird"].values

reviewed.to_csv("./random_reviewed_w_score.csv", index=False)

for year, subset in reviewed.groupby("year"):
    auroc = roc_auc_score(
        (subset.annotation == "yes").astype(int).values, subset["hawkears_OVEN_score"]
    )
    precision = precision_score(
        (subset.annotation == "yes").astype(int).values,
        subset["hawkears_OVEN_score"] > 1,
    )
    recall = recall_score(
        (subset.annotation == "yes").astype(int).values,
        subset["hawkears_OVEN_score"] > 1,
    )

    print(
        f"{year}: n={(subset.annotation=='yes').sum()}/{len(subset)} auroc={auroc:0.3f}, precision={precision:0.3f}, recall={recall:0.3f}"
    )
