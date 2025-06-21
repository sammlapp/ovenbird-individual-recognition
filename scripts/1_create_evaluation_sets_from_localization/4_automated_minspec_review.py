import json
import shutil
from glob import glob
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from opensoundscape import Audio, Spectrogram
from opensoundscape.localization import PositionEstimate
from opensoundscape.localization.position_estimate import positions_to_df
from opensoundscape.ml import bioacoustics_model_zoo as bmz

# Paths
out_dir = f"REDACTED"
localized_events_path = "REDACTED"
temp_audio_dir = "REDACTED"
array_event_jsons = glob(f"{localized_events_path}/SBT*/*Ovenbird*.json")


def spec_to_audio(spec, sr):
    y_inv = librosa.griffinlim(spec.spectrogram, hop_length=256, win_length=512)
    return Audio(y_inv, sr)


def distances_to_receivers(p, dims=2):
    return [
        np.linalg.norm(p.location_estimate[:dims] - r[:dims])
        for r in p.receiver_locations
    ]


def min_spec_to_audio(position, discard_over_distance=50, plot=False):

    clips = position.load_aligned_audio_segments()
    distances = distances_to_receivers(position)
    # filter out far clips
    clips = [c for i, c in enumerate(clips) if distances[i] < discard_over_distance]
    specs = [Spectrogram.from_audio(c, dB_scale=False) for c in clips]
    minspec = specs[0]._spawn(
        spectrogram=np.min(np.array([s.spectrogram for s in specs]), axis=0)
    )

    # normalize to same level as loudest clip
    max = np.max([c.samples.max() for c in clips])
    return (
        spec_to_audio(minspec, clips[0].sample_rate)
        .normalize(max)
        .extend_to(clips[0].duration)
    )


# beyond this, receivers are not included when generating minspec;
# avoids losing the signal because of minimum on distant receiver
minspec_discard_distance = 35  # meters

print(f"found {len(array_event_jsons)} arrays with events .json files")

for path in array_event_jsons:
    array_name = Path(path).parent.stem
    print(f"generating minspec audio for {array_name}")

    with open(path, "r") as f:
        reloaded_events = json.load(f)
    positions = [
        PositionEstimate.from_dict(d) for d in reloaded_events["localized_events"]
    ]

    # filter by residual rms
    print(f"n positions: {len(positions)}")
    positions = [p for p in positions if p.residual_rms < 5]
    print(f"n positions with rms<5: {len(positions)}")
    if len(positions) < 1:
        continue

    clip_dir = f"{temp_audio_dir}/{array_name}"

    Path(clip_dir).mkdir(exist_ok=True)

    # create minspec audio clips
    # if none of the clips are within minspec_discard_distance, we create an empty audio clip instead
    def process(p, i):
        try:
            min_spec_to_audio(p, discard_over_distance=minspec_discard_distance).save(
                f"{clip_dir}/{i}.wav"
            )
            return 0
        except:
            Audio.silence(sample_rate=48000, duration=3).save(f"{clip_dir}/{i}.wav")
            return 1

    results = Parallel(n_jobs=12)(
        delayed(process)(p, i) for i, p in enumerate(positions)
    )
    print(f"n failures: {sum(results)} of {len(results)}")

    hawkears = bmz.load("HawkEars")
    audio_clips = [
        f"{clip_dir}/{i}.wav" for i in range(len(positions))
    ]  # some may not exist, will get nan scores
    # get both species predicitons and embeddings - species predictions will allow us to filter out clips where wrong species is aligned
    # can test this with a random sample of 500 events to manually review
    preds = hawkears.predict(audio_clips, batch_size=512, num_workers=8)

    # bug: hawkears outputs include scores for non-existant audio
    nan_mask = preds.index.get_level_values("start_time").isna()
    preds[nan_mask] = np.nan
    positions_df = positions_to_df(positions)
    positions_df["hawkears_minspec_score_OVEN"] = preds["Ovenbird"].values
    positions_df.to_csv(f"{out_dir}/{array_name}_positions.csv")

    # remove temporary minspec clips
    shutil.rmtree(clip_dir)
