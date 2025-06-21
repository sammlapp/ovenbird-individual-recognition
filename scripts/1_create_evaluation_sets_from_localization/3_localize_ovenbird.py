# localize loca2024grid1 detections with stricter filters
# Note: the full audio data used by this script is not provided in this repository, the script is provided for reference only

from opensoundscape.localization import SynchronizedRecorderArray
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from time import time as timer
from tqdm.autonotebook import tqdm
import json
import datetime
import pytz
import warnings

# localization parameters
min_n_receivers = 5
max_receiver_dist = 60
cc_threshold = 0.01  # 0
skip_if_json_exists = True

# Paths
# location of audio dataset, not included in this GitHub repository
data_root = "REDACTED"
deployment_folders = glob(f"{data_root}/SBT*")
detections_path = "REDACTED"  # matches save_dir of 2_select_detections_to_localize
localized_events_path = "REDACTED"  # matches

# table of min/max frequency to bandpass audio to for each species localized
# sp_freq_ranges_path = "REDACTED"
# here, instead of using a table we hard-code the values for Ovenbird

def get_dt(f, offset):
    dt = datetime.datetime.strptime(
        "_".join(Path(f).stem.split("_")[-2:]), "%Y%m%d_%H%M%S"
    ) + datetime.timedelta(seconds=offset)
    tz = pytz.timezone("US/Mountain")
    return tz.localize(dt)


# for deployment_folder in deployment_folders:
for deployment_folder in deployment_folders:

    # deployment_folder = deployment_folders[0]  # do one for now
    deployment_name = Path(deployment_folder).name
    print(f"localizing events for {deployment_name}")

    # configure paths
    # score_files = glob(f"{detections_path}/{deployment_name}/*/*.csv")

    audio_files = glob(f"{deployment_folder}/*/*.wav")
    print(f"found {len(audio_files)} files for {deployment_name}")
    if len(audio_files) < 1:
        continue

    # save json of SpatialEvents here:
    out_dir = f"{localized_events_path}/{deployment_name}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    # create table providing xyz position for each audio file
    # using the rtk_points table, where point_folder_name matches
    # the name of the folder containing an audio file
    recorder_locations = pd.read_csv(
        f"{deployment_folder}/{deployment_name}_rtk_points.csv"
    )
    point_locs = recorder_locations.set_index("point_folder_name")
    file_position_df = pd.DataFrame({"file": audio_files})
    # point name is name of subfolder with audio files in it:
    file_position_df["point"] = file_position_df["file"].apply(
        lambda x: Path(x).parent.stem
    )
    file_position_df["file"] = file_position_df["file"].apply(lambda x: Path(x).name)
    file_position_df["file"] = [
        f"{deployment_folder}/{p.split('_')[0]}/{p}" for p in file_position_df["file"]
    ]
    # if necessary, remove files from recorders missing/incorrect metadata here
    # add coordinates to the table listing every audio file
    file_position_df[["x", "y", "z"]] = [
        point_locs.loc[c][["EASTING", "NORTHING", "Ortho_Ht"]]
        for c in file_position_df["point"]
    ]
    # format to match expectation of SynchronizedRecorderArray file_coords argument:
    # index=file path, columns = x,y,(optional z) position

    file_coords = file_position_df.set_index("file")[["x", "y", "z"]]

    # load detections saved as sparse df
    pickled_detections = (
        f"{detections_path}/{deployment_name}/dets_{deployment_name}_thresh-1.pkl"
    )
    detections = pd.read_pickle(pickled_detections)

    # optionally save csv of detection counts per species
    # sp_counts = detections.astype(int).sum()
    # sp_counts.sort_values(ascending=False).to_csv(
    #     f"./det_cnts/{deployment_name}_det_cnts.csv"
    # )

    # add timestamp into detections df multi-index, since
    # stamp not available in the audio file metadata
    detections = detections.reset_index(drop=False)
    detections["start_timestamp"] = [
        get_dt(f, st) for f, st in zip(detections["file"], detections["start_time"])
    ]
    detections["file"] = [
        f"{deployment_folder}/{p.split('_')[0]}/{p}" for p in detections["file"]
    ]
    detections = detections.set_index(
        ["file", "start_time", "end_time", "start_timestamp"]
    )

    sp_list = ["Ovenbird"]
    detections = detections[sp_list]

    # species_freq_ranges = pd.read_csv(sp_freq_ranges_path).set_index("english_common")
    # bandpass_ranges = {
    #     sp: [species_freq_ranges.at[sp, "low_f"], species_freq_ranges.at[sp, "high_f"]]
    #     for sp in detections.columns
    # }
    bandpass_ranges = {"Ovenbird": [2000, 10000]}

    # warn if species missing bandpass ranges:
    sp_missing_f_range = [
        s  # species_freq_ranges.at[s, "english_common"]
        for s in sp_list
        if np.isnan(bandpass_ranges[s][0]) or bandpass_ranges[s][0] is None
    ]
    if len(sp_missing_f_range) > 0:
        warnings.warn(
            f"{len(sp_missing_f_range)} classes have detections but bandpass range is unspecified: \n{sp_missing_f_range}"
        )

    # create relative positions, to avoid possible numeric convergence issues w/UTM vs elevation
    for c in file_coords.columns:
        file_coords[c] = file_coords[c] - file_coords[c].min()

    # TODO: consider converting back to real UTM/elevation at end

    t0 = timer()
    array = SynchronizedRecorderArray(file_coords)
    for sp in tqdm(sp_list):
        out_file = (
            f'{out_dir}/{deployment_name}_{sp.replace(" ", "_")}_localized_events.json'
        )
        if Path(out_file).exists() and skip_if_json_exists:
            continue
        if bandpass_ranges[sp][0] is None or np.isnan(bandpass_ranges[sp][0]):
            continue  # we'll just skip these classes for now?
        print(f"now localizing {sp} detections")
        localized_events, unlocalized_events = array.localize_detections(
            detections=detections[[sp]],
            max_receiver_dist=max_receiver_dist,
            min_n_receivers=min_n_receivers,
            bandpass_ranges=bandpass_ranges,
            return_unlocalized=True,
            cc_threshold=cc_threshold,
            localization_algorithm="soundfinder",
            num_workers=16,
        )

        if len(localized_events) > 0:
            events_dict = {"localized_events": [e.to_dict() for e in localized_events]}
            with open(out_file, "w+") as f:
                json.dump(events_dict, f)

            print(
                f"saved {len(localized_events)} localized {sp} detections to {out_dir}"
            )
        else:
            print(f"Zero events localized for {sp}")
        print(f"time elapsed: {(timer()-t0)/60} min")
    print("finished all species")


# example of reloading the events:
# from loca_utils import from_dict
# with open('test_saving.json','r') as f:
#     reloaded_events = json.load(f)
# events = [from_dict(d) for d in reloaded_events['localized_events']]

# post-processing filters to apply:
# residual_rms < 1
#
# underground detections:
# distances = np.linalg.norm(
#     e.receiver_locations - e.location_estimate, axis=1
# )
# nearest_recorder_z_position = e.receiver_locations[np.argmin(distances)][2]
# if e.location_estimate[2] + 3 >= nearest_recorder_z_position:# and e.location_estimate[2]-10 <= nearest_recorder_z_position:
#     new_events.append(e)
#
# manual review using same workflow as 2023
