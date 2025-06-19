import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from matplotlib import pyplot as plt


def figsize(w, h):
    plt.rcParams["figure.figsize"] = [w, h]


figsize(15, 5)  # for big visuals
# %config InlineBackend.figure_format = 'retina'
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from opensoundscape.localization.position_estimate import df_to_positions
import ast
import re


def parse_list_of_lists(cell):
    # Replace newline characters and convert the cell string to a list of lists
    cell = (
        re.sub(r"\s+", ",", cell.replace("\n", ""))
        .replace("[,", "[")
        .replace(",,", ",")
    )
    # cell = cell.strip(', ')
    return ast.literal_eval(cell)


from sklearn.cluster import DBSCAN
from IPython.display import clear_output
import seaborn as sns

# paths
event_dir = "REDACTED"  # matches out_dir of 4_automatedd_minspec_review.py
out_dir = "REDACTED"  # matches filtered_events_dir of 6_select_clips_to_anotate.py

# parameters
# within this distance events at same time are considered duplicates
grouping_radius = 10  # meters
# threshold score for Hawkears Ovenbird class on minspec
threshold = -2.5  # logit score

plot = False

for csv in glob(f"{event_dir}/SBT-*_positions.csv"):
    array_name = Path(csv).stem.split("_")[0]
    print(array_name)
    df = pd.read_csv(
        csv,
        index_col=0,
        parse_dates=["start_timestamp"],
        converters={
            "receiver_locations": parse_list_of_lists,
            "receiver_start_time_offsets": parse_list_of_lists,
            "location_estimate": parse_list_of_lists,
            "tdoas": parse_list_of_lists,
            "cc_maxs": parse_list_of_lists,
            "distance_residuals": parse_list_of_lists,
            "receiver_files": parse_list_of_lists,
        },
    )

    # filter by minspec HawkEars classifier score:
    # automatically removes events with bad alignment or wrong localized species
    df = df[df.hawkears_minspec_score_OVEN >= threshold]
    positions = df_to_positions(df.drop(columns=["hawkears_minspec_score_OVEN"]))

    df["start_t_sec"] = df.start_timestamp.apply(lambda x: pd.Timestamp(x).timestamp())
    # df.start_t_sec = df.start_t_sec - df.start_t_sec.min()
    df["x"] = np.array(list(df.location_estimate.values))[:, 0]
    df["y"] = np.array(list(df.location_estimate.values))[:, 1]
    df["residual"] = [p.residual_rms for p in positions]

    df_reduced = df.copy()
    for time, time_group in df.groupby("start_timestamp"):
        # remove duplicates using dbscan
        if len(time_group) < 2:
            continue
        points = np.array(list(time_group.location_estimate))[:, 0:2]
        db = DBSCAN(eps=grouping_radius, min_samples=2)
        db.fit(points)

        time_group["dblabel"] = db.labels_

        for dblabel, label_group in time_group.groupby("dblabel"):
            # these events are very close in space and occur at the same time
            # only retain the one with lowest rms residual, discard the rest
            rows_to_discard = label_group.drop(label_group.residual.idxmin())
            df_reduced.drop(rows_to_discard.index, inplace=True)

        if plot:
            sns.scatterplot(data=time_group, x="x", y="y", hue="dblabel")
            print(points)
            print(db.labels_)
            plt.xlim(-50, 450)
            plt.ylim(-50, 250)
            plt.show()
            if input() == "x":
                break
            clear_output()

    df_reduced.to_csv(f"{out_dir}/{array_name}_events_filtered_no_dup.csv", index=False)
