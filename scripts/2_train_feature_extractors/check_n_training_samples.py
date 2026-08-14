import numpy as np
import pandas as pd
import yaml

with open(
    "oven_aiid/develop_and_evaluate_aiid/4_train_aiid/train_configs/base.yml", "r"
) as f:
    config = yaml.safe_load(f)

train_df = pd.read_csv(
    f"{config['paths']['train_clips_path']}/ovenbird_train_clips.csv"
)

medians, maxes, mins = [], [], []
for experiment_repeat in range(config["repeats"]):
    # potentially subset the training data to a smaller size
    points = list(train_df.point_code.unique())
    points = np.random.choice(points, 64, replace=False)
    train_df = train_df[train_df["point_code"].apply(lambda x: x in points)]
    n_per_point = train_df.groupby("point_code").size()
    medians.append(n_per_point.median())
    maxes.append(n_per_point.max())
    mins.append(n_per_point.min())
print(medians, maxes, mins)
