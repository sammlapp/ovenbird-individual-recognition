# find Ovenbird detections with score > 1, removing detections where other species are detected
# for 4 years of Pennsylvania passive acoustic monitoring data

from glob import glob
import pandas as pd
from tqdm.autonotebook import tqdm
from pathlib import Path
import joblib

# find clean predictions with no other species detected
ds_paths = {
    # REDACTED
}
jobs = 10

save_dir = "REDACTED"
focal_sp_threshold = 1.0
other_sp_threshold = 0.0


def get_file_dets(p):
    oven_dets = pd.read_csv(p)
    # threshold by Ovenbird score
    oven_dets = oven_dets[oven_dets["Ovenbird"] > focal_sp_threshold]
    # drop dets with other species
    oven_dets = oven_dets[
        oven_dets.drop(columns=["Ovenbird", "file", "start_time", "end_time"]).max(
            axis=1
        )
        < other_sp_threshold
    ][["file", "start_time", "Ovenbird"]]
    oven_dets["start_time"] = oven_dets["start_time"].astype(int)
    return oven_dets


import contextlib
import joblib
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


for dataset in ds_paths.keys():
    print(dataset)
    save_path = f"{save_dir}/oven_dets_{dataset}.csv"
    if Path(save_path).exists():
        continue  # don't repeat if already completed
    preds_paths = glob(f"{ds_paths[dataset]}/*.csv")
    with tqdm_joblib(
        tqdm(desc="Load and subset files", total=len(preds_paths))
    ) as progress_bar:
        all_oven_dets = joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(get_file_dets)(p) for p in preds_paths
        )

    all_oven_dets = pd.concat(all_oven_dets)

    print(
        f"number of detections of only Ovenbird (threshold: {focal_sp_threshold}) in {dataset}: {len(all_oven_dets)}"
    )
    # save
    all_oven_dets.round(3).to_csv(save_path, index=False)
