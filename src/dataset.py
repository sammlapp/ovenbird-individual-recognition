import numpy as np
import pandas as pd
from torch.utils.data import Sampler, Dataset, RandomSampler
from opensoundscape.annotations import categorical_to_multi_hot
from opensoundscape.sample import AudioSample


class AIIDLocalizedClipDataset(Dataset):
    def __init__(
        self, aiid_df, preprocessor, bypass_augmentations=False, unique_labels=None
    ):
        self.aiid_df = aiid_df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.bypass_augmentations = bypass_augmentations
        if unique_labels is None and "aiid_label" in self.aiid_df.columns:
            unique_labels = self.aiid_df["aiid_label"].unique()
            # check for non-integer labels, 'n' 'u' etc?
            if not all(
                [isinstance(label, (int, np.integer)) for label in unique_labels]
            ):
                raise ValueError("Labels must be integers.")
        self.aiid_label_list = unique_labels

        # initialize empty pseudo-labels if not provided
        # -1 means not belonging to any cluster
        if "pseudo_label" not in self.aiid_df.columns:
            self.aiid_df["pseudo_label"] = -1

        # create sparse df for labels

        # determine clip starts and ends from center-of-song time
        clip_duration = self.preprocessor.sample_duration
        start_times = self.aiid_df.song_center_time - clip_duration / 2
        end_times = start_times + clip_duration

        # make one hot labels clip dataframe
        index = pd.DataFrame(
            {
                "file": self.aiid_df.file,
                "start_time": start_times,
                "end_time": end_times,
            }
        ).set_index(["file", "start_time", "end_time"])
        # labels = torch.nn.functional.one_hot(
        #     self.aiid_df.aiid_label, len(self.aiid_label_list)
        # )
        if "aiid_label" in self.aiid_df.columns:
            multihot_labels_sp, _ = categorical_to_multi_hot(
                [[a] for a in aiid_df["aiid_label"].values], unique_labels, sparse=True
            )
            self.label_df = pd.DataFrame.sparse.from_spmatrix(
                multihot_labels_sp,
                index=index.index,
                columns=unique_labels,
            )
        else:  # no labels, just clip file/start/end df
            self.label_df = pd.DataFrame(index=index.index)

    def __getitem__(self, idx, break_on_key=None, break_on_type=None):
        if not isinstance(idx, int):
            raise ValueError(
                f"idx must be an integer, got {type(idx)}. "
                f"This could happen if you specified a custom sampler that results in returning "
                "lists of indices rather than a single index. AIIDLocalizedClipDataset.__getitem__ "
                "requires that idx is a single integer index."
            )
        sample = AudioSample.from_series(self.label_df.iloc[idx])

        # preprocessor.forward will raise PreprocessingError if something fails
        sample = self.preprocessor.forward(
            sample,
            bypass_augmentations=self.bypass_augmentations,
            break_on_key=break_on_key,
            break_on_type=break_on_type,
        )

        sample.idx = idx
        if "array" in self.aiid_df.columns:
            sample.grid_id = self.aiid_df.iloc[idx]["array"]
        if "event_id" in self.aiid_df.columns:
            sample.event_id = self.aiid_df.iloc[idx]["event_id"]
        if "pseudo_label" in self.aiid_df.columns:
            sample.pseudo_label = self.aiid_df.iloc[idx]["pseudo_label"]
        if "aiid_label" in self.aiid_df.columns:
            sample.aiid_label = self.aiid_df.iloc[idx]["aiid_label"]

        return sample

    def __len__(self):
        return len(self.label_df)


class GridAsLabelDataset(Dataset):
    """dataset that uses the array/localization grid ID as the label"""

    def __init__(
        self, aiid_df, preprocessor, bypass_augmentations=False, unique_grids=None
    ):
        self.aiid_df = aiid_df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.bypass_augmentations = bypass_augmentations
        if unique_grids is None and "array" in self.aiid_df.columns:
            unique_grids = self.aiid_df["array"].unique()
        self.grid_list = list(unique_grids)

        # initialize empty pseudo-labels if not provided
        # -1 means not belonging to any cluster
        if "pseudo_label" not in self.aiid_df.columns:
            self.aiid_df["pseudo_label"] = -1

        # create sparse df for labels

        # determine clip starts and ends from center-of-song time
        clip_duration = self.preprocessor.sample_duration
        start_times = self.aiid_df.song_center_time - clip_duration / 2
        end_times = start_times + clip_duration
        index = pd.DataFrame(
            {
                "file": self.aiid_df.file,
                "start_time": start_times,
                "end_time": end_times,
            }
        ).set_index(["file", "start_time", "end_time"])

        self.label_df = pd.DataFrame(
            {"grid_id": self.aiid_df["array"].values},
            index=index.index,
        )

    def __getitem__(self, idx, break_on_key=None, break_on_type=None):
        if not isinstance(idx, int):
            raise ValueError(
                f"idx must be an integer, got {type(idx)}. "
                f"This could happen if you specified a custom sampler that results in returning "
                "lists of indices rather than a single index. AIIDLocalizedClipDataset.__getitem__ "
                "requires that idx is a single integer index."
            )
        sample = AudioSample.from_series(self.label_df.iloc[idx])

        sample.idx = idx
        sample.grid_id = self.aiid_df.iloc[idx]["array"]
        # integer label for the grid_id
        sample.numeric_label = self.grid_list.index(sample.grid_id)

        # preprocessor.forward will raise PreprocessingError if something fails
        sample = self.preprocessor.forward(
            sample,
            bypass_augmentations=self.bypass_augmentations,
            break_on_key=break_on_key,
            break_on_type=break_on_type,
        )

        sample.event_id = self.aiid_df.iloc[idx]["event_id"]
        sample.pseudo_label = self.aiid_df.iloc[idx]["pseudo_label"]
        if "aiid_label" in self.aiid_df.columns:
            sample.aiid_label = self.aiid_df.iloc[idx]["aiid_label"]

        return sample

    def __len__(self):
        return len(self.label_df)

class PointCodeDataset(Dataset):
    """dataset that adds .clip_id and .point_code to preprocessed samples
    
    clip_df: (file,start_time,end_time) multi-index; requires 'point_code' column

    
    """

    def __init__(
        self, clip_df, preprocessor, bypass_augmentations=False
    ):
        self.clip_df = clip_df.copy()
        self.preprocessor = preprocessor
        self.bypass_augmentations = bypass_augmentations
        self.unique_points = self.clip_df['point_code'].unique()

    def __getitem__(self, idx, break_on_key=None, break_on_type=None):
        """returns preprocessed sample with .point_code and .clip_id (= numeric indexer)"""
        if not isinstance(idx, int):
            raise ValueError(
                f"idx must be an integer, got {type(idx)}. "
                f"This could happen if you specified a custom sampler that results in returning "
                "lists of indices rather than a single index. AIIDLocalizedClipDataset.__getitem__ "
                "requires that idx is a single integer index."
            )
        sample = AudioSample.from_series(self.clip_df.iloc[idx])
        
        sample = self.preprocessor.forward(
            sample,
            bypass_augmentations=self.bypass_augmentations,
            break_on_key=break_on_key,
            break_on_type=break_on_type,
        )

        sample.clip_id = idx
        sample.point_code = self.clip_df.iloc[idx]["point_code"]
        if "pseudo_label" in self.clip_df.columns:
            sample.pseudo_label = self.clip_df.iloc[idx]['pseudo_label']
        
        return sample

    def __len__(self):
        return len(self.clip_df)

class TrainingEventSampler(Sampler):
    r"""Create batches where all clips from an event are in the same batch

    Beyond this, a batch should contain clips from different events and different arrays.

    Args:
        data: Dataset for building sampling logic.
        batch_size: Size of mini-batch.
    """

    def __init__(self, aiid_df, batch_size):
        # build data for sampling here
        self.aiid_df = aiid_df.reset_index()
        self.batch_size = batch_size

        # get unique event_ids and array_ids
        self.event_ids = self.aiid_df["event_id"].unique()
        self.array_ids = self.aiid_df["array"].unique()

    def __iter__(self):
        # implement logic of sampling here
        # shuffle the event_ids
        np.random.shuffle(self.event_ids)

        batch = []
        for i, event_id in enumerate(self.event_ids):

            # get all the clips for this event
            clip_idxs = self.aiid_df[self.aiid_df["event_id"] == event_id].index

            if len(batch) + len(clip_idxs) >= self.batch_size:
                # we can't fit all the clips from the next event in this batch
                # yeild the batch
                yield batch
                batch = []

            batch.extend(list(clip_idxs))

            # if this is the last event, yield the batch
            if i == len(self.event_ids) - 1:
                yield batch

    def __len__(self):
        return len(self.aiid_df) // self.batch_size  # this could be an under-estimate!

class MultipleClipVersionsSampler(RandomSampler):

    r"""Create batches with multiple copies of same clip (can receive different augmentation)

    Args:
        train_df: clip dataframe with (file, start_time, end_time, point_code) columns
        batch_size: Size of mini-batch.
        n_versions: number of augmented variants to create for each training clip
    """

    def __init__(self, data_source, n_versions, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.indices = np.arange(len(data_source))
        self.n_versions = n_versions

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        duplicated_indices = np.repeat(self.indices, self.n_versions).astype(int)  
        return iter(map(int, duplicated_indices))

    def __len__(self):
        return len(self.data_source) * self.n_versions
    

class PointCodeSampler(Sampler):
    r"""Create batches where a batch has a handful of samples from each of a handful of points

    optionally includes replicates of identical indices, for comparing augmented variants of same sample

    Args:
        clip_df: dataframe with 'point_code'
        batch_size: Size of mini-batch.
        n_points_per_batch: number of unique points included in batch
        n_clip_replicates: number of repetitions of same 
    
    """

    def __init__(self, clip_df, batch_size,n_points_per_batch,n_clip_replicates):
        # want batch_size / (n_points_per_batch * n_clip_replicates) to be an integer

        self.clip_df = clip_df.reset_index()
        self.batch_size = batch_size
        self.n_points_per_batch = n_points_per_batch
        self.n_clip_replicates = n_clip_replicates

        # get unique point_code values
        self.point_codes = self.clip_df["point_code"].unique()

    def __iter__(self):
    
        n_clips_per_point_in_batch = self.batch_size // (self.n_points_per_batch*self.n_clip_replicates)

        batch = []
        # iterate through all points several times to cover approx all of the samples in the dataset
        # num samples used per iteration over all points: num points * n_clips_per_point_in_batch
        # num iterations: num_samples // num samples per iteration
        n_iter_over_all_points = len(self.clip_df) //  (len(self.point_codes)*n_clips_per_point_in_batch)
        print(f"will iterate over points {n_iter_over_all_points} times in one epoch")
        print(f"len self.clip_df: {len(self.clip_df)}, batch size {self.batch_size}, len(self):{len(self)}")
        print(f"num points: {len(self.point_codes)}, per batch: {self.n_points_per_batch}")
        # count uses of each clip for use in determining sampling probability
        self.clip_df['n_uses'] = 0

        for all_points_iter in range(n_iter_over_all_points):
            # iterates over each point, selecting random samples from n_points_per_batch points
            # for each batch, and including n_clip_replicates copies of identical indices in the batch

            # shuffle the order of the point_codes before iterating through them
            np.random.shuffle(self.point_codes)

            # for each iteration over all points, we choose n_clips_per_point_in_batch clips from the point
            for i, point_code in enumerate(self.point_codes):
                # subset clip df to this point's detections
                point_idxs = self.clip_df[self.clip_df["point_code"] == point_code].index.values

                # define sampling probabilities for each clip: start equal, multipy by 0.01 when used in a batch
                # this allows us to generally use new clips, but use old clips if we run out of new ones
                sampling_probs = 0.01**self.clip_df.loc[point_idxs]['n_uses']
                sampling_probs = sampling_probs/sum(sampling_probs) # should sum to 1

                # randomly select clips from the point to include in the batch
                if len(point_idxs)>n_clips_per_point_in_batch:
                    point_idxs=np.random.choice(point_idxs,n_clips_per_point_in_batch,replace=False,p=sampling_probs)

                # repeat the same clip in the batch, for augmentation variants
                point_idxs = np.repeat(point_idxs, self.n_clip_replicates).astype(int)  
                # cast np.int64 to list of int
                point_idxs = list(map(int, point_idxs))

                if len(batch) + len(point_idxs) >= self.batch_size:
                    # we can't fit all the clips in this batch
                    # yeild the batch, keeping these indices for the next batch
                    yield batch
                    batch = []

                batch.extend(point_idxs)

                # if we reach the last point, and we haven't filled the batch
                # discard 
                if i == len(self.point_codes) - 1:
                    #yield batch # could provide incomplete batch instead
                    continue 

    def __len__(self):
        return len(self.clip_df)//self.batch_size