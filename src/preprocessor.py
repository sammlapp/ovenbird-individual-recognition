import opensoundscape as opso
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.audio import Audio
from opensoundscape.preprocess.actions import Action, BaseAction, register_action_cls
from opensoundscape.preprocess.action_functions import register_action_fn
import pandas as pd
import random


@register_action_fn
def noise_and_mute(a, noise_level_dBFS):
    # mute audio outside range of 1-2 seconds
    a = opso.audio.concat(
        [
            Audio.silence(1, sample_rate=a.sample_rate),
            a.trim(1, 2),
            Audio.silence(1, sample_rate=a.sample_rate),
        ]
    )
    # add noise
    a = a.normalize()
    noise = Audio.noise(
        duration=a.duration,
        dBFS=noise_level_dBFS,
        color="white",
        sample_rate=a.sample_rate,
    )
    return opso.audio.mix([a, noise])


@register_action_fn
def mute_and_normalize(a):
    # mute audio outside central 1 second (clip is 3s)
    a = opso.audio.concat(
        [
            Audio.silence(1, sample_rate=a.sample_rate),
            a.trim(1, 2),
            Audio.silence(1, sample_rate=a.sample_rate),
        ]
    )
    # normalize the audio signal
    a = a.normalize()
    return a


@register_action_cls
class JitterClipTime(BaseAction):
    """randomly modify the offset time of the audio to extract from a file

    Args:
        max_shift: maximum amount (seconds) to shift clip start time

    """

    def __init__(self, max_shift=0.25):
        super().__init__()
        self.params["max_shift"] = max_shift
        self.is_augmentation = True

    def __call__(self, sample):
        if sample.start_time is None:
            return  # no effect
        # move start time by uniform random amount
        sample.start_time += random.uniform(
            -self.params["max_shift"], self.params["max_shift"]
        )
        # don't allow start time < 0
        sample.start_time = max(sample.start_time, 0)


class OvenbirdPreprocessor(SpectrogramPreprocessor):
    def __init__(self, overlay_df=None, noisereduce_kwargs=None):
        super().__init__(sample_duration=2, overlay_df=overlay_df)
        self.width = None
        self.height = None
        self.pipeline.bandpass.set(min_f=2000, max_f=10000)
        self.pipeline.to_spec.set(overlap_fraction=0.5)

        self.insert_action(
            "normalize",
            Action(Audio.normalize, is_augmentation=False),
            after_key="trim_audio",
        )
        self.pipeline.frequency_mask.bypass = True

        # self.insert_action(
        #     "noise_reduce",
        #     after_key="trim_audio",
        #     action=Action(
        #         Audio.reduce_noise,
        #         is_augmentation=False,
        #         noisereduce_kwargs=noisereduce_kwargs,
        #     ),
        # )

        # replace random affine and random trim with true jitter of
        # what audio time period is selected from longer file

        self.insert_action(
            "time_jitter",
            JitterClipTime(max_shift=0.5),
            before_key="load_audio",
        )
        self.remove_action("random_affine")
        self.remove_action("random_trim_audio")

        # self.pipeline.add_noise.set(std=0.05)

        self.pipeline.overlay.set(
            overlay_prob=0.75, overlay_weight=[0.01, 0.6]
        )  # irrelevant if use_overlay is False

        # resample audio to 32 kHz if necessary
        # also avoids warnings about loading audio metadata
        self.pipeline.load_audio.set(
            sample_rate=32000, load_metadata=False, out_of_bounds_mode="ignore"
        )
