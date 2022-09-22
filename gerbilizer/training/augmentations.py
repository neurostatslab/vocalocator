import torch
from torch_audiomentations import Compose, Shift, PolarityInversion


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).cuda() * self.std


def build_augmentations(CONFIG):

    if CONFIG["AUGMENT_DATA"]:

        return Compose(
            [
                # TimeStretch(
                #     min_rate=CONFIG["AUGMENT_STRETCH_MIN"],
                #     max_rate=CONFIG["AUGMENT_STRETCH_MAX"],
                #     p=CONFIG["AUGMENT_STRETCH_PROB"]
                # ),
                # PitchShift(
                #     min_transpose_semitones=CONFIG["AUGMENT_PITCH_MIN"],
                #     max_transpose_semitones=CONFIG["AUGMENT_PITCH_MAX"],
                #     p=CONFIG["AUGMENT_PITCH_PROB"],
                #     sample_rate=CONFIG["AUDIO_SAMPLE_RATE"],
                #     mode="per_example",
                # ),
                Shift(
                    min_shift=CONFIG["AUGMENT_SHIFT_MIN"],
                    max_shift=CONFIG["AUGMENT_SHIFT_MAX"],
                    shift_unit="fraction",
                    p=CONFIG["AUGMENT_SHIFT_PROB"],
                    mode="per_example",
                ),
                PolarityInversion(
                    p=CONFIG["AUGMENT_INVERSION_PROB"],
                    mode="per_example",
                ),
                # AddGaussianNoise(CONFIG["AUGMENT_GAUSS_NOISE"])
            ]
        )

    else:

        def identity(x, sample_rate=None):
            return x

        return identity  # No augmentations.
