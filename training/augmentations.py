from torch_audiomentations import Compose, PitchShift, Shift


def build_augmentations(CONFIG):

    if CONFIG["AUGMENT_DATA"]:

        return Compose([
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
            )
            # AddGaussianNoise(
            #     min_amplitude=CONFIG["AUGMENT_GAUSS_MIN"],
            #     max_amplitude=CONFIG["AUGMENT_GAUSS_MAX"],
            #     p=CONFIG["AUGMENT_GAUSS_PROB"]
            # ),
        ])

    else:
        def identity(x, sample_rate=None):
            return x
        return identity # No augmentations.
