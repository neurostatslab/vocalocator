import torch
from torch import nn
from torch_audiomentations import AddColoredNoise, Compose, Shift, PolarityInversion, PitchShift


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, snr):
        super().__init__()
        self.snr = snr

    def forward(self, x):
        pass


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def build_augmentations(CONFIG):
    augmentations = []
    aug_config = CONFIG['AUGMENTATIONS']

    if not CONFIG["DATA"]["AUGMENT_DATA"]:
        return Identity()
    
    # TODO: This might not be implemented for our sample rate, double check later
    # if (pitch_config := aug_config.get("PITCH_SHIFT", False)):
    #     pitch_shift = PitchShift(
    #         sample_rate=CONFIG["DATA"]["SAMPLE_RATE"],
    #         min_transpose_semitones=pitch_config["MIN_SHIFT_SEMITONES"],
    #         max_transpose_semitones=pitch_config["MAX_SHIFT_SEMITONES"],
    #         p=pitch_config["PROB"],
    #         mode="per_example",
    #     )
    #     augmentations.append(pitch_shift)
    
    if (shift_config := aug_config.get("SAMPLE_SHIFT", False)):
        # Shifts the audio forward or backward
        shift = Shift(
            min_shift=shift_config["MIN_SHIFT"],
            max_shift=shift_config["MAX_SHIFT"],
            shift_unit=shift_config["SHIFT_UNIT"],
            p=shift_config["PROB"],
            mode="per_example",
            sample_rate=CONFIG["DATA"]["SAMPLE_RATE"],
        )
        augmentations.append(shift)
    
    if (inversion_config := aug_config.get("INVERSION", False)):
        # Inverts the polarity of the audio
        inversion = PolarityInversion(
            p=inversion_config["PROB"],
            mode="per_example",
            sample_rate=CONFIG["DATA"]["SAMPLE_RATE"],
        )
        augmentations.append(inversion)
    
    if (noise_config := aug_config.get("NOISE", False)):
        # Adds white background noise to the audio
        noise = AddColoredNoise(
            min_snr_in_db=noise_config["MIN_SNR"],
            max_snr_in_db=noise_config["MAX_SNR"],
            min_f_decay=0,  # 0 for white noise
            max_f_decay=2,  # 0 for white noise
            p=noise_config["PROB"],
            sample_rate=CONFIG["DATA"]["SAMPLE_RATE"],
        )
        augmentations.append(noise)

    return Compose(augmentations)