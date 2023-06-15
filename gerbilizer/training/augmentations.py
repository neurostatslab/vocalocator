import torch
from torch import nn
from audiomentations import AddGaussianSNR, Compose, PolarityInversion, PitchShift, Shift, TimeMask


class AudiomentationsWrapper(torch.nn.Module):
    """ Wrapper for audiomentations' modules to hold the sample rate and support Tensors
    """
    def __init__(self, module, sample_rate: int):
        super().__init__()
        self.aug = module
        self.sample_rate = sample_rate

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return torch.from_numpy(self.aug(x.numpy(), sample_rate=self.sample_rate))
        return self.aug(x, sample_rate=self.sample_rate)


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
    
    if (pitch_config := aug_config.get("PITCH_SHIFT", False)):
        pitch_shift = PitchShift(
            min_semitones=pitch_config.get("MIN_SHIFT_SEMITONES", -2),
            max_semitones=pitch_config.get("MAX_SHIFT_SEMITONES", 2),
            p=pitch_config.get("PROB", 0.5),
        )
        augmentations.append(pitch_shift)
    
    # if (shift_config := aug_config.get("SAMPLE_SHIFT", False)):
    #     # Shifts the audio forward or backward
    #     shift = Shift(
    #         min_shift=shift_config["MIN_SHIFT"],
    #         max_shift=shift_config["MAX_SHIFT"],
    #         shift_unit=shift_config["SHIFT_UNIT"],
    #         p=shift_config["PROB"],
    #         sample_rate=CONFIG["DATA"]["SAMPLE_RATE"],
    #     )
    #     augmentations.append(shift)
    
    if (inversion_config := aug_config.get("INVERSION", False)):
        # Inverts the polarity of the audio
        inversion = PolarityInversion(
            p=inversion_config.get("PROB", 0.5),
        )
        augmentations.append(inversion)
    
    if (noise_config := aug_config.get("NOISE", False)):
        # Adds white background noise to the audio
        noise = AddGaussianSNR(
            min_snr_in_db=noise_config.get("MIN_SNR", 0),
            max_snr_in_db=noise_config.get("MAX_SNR", 10),
            p=noise_config.get("PROB", 0.5),
        )
        augmentations.append(noise)
    
    if (mask_config := aug_config.get("MASK", False)):
        # Masks a random part of the audio
        mask = TimeMask(
            min_band_part=0,
            max_band_part=0.1,
            fade=False,
            p=mask_config.get("PROB", 0.5),
        )
        augmentations.append(mask)
    
    sample_rate=CONFIG["DATA"]["SAMPLE_RATE"]
    return AudiomentationsWrapper(Compose(augmentations), sample_rate=sample_rate)