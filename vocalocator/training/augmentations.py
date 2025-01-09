import torch
from numpy import sqrt
from torch import nn


class Identity(nn.Module):
    """Implements the identity function as a placeholder augmentation"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PolarityInversion(nn.Module):
    """Inverts the polarity of the audio."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) > self.p:
            return x
        return -x


class TimeMask(nn.Module):
    """Time masking data augmentation. Masks a random part of the audio across all channels.
    Mutates the input tensor in-place. A bit slow because it has to do a lot of indexing.
    """

    def __init__(
        self, min_mask_length: int = 0, max_mask_length: int = 0, prob: float = 0.5
    ):
        super().__init__()
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length
        self.prob = prob

    def forward(self, x):
        if x.dim() <= 2:
            # unbatched input
            if torch.rand() > self.prob:
                return x
            num_samples, num_channels = x.shape
            mask_start = torch.randint(0, num_samples - self.max_mask_length, (1,))
            mask_end = mask_start + torch.randint(
                self.min_mask_length, self.max_mask_length, (1,)
            )
            x[mask_start:mask_end, :] = 0
            return x
        # Batched input
        bshape = x.shape[:-2]
        num_samples, num_channels = x.shape[-2:]
        x = x.reshape(-1, num_samples, num_channels)
        bsz = x.shape[0]

        prob_mask = torch.rand(bsz) < self.prob  # True where we should apply the mask
        prob_idx = torch.nonzero(prob_mask).squeeze(1)
        num_true = prob_idx.shape[0]
        mask_start = torch.randint(0, num_samples - self.max_mask_length, (num_true,))
        mask_lengths = torch.randint(
            self.min_mask_length, self.max_mask_length, (num_true,)
        )
        mask_end = mask_start + mask_lengths
        for i, start, end in zip(prob_idx, mask_start, mask_end):
            x[i, start:end, :] = 0
        return x.reshape(*bshape, num_samples, num_channels)


class AddWhiteNoise(nn.Module):
    """Add white noise to the input tensor"""

    def __init__(
        self, min_snr_in_db: float = 0, max_snr_in_db: float = 10, p: float = 0.5
    ):
        super().__init__()
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds white noise to the input tensor

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Augmented tensor
        """
        # Assuming x has shape (*batch_size, num_samples, num_channels)
        if torch.rand(1) > self.p:
            return x
        snr = (
            torch.rand(*x.shape[:-2]) * (self.max_snr_in_db - self.min_snr_in_db)
            + self.min_snr_in_db
        )  # Shape: (*batch_size)
        snr = snr.to(x.device)

        noise = torch.randn_like(x)
        signal_power = torch.sum(x**2, dim=(-2, -1))  # Shape: (*batch_size)
        orig_noise_power = torch.sqrt(
            torch.sum(noise**2, dim=(-1, -2))
        )  # Shape: (*batch_size)
        new_noise_power = torch.sqrt(
            signal_power / 10 ** (snr / 10)
        )  # Shape: (*batch_size)
        noise = noise * new_noise_power[:, None, None] / orig_noise_power[:, None, None]
        return x + noise


class AddLowFreqNoise(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, audio: torch.Tensor):
        """Adds low frequency noise to an audio tensor of shape (B, T, C)"""
        if torch.rand(1) > self.p:
            return audio

        rand_shape = (audio.shape[0], audio.shape[1] // 2 + 1, audio.shape[2])
        spectrum = torch.rand(rand_shape) * torch.exp(
            -1.0j * torch.rand(rand_shape) * 2 * torch.pi
        )
        spectrum = spectrum.to(audio.device)
        freq_decay = torch.exp(-2 * torch.linspace(0, 1, spectrum.shape[-2])).to(
            audio.device
        )
        freq_decay[0] = 0

        noise = torch.fft.irfft(spectrum * freq_decay[None, :, None], dim=-2)
        noise = (noise - noise.mean()) / noise.std()
        mix = (audio + noise) / sqrt(2)
        return mix


def build_augmentations(CONFIG: dict) -> nn.Module:
    """Builds an augmentation module using the config parameters
    The augmentation module is a subclass of nn.Module which takes in a batched data tensor
    and returns a batched data tensor with the augmentations applied.

    Args:
        CONFIG (dict): Whole config dictionary

    Returns:
        nn.Module: Augmemtation module
    """
    augmentations = []
    aug_config = CONFIG["AUGMENTATIONS"]

    if not CONFIG["DATA"]["AUGMENT_DATA"]:
        return Identity()

    if inversion_config := aug_config.get("INVERSION", False):
        # Inverts the polarity of the audio
        inversion = PolarityInversion(
            p=inversion_config.get("PROB", 0.5),
        )
        augmentations.append(inversion)

    if noise_config := aug_config.get("NOISE", False):
        # Adds white background noise to the audio
        noise = AddWhiteNoise(
            min_snr_in_db=noise_config.get("MIN_SNR", 0),
            max_snr_in_db=noise_config.get("MAX_SNR", 10),
            p=noise_config.get("PROB", 0.5),
        )
        augmentations.append(noise)

    if low_freq_noise_config := aug_config.get("LOW_FREQ_NOISE", False):
        # Adds low frequency noise to the audio
        low_freq_noise = AddLowFreqNoise(
            p=low_freq_noise_config.get("PROB", 0.5),
        )
        augmentations.append(low_freq_noise)

    if mask_config := aug_config.get("MASK", False):
        # Masks a random part of the audio
        mask = TimeMask(
            min_mask_length=mask_config.get("MIN_LENGTH", 75),
            max_mask_length=mask_config.get("MAX_LENGTH", 250),
            prob=mask_config.get("PROB", 0.5),
        )
        augmentations.append(mask)

    return nn.Sequential(*augmentations)
