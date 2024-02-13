from itertools import combinations
from math import comb
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.outputs import ModelOutputFactory

from .simplenet import GerbilizerSimpleLayer


class SkipConnection(torch.nn.Module):
    def __init__(self, submodule: nn.Module):
        super(SkipConnection, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class FrequencyNetwork(GerbilizerArchitecture):
    defaults = {
        "USE_BATCH_NORM": True,
        "OUTPUT_COV": True,
        "REGULARIZE_COV": False,
        "CPS_NUM_LAYERS": 3,
        "CPS_HIDDEN_SIZE": 1024,
    }

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(FrequencyNetwork, self).__init__(CONFIG, output_factory)

        N = CONFIG["DATA"]["NUM_MICROPHONES"]

        self.xcorr_length = 256
        self.nfft = CONFIG["DATA"]["CROP_LENGTH"]

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = FrequencyNetwork.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))
        CONFIG["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        use_batch_norm = model_config["USE_BATCH_NORM"]

        # layers for the cps branch of the network:
        cps_initial_channels = self.xcorr_length * N
        cps_num_layers = model_config["CPS_NUM_LAYERS"]
        cps_hidden_size = model_config["CPS_HIDDEN_SIZE"]

        # Add a batch norm to handle scaling the cps
        cps_network = [
            nn.BatchNorm1d(cps_initial_channels),
            nn.Linear(cps_initial_channels, cps_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(cps_num_layers - 1):
            cps_network.append(
                SkipConnection(
                    nn.Sequential(
                        nn.Linear(cps_hidden_size, cps_hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        self.cps_network = nn.Sequential(*cps_network)

        if not isinstance(self.n_outputs, int):
            raise ValueError(
                "Number of parameters to output is undefined! Maybe check the model configuration and ModelOutputFactory object?"
            )
        self.coord_readout = nn.Sequential(
            nn.Linear(cps_hidden_size, cps_hidden_size),
            nn.ReLU(),
            nn.Linear(cps_hidden_size, self.n_outputs),
        )

    def make_cps(self, audio: torch.Tensor):
        # audio shape: (batch, n_samples, n_channels)
        # Generates the cross power spectrum of the audio channels in a pairwise fasion

        output_length = self.xcorr_length

        seq_len, n_channels = audio.shape[-2:]

        # Get combinations of adjacent channels
        # will have the form (0, 1), (1, 2), (2, 3), (3, 0) for 4 channels
        channel_pairs = torch.stack(
            (
                torch.arange(n_channels),
                ((torch.arange(n_channels) + 1) % n_channels),
            ),
            axis=1,
        )

        rfft_audio = torch.fft.rfft(audio, dim=-2)  # (batch, n_rfreq, n_channels)
        # Compute cross correlations between channel pairs
        ch1 = rfft_audio[..., :, channel_pairs[:, 0]]
        ch2 = rfft_audio[..., :, channel_pairs[:, 1]]
        cps = ch1 * torch.conj(ch2)
        xcorr = torch.fft.irfft(cps, dim=-2)  # (batch, n_samples, n_pairs=nmics)

        # Select the central part of the cross-correlation
        center_idx = seq_len // 2
        start, end = center_idx - output_length // 2, center_idx + output_length // 2
        xcorr = xcorr[..., start:end, :]
        return xcorr.flatten(start_dim=1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        audio = x
        cps = self.make_cps(audio)

        cps_branch = self.cps_network(cps)

        coords = self.coord_readout(cps_branch)
        return coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
