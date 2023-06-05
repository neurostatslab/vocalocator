import logging

from math import comb

import torch
from torch import nn

from gerbilizer.architectures.util import build_cov_output

logging.basicConfig(level=logging.DEBUG)


class GerbilizerSimpleLayer(torch.nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        filter_size,
        *,
        downsample,
        dilation,
        use_bn=True
    ):
        super(GerbilizerSimpleLayer, self).__init__()

        self.fc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            # padding=(filter_size * dilation - 1) // 2,
            padding=0,
            stride=(2 if downsample else 1),
            dilation=dilation,
        )
        self.gc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            # padding=(filter_size * dilation - 1) // 2,
            padding=0,
            stride=(2 if downsample else 1),
            dilation=dilation,
        )
        self.batch_norm = (
            torch.nn.BatchNorm1d(channels_out) if use_bn else nn.Identity()
        )

    def forward(self, x):
        fcx = self.fc(x)
        return self.batch_norm(
            (torch.tanh(fcx) + 0.05 * fcx) * torch.sigmoid(self.gc(x))
        )


def ceiling_division(n, d):
    q, r = divmod(n, d)
    return q + bool(r)


class GerbilizerSimpleNetwork(torch.nn.Module):
    defaults = {
        "USE_BATCH_NORM": True,
        "SHOULD_DOWNSAMPLE": [False, True, True, True, True, True, False],
        "CONV_FILTER_SIZES": [19, 7, 39, 41, 23, 29, 33],
        "CONV_NUM_CHANNELS": [16, 16, 16, 32, 32, 32, 64],
        "CONV_DILATIONS": [1, 1, 1, 1, 1, 1, 1],
        "OUTPUT_COV": True,
        "REGULARIZE_COV": False,
    }

    def __init__(self, CONFIG):
        super(GerbilizerSimpleNetwork, self).__init__()

        N = CONFIG["DATA"]["NUM_MICROPHONES"]

        if CONFIG["DATA"].get("COMPUTE_XCORRS", False):
            N += comb(N, 2)

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = GerbilizerSimpleNetwork.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))

        should_downsample = model_config["SHOULD_DOWNSAMPLE"]
        self.n_channels = model_config["CONV_NUM_CHANNELS"]
        filter_sizes = model_config["CONV_FILTER_SIZES"]
        dilations = model_config["CONV_DILATIONS"]

        min_len = min(
            len(self.n_channels),
            len(filter_sizes),
            len(should_downsample),
            len(dilations),
        )
        self.n_channels = self.n_channels[:min_len]
        filter_sizes = filter_sizes[:min_len]
        should_downsample = should_downsample[:min_len]
        dilations = dilations[:min_len]

        use_batch_norm = model_config["USE_BATCH_NORM"]

        self.n_channels.insert(0, N)

        convolutions = [
            GerbilizerSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=use_batch_norm,
            )
            for in_channels, out_channels, filter_size, downsample, dilation in zip(
                self.n_channels[:-1],
                self.n_channels[1:],
                filter_sizes,
                should_downsample,
                dilations,
            )
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        # Final linear layer to reduce the number of channels.
        # self.coord_readout = torch.nn.Linear(self.n_channels[-1], 2)
        self.output_cov = model_config["OUTPUT_COV"]
        N_OUTPUTS = 5 if self.output_cov else 2

        self.coord_readout = torch.nn.Linear(self.n_channels[-1], N_OUTPUTS)

    def forward(self, x):
        x = x.transpose(
            -1, -2
        )  # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        return build_cov_output(coords, x.device) if self.output_cov else coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
