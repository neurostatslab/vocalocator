from math import comb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .util import build_cov_output


class GerbilizerDenseLayer(torch.nn.Module):
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
        super(GerbilizerDenseLayer, self).__init__()

        if not downsample:
            padding = ((filter_size - 1) * dilation // 2,)
        else:
            # L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # for L_out to be L_in / 2, we need padding = (kernel_size - 1) / 2
            padding = (filter_size) * dilation // 2

        self.fc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            padding=padding,
            stride=(2 if downsample else 1),
            dilation=dilation,
        )
        self.gc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            padding=padding,
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


class GerbilizerDenseNet(torch.nn.Module):
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
        super(GerbilizerDenseNet, self).__init__()

        # Initial number of audio channels.
        N = CONFIG["DATA"]["NUM_MICROPHONES"]
        if CONFIG["DATA"].get("COMPUTE_XCORRS", False):
            N += comb(N, 2)

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = GerbilizerDenseNet.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))
        CONFIG[
            "MODEL_PARAMS"
        ] = model_config  # Save the parameters used in this run for backward compatibility

        should_downsample = model_config["SHOULD_DOWNSAMPLE"]
        n_channels = model_config["CONV_NUM_CHANNELS"]
        filter_sizes = model_config["CONV_FILTER_SIZES"]
        dilations = model_config["CONV_DILATIONS"]

        min_len = min(
            len(n_channels),
            len(filter_sizes),
            len(should_downsample),
            len(dilations),
        )
        n_channels = n_channels[:min_len]
        filter_sizes = filter_sizes[:min_len]
        should_downsample = should_downsample[:min_len]
        self.should_downsample = should_downsample
        dilations = dilations[:min_len]

        use_batch_norm = model_config["USE_BATCH_NORM"]

        n_channels.insert(0, N)
        cumulative_channels = np.cumsum(n_channels)

        self.simple_layers = torch.nn.ModuleList()

        for in_channels, out_channels, filter_size, dilation, downsample in zip(
            cumulative_channels[:-1],
            n_channels[1:],
            filter_sizes,
            dilations,
            should_downsample,
        ):
            simple_layer = GerbilizerDenseLayer(
                in_channels,
                out_channels,
                filter_size,
                dilation=dilation,
                downsample=downsample,
                use_bn=False,
            )

            if not use_batch_norm:
                self.simple_layers.append(simple_layer)
            else:
                self.simple_layers.append(
                    nn.Sequential(
                        simple_layer,
                        nn.BatchNorm1d(out_channels),
                    )
                )

        self.final_pooling = torch.nn.AdaptiveAvgPool1d(1)

        self.output_cov = model_config["OUTPUT_COV"]
        N_OUTPUTS = 5 if self.output_cov else 2

        self.coord_readout = torch.nn.Linear(cumulative_channels[-1], N_OUTPUTS)

    def forward(self, x):
        x = x.transpose(-1, -2)  # new shape: (batch, channels, time)
        for layer, ds in zip(self.simple_layers, self.should_downsample):
            proc_x = layer(x)
            if ds:
                x = F.max_pool1d(x, 2, 2)
            x = torch.cat([x, proc_x], dim=1)
        x = torch.squeeze(self.final_pooling(x), dim=-1)
        coords = self.coord_readout(x)
        return build_cov_output(coords, x.device) if self.output_cov else coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
