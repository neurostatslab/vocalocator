from math import comb

import torch
from torch import nn


class GerbilizerSimpleLayer(torch.nn.Module):

    def __init__(
            self, channels_in, channels_out, filter_size, *,
            downsample, dilation, use_bn=True
        ):
        super(GerbilizerSimpleLayer, self).__init__()

        self.fc = torch.nn.Conv1d(
            channels_in, channels_out, filter_size,
            padding=(filter_size * dilation - 1) // 2,
            stride=(2 if downsample else 1),
            dilation=dilation
        )
        self.gc = torch.nn.Conv1d(
            channels_in, channels_out, filter_size,
            padding=(filter_size * dilation - 1) // 2,
            stride=(2 if downsample else 1),
            dilation=dilation
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels_out) if use_bn else nn.Identity()

    def forward(self, x):
        fcx = self.fc(x)
        return self.batch_norm(
            (torch.tanh(fcx) + 0.05 * fcx) * torch.sigmoid(self.gc(x))
        )

def ceiling_division(n, d):
    q, r = divmod(n, d)
    return q + bool(r)


class GerbilizerSimpleNetwork(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerSimpleNetwork, self).__init__()

        T = CONFIG["SAMPLE_LEN"]
        N = CONFIG["NUM_MICROPHONES"]

        if CONFIG['COMPUTE_XCORRS']:
            N += comb(N, 2)

        should_downsample = CONFIG['SHOULD_DOWNSAMPLE']
        n_channels = CONFIG['CONV_NUM_CHANNELS']  # Converting this to a JSON array in the config for convenience
        n_channels.insert(0, N)
        filter_sizes = CONFIG['CONV_FILTER_SIZES']  # Also making this an array, along with the others
        dilations = CONFIG['CONV_DILATIONS']
        use_batch_norm = CONFIG['USE_BATCH_NORM']
        convolutions = [
            GerbilizerSimpleLayer(
                in_channels, out_channels, filter_size, downsample=downsample, dilation=dilation, use_bn=use_batch_norm
            )
            for in_channels, out_channels, filter_size, downsample, dilation
            in zip(n_channels[:-1], n_channels[1:], filter_sizes, should_downsample, dilations)
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        self.final_pooling = torch.nn.Conv1d(
            n_channels[-1],
            n_channels[-1],
            kernel_size=ceiling_division(T, 32),
            groups=n_channels[-1],
            padding=0
        )

        # Final linear layer to reduce the number of channels.
        self.coord_readout = torch.nn.Linear(
            n_channels[-1],
            2
        )

    def forward(self, x):

        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        return coords
