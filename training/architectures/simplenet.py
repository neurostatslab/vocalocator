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

        should_downsample = CONFIG['SHOULD_DOWNSAMPLE']
        n_channels = CONFIG['CONV_NUM_CHANNELS']  # Converting this to a JSON array in the config for convenience
        a = list()
        n_channels.insert(0, N)
        filter_sizes = CONFIG['CONV_FILTER_SIZES']  # Also making this an array, along with the others
        dilations = CONFIG['CONV_DILATIONS']
        convolutions = [
            GerbilizerSimpleLayer(
                in_channels, out_channels, filter_size, downsample=downsample, dilation=dilation
            )
            for in_channels, out_channels, filter_size, downsample, dilation
            in zip(n_channels[:-1], n_channels[1:], filter_sizes, should_downsample, dilations)
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)
        """
        self.conv_layers = torch.nn.Sequential(
            GerbilizerSimpleLayer(
                CONFIG["NUM_MICROPHONES"],
                CONFIG["NUM_CHANNELS_LAYER_1"],
                CONFIG["FILTER_SIZE_LAYER_1"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_1"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_1"],
                CONFIG["NUM_CHANNELS_LAYER_2"],
                CONFIG["FILTER_SIZE_LAYER_2"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_2"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_2"],
                CONFIG["NUM_CHANNELS_LAYER_3"],
                CONFIG["FILTER_SIZE_LAYER_3"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_3"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_3"],
                CONFIG["NUM_CHANNELS_LAYER_4"],
                CONFIG["FILTER_SIZE_LAYER_4"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_4"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_4"],
                CONFIG["NUM_CHANNELS_LAYER_5"],
                CONFIG["FILTER_SIZE_LAYER_5"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_5"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_5"],
                CONFIG["NUM_CHANNELS_LAYER_6"],
                CONFIG["FILTER_SIZE_LAYER_6"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_6"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_6"],
                CONFIG["NUM_CHANNELS_LAYER_7"],
                CONFIG["FILTER_SIZE_LAYER_7"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_7"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_7"],
                CONFIG["NUM_CHANNELS_LAYER_8"],
                CONFIG["FILTER_SIZE_LAYER_8"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_8"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_8"],
                CONFIG["NUM_CHANNELS_LAYER_9"],
                CONFIG["FILTER_SIZE_LAYER_9"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_9"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_9"],
                CONFIG["NUM_CHANNELS_LAYER_10"],
                CONFIG["FILTER_SIZE_LAYER_10"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_10"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_10"],
                CONFIG["NUM_CHANNELS_LAYER_11"],
                CONFIG["FILTER_SIZE_LAYER_11"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_11"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_11"],
                CONFIG["NUM_CHANNELS_LAYER_12"],
                CONFIG["FILTER_SIZE_LAYER_12"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_12"]
            ),
        )
        """

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
        """
        self.x_coord_readout = torch.nn.Linear(
            CONFIG["NUM_CHANNELS_LAYER_12"],
            CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            CONFIG["NUM_CHANNELS_LAYER_12"],
            CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        """


    def forward(self, x):

        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        # px = self.x_coord_readout(h2)
        # py = self.y_coord_readout(h2)
        # return torch.stack((px, py), dim=-1)
        return coords
