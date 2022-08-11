import logging

from math import comb

import torch
from torch import nn
from torch.nn import functional as F


logging.basicConfig(level=logging.DEBUG)


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
        self.n_channels = CONFIG['CONV_NUM_CHANNELS']  # Converting this to a JSON array in the config for convenience
        self.n_channels.insert(0, N)
        filter_sizes = CONFIG['CONV_FILTER_SIZES']  # Also making this an array, along with the others
        dilations = CONFIG['CONV_DILATIONS']
        use_batch_norm = CONFIG['USE_BATCH_NORM']
        convolutions = [
            GerbilizerSimpleLayer(
                in_channels, out_channels, filter_size, downsample=downsample, dilation=dilation, use_bn=use_batch_norm
            )
            for in_channels, out_channels, filter_size, downsample, dilation
            in zip(self.n_channels[:-1], self.n_channels[1:], filter_sizes, should_downsample, dilations)
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        """self.final_pooling = torch.nn.Conv1d(
            n_channels[-1],
            n_channels[-1],
            kernel_size=ceiling_division(T, 32),
            groups=n_channels[-1],
            padding=0
        )"""
        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        # Final linear layer to reduce the number of channels.
        self.coord_readout = torch.nn.Linear(
            self.n_channels[-1],
            2
        )

    def forward(self, x):

        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        return coords


class GerbilizerSimpleWithCovariance(GerbilizerSimpleNetwork):
    def __init__(self, config):
        super().__init__(config)

        # replace the final coordinate readout with a block that outputs
        # 5 numbers. Two are the coordinates, and then the
        # remaining four determine the covariance matrix.
    
        # to guarantee that the resulting matrix will be symmetric positive
        # definite, we reshape those extra four numbers into a 2x2 matrix,
        # then use the torch.tril() function to convert that to a lower
        # triangular matrix L. Since L is real and full rank, we have that
        # L @ L.T is symmetric positive definite and is thus a
        # valid covariance matrix.

        del self.coord_readout

        self.last_layer = torch.nn.Linear(
            self.n_channels[-1],
            5
        )

    def forward(self, x):
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        output = self.last_layer(h2)
        # extract the location estimate
        logging.debug(f'output shape: {output.shape}')
        y_hat = output[:, :2]  # (batch, 2)
        # calculate the lower triangular Cholesky factor
        # of the covariance matrix
        len_batch = output.shape[0]
        # initialize an array of zeros into which we'll put
        # the model output
        L = torch.zeros(len_batch, 2, 2)
        # embed the elements into the matrix
        idxs = torch.tril_indices(2, 2)
        L[:, idxs[0], idxs[1]] = output[:, 2:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        y_hat = y_hat.reshape((-1, 1, 2))  # (batch, 1, 2)
        # concat the two to make a (batch, 3, 2) tensor
        concatenated = torch.cat((y_hat, L), dim=-2)
        return concatenated

    def _clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

