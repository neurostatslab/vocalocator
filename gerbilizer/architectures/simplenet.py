import logging

from math import comb
from typing import NewType, Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
            #padding=(filter_size * dilation - 1) // 2,
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
    def __init__(self, CONFIG):
        super(GerbilizerSimpleNetwork, self).__init__()

        T = CONFIG["SAMPLE_LEN"]
        N = CONFIG["NUM_MICROPHONES"]

        if CONFIG["COMPUTE_XCORRS"]:
            N += comb(N, 2)

        should_downsample = CONFIG["SHOULD_DOWNSAMPLE"]
        self.n_channels = CONFIG[
            "CONV_NUM_CHANNELS"
        ]  # Converting this to a JSON array in the config for convenience
        filter_sizes = CONFIG[
            "CONV_FILTER_SIZES"
        ]  # Also making this an array, along with the others

        min_len = min(len(self.n_channels), len(filter_sizes))
        self.n_channels = self.n_channels[:min_len]
        filter_sizes = filter_sizes[:min_len]
        should_downsample = should_downsample[:min_len]

        dilations = CONFIG["CONV_DILATIONS"]
        use_batch_norm = CONFIG["USE_BATCH_NORM"]

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

        """self.final_pooling = torch.nn.Conv1d(
            n_channels[-1],
            n_channels[-1],
            kernel_size=ceiling_division(T, 32),
            groups=n_channels[-1],
            padding=0
        )"""
        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        # Final linear layer to reduce the number of channels.
        # self.coord_readout = torch.nn.Linear(self.n_channels[-1], 2)
        self.output_cov = bool(CONFIG.get('OUTPUT_COV'))
        N_OUTPUTS = 5 if self.output_cov else 2

        self.coord_readout = torch.nn.Linear(self.n_channels[-1], N_OUTPUTS)

    def forward(self, x):
        x = x.transpose(-1, -2)  # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        coords = self.coord_readout(h2)
        return build_cov_output(coords, x.device) if self.output_cov else coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)


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

    def forward(self, x: torch.Tensor):
        """
        Output parameters that define a predictive distribution p(location | audio),
        which we assume to be normally distributed.

        Specifically, output a (3, 2) torch.Tensor where the first row corresponds
        to the mean of this Gaussian, and the remaining entries define the lower
        triangular factor of the covariance matrix.
        """
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        output = self.last_layer(h2)
        return build_cov_output(output, x.device)

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)


# class GerbilizerSimpleIsotropicCovariance(GerbilizerSimpleNetwork):
#     def __init__(self, config):
#         super().__init__(config)
#
#         # replace the final coordinate readout with a block that outputs
#         # 3 numbers. Two are the coordinates, and then the
#         # remaining one, lambda, determines the isotropic covariance matrix:
#         # \Sigma = \lambda * I
#     
#         del self.coord_readout
#
#         self.last_layer = torch.nn.Linear(
#             self.n_channels[-1],
#             3
#         )
#
#     def forward(self, x: torch.Tensor):
#         """
#         Output parameters that define a predictive distribution p(location | audio),
#         which we assume to be normally distributed.
#
#         Specifically, output a (3, 2) torch.Tensor where the first row corresponds
#         to the mean of this Gaussian, and the remaining entries define the lower
#         triangular factor of the covariance matrix.
#
#         The lower triangular factor of Sigma is returned instead of the actual
#         covariance matrix for consistency with the `GerbilizerSimpleWithCovariance` class.
#         """
#         h1 = self.conv_layers(x)
#         h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
#         output = self.last_layer(h2)
#         # extract the location estimate
#         y_hat = output[:, :2]  # (batch, 2)
#         # construct the triangular factor
#         lambda_ = F.softplus(output[:, 2]) # (batch,)
#         L = torch.eye(2, device=x.device)[None] * lambda_[:, None, None]
#         # reshape y_hat so we can concatenate it to L
#         y_hat = y_hat.reshape((-1, 1, 2))  # (batch, 1, 2)
#         # concat the two to make a (batch, 3, 2) tensor
#         concatenated = torch.cat((y_hat, L), dim=-2)
#         return concatenated
#
#     def _clip_grads(self):
#         nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
#
