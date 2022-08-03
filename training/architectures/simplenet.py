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
        # 6 numbers. Two are the coordinates, and then the
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
            6
        )

    #     METHODS = {
    #         'CHOLESKY': self._cholesky_covariance_calculation,
    #         'TANH': self._tanh_covariance_calculation,
    #     }
    #     # calculate covariance using the cholesky factorization by default
    #     self.calculate_covariance = METHODS['CHOLESKY']
    #     if method_name := config.get('COVARIANCE_CALC_METHOD'):
    #         if method_name not in METHODS:
    #             # TODO RAISE SMTH handle error somehow
    #             pass
    #         else:
    #             self.calculate_covariance = METHODS[method_name]

    # # function spec: take in a [B, 4] vector
    # # output a [B, 2, 2] psd matrix
    # def _cholesky_covariance_calculation(self, output_vec):
    #     """
    #     Build 2x2 covariance matrices from vectors of length 4,
    #     by taking 3 values to form a lower triangular matrix L. Then
    #     apply softplus to make sure L is full rank, and lastly output
    #     L @ L.T.
    #     """
    #     assert output_vec.shape[1] == 4
    #     # reshape to be a matrix
    #     reshaped = output_vec.reshape((-1, 2, 2))
    #     L = reshaped.tril()
    #     # apply a softplus to make sure none of the entries are zero
    #     # this guarantees that the matrix is full rank
    #     epsilon = 1e-3
    #     softplussed_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1)) + epsilon
    #     L = L.diagonal_scatter(softplussed_diagonals, dim1=-2, dim2=-1)
    #     # compute LL^T for each matrix in the batch
    #     S = torch.matmul(L, L.transpose(1, 2))  # (batch, 2, 2)
    #     return S
    
    # def _tanh_covariance_calculation(self, output_vec):
    #     """
    #     Build a 2x2 covariance matrix from 4 outputted values, as described in 
    #     (Russell and Reale, 2021).
    #     """
    #     # let the first two elements represent the
    #     # diagonal entries
    #     diag = S[:, :2] # (B, 2)
    #     offdiag = S[:, 3] # (B, 1)
    #     # exponentiate the diagonal entries
    #     variances = torch.exp(diag)
    #     # and place into a matrix
    #     batch_dim = len(output_vec)
    #     S = torch.zeros(batch_dim, 2, 2)
    #     # for each batch, place the two entries into the diagonal
    #     S = torch.diagonal_scatter(S, variances, dim1=-2, dim2=-1)
    #     # calculate the covariance value
    #     correlation_coefficient = torch.tanh(offdiag)
    #     # cov_ij = r * sqrt(sigma_i^2 * sigma_j*2)
    #     # shape: (B, )
    #     cov = correlation_coefficient * torch.sqrt(variances.prod(dim=-1))
    #     # repeat the entry to be shape (B, 2) so we get cov_ji
    #     cov = cov[:, None].repeat(1, 2)
    #     # place the covariance value into the upper right
    #     # and lower right corners
    #     flipped = S.fliplr()  # matrix where variances are on the off diagonals
    #     # put the covariances on the diagonals on this flipped matrix
    #     flipped = torch.diagonal_scatter(flipped, cov, dim1=-2, dim2=-1)
    #     # flip it back
    #     S = flipped.fliplr()
    #     return S

    def forward(self, x):
        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        output = self.last_layer(h2)
        # extract the location estimate
        logging.debug(f'output shape: {output.shape}')
        y_hat = output[:, :2]  # (batch, 2)
        # calculate the lower triangular Cholesky factor
        # of the covariance matrix
        reshaped = output[:, 2:].reshape((-1, 2, 2))
        L = reshaped.tril()
        # apply the softplus to the diagonal entries
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(L, new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        y_hat = y_hat.reshape((-1, 1, 2))  # (batch, 1, 2)
        # concat the two to make a (batch, 3, 2) tensor
        concatenated = torch.cat((y_hat, L), dim=-2)
        return concatenated

    def _clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

