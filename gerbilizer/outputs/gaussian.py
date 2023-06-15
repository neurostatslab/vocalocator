import torch

from torch.nn import functional as F

from gerbilizer.outputs.base import BaseDistributionOutput, Unit


class GaussianOutput(BaseDistributionOutput):
    """
    Base class representing a (batch of) Gaussian response distribution(s) that
    packages model output together with its units.
    """

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        ):
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # expect subclasses to provide this instance variable
        # in their init methods
        self.cholesky_covs: torch.Tensor

        if raw_output.shape[-1] != self.N_OUTPUTS_EXPECTED:
            raise ValueError(
                f'Given Gaussian output class {self.__class__.__name__}, expected '
                f'output to have dimension {self.N_OUTPUTS_EXPECTED} '
                f'along final axis, but instead encountered dimension '
                f'{raw_output.shape[-1]}.'
                )

    def _point_estimate(self):
        """
        Return the mean of the Gaussian(s) in the specified units.
        """
        # first two values of model output are always interpreted as the mean
        return torch.clamp(self.raw_output[:, :2], -1, 1)

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output.
        
        Expects x to have shape (..., self.batch_size, 2), and to be provided
        in arbitrary units (i.e. on the square [-1, 1]^2).
        """
        # check that the second-to-last dimension is the same as
        # the batch dimension
        if x.shape[-2] != self.batch_size:
            raise ValueError(
                f'Incorrect shape for input! Since batch size is {self.batch_size}, '
                f'expected second-to-last dim of input `x` to have the same shape. Instead '
                f'found shape {x.shape}.'
                )
        distr = torch.distributions.MultivariateNormal(
            loc=self.point_estimate(),
            scale_tril=self.cholesky_covs
            )
        return distr.log_prob(x)

                             
class GaussianOutputFixedVariance(GaussianOutput):

    N_OUTPUTS_EXPECTED = 2
    config_name = 'GAUSSIAN_FIXED_VARIANCE'

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        variance: torch.Tensor,
        units: Unit,
        ):
        """
        Construct a GaussianOutput object with fixed spherical covariance
        matrix across all inputs, with diagonal entries given by `variance` in
        units `units`.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        if not variance or not units:
            raise ValueError(
                'For spherical fixed variance parameterization, '
                'arguments `variance` and `units` are required!'
                )
        # if necessary, convert to arbitrary units
        if units == Unit.ARBITRARY:
            variances = torch.ones(2) * variance
        else:
            variances = (torch.ones(2) * variance) / self.arena_dims[units]

        cholesky_cov = torch.diag_embed(torch.sqrt(variances))
        self.cholesky_covs = cholesky_cov[None].repeat(self.batch_size, 1, 1)

class GaussianOutputSphericalCov(GaussianOutput):

    N_OUTPUTS_EXPECTED = 3
    config_name = 'GAUSSIAN_SPHERICAL_COV'

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        ):
        """
        Construct a GaussianOutput object with spherical covariance matrices.

        Unlike GaussianOutputFixedVariance, this class does NOT impose the
        restriction that covariance matrices are fixed, instead they may vary
        by input.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # for now, just do a softplus so the diagonal entries of the
        # Cholesky (lower triangular) factor of the covariance matrix
        # are positive
        diagonal_entries = F.softplus(self.raw_output[:, 2])
        # (self.batch_size, 1, 1) * (2, 2) -> (self.batch_size, 2, 2)
        self.cholesky_covs = diagonal_entries[:, None, None] * torch.eye(2)

class GaussianOutputDiagonalCov(GaussianOutput):

    N_OUTPUTS_EXPECTED = 4
    config_name = 'GAUSSIAN_DIAGONAL_COV'

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        ):
        """
        Construct a GaussianOutput object with diagonal covariance matrices.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # for now, just do a softplus so the diagonal entries of the
        # Cholesky (lower triangular) factor of the covariance matrix
        # are positive
        diagonal_entries = F.softplus(self.raw_output[:, 2:]) # (self.batch_size, 2)
        self.cholesky_covs = torch.diag_embed(diagonal_entries) # (self.batch_size, 2, 2)

class GaussianOutputFullCov(GaussianOutput):

    N_OUTPUTS_EXPECTED = 5
    config_name = 'GAUSSIAN_FULL_COV'

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        ):
        """
        Construct a GaussianOutput object with diagonal covariance matrices.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        L = torch.zeros(self.batch_size, 2, 2, device=self.device)
        # embed the elements into the matrix
        idxs = torch.tril_indices(2, 2)
        L[:, idxs[0], idxs[1]] = self.raw_output[:, 2:]
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        self.cholesky_covs = L
