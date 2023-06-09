import enum
import logging
import typing

import torch

from torch.nn import functional as F


class Unit(enum.Enum):
    """
    Enum type for the different units understood by the program.
    """
    ARBITRARY = enum.auto()
    CM = enum.auto()
    MM = enum.auto()

class ModelOutput:
    """
    Base class packaging the raw Tensor output from a deep neural network
    with information about its unit and potentially the way it parameterizes
    a (batch of) response distributions, if probabilistic.
    """

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: str | Unit,
        ):
        """Initialize a ModelOutput object."""
        self.data = raw_output
        self.batch_size = raw_output.shape[0]
        self.device = raw_output.device

        if isinstance(arena_dim_units, str):
            try:
                arena_dim_units = Unit[arena_dim_units]
            except KeyError:
                raise ValueError(
                    'Arena dimensions must be provided in either centimeters or millimeters! '
                    f'Instead encountered unit: {arena_dim_units}.'
                    )
        self.arena_dims = {
            Unit.MM: self._convert(arena_dims, arena_dim_units, Unit.MM),
            Unit.CM: self._convert(arena_dims, arena_dim_units, Unit.CM),
        }

    def _convert(self, x: torch.Tensor, in_units: Unit, out_units: Unit) -> torch.Tensor:
        """
        Convert an input array-like representing a (collection of) point(s)
        from one unit to another.
        """
        if in_units == out_units:
            return x
        elif in_units == Unit.MM and out_units == Unit.CM:
                return x / 10
        elif in_units == Unit.CM and out_units == Unit.MM:
                return x * 10
        elif in_units == Unit.ARBITRARY:
            return (x + 1) * 0.5 * self.arena_dims[out_units]
        elif out_units == Unit.ARBITRARY:
            return 2 * (x / self.arena_dims[in_units]) - 1
        else:
            raise ValueError(
                'Expected both `in_units` and `out_units` to be instances of '
                f'the Unit class. Instead encountered in_units: {in_units} and '
                f'out_units: {out_units}.'
                )

    def point_estimate(self, units: Unit = Unit.ARBITRARY):
        """Return a single point estimate in the specified units."""
        raise NotImplementedError


class ProbabilisticOutput(ModelOutput):
    """
    Base class packaging the raw Tensor output from a deep neural network that
    parameterizes a (batch of) distributions with its unit and choice of
    parameterization.
    """
    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the distribution p parameterized by the model
        output, where x is an array-like with expected shape (..., 2). Importantly,
        x is expected to be provided in arbitrary units (i.e. on the square [-1, 1]^2).
        """
        raise NotImplementedError

    def pmf(self, coordinate_grid: torch.Tensor, units: Unit) -> torch.Tensor:
        """
        Calculate p(x) at each point on the coordinate grid for the
        distribution p parameterized by this model output instance, in a
        vectorized and numerically stable way.
        """
        # convert coordinate grid to arbitrary units
        coordinate_grid = self._convert(coordinate_grid, units, Unit.ARBITRARY)
        pdf_on_arbitrary_grid = torch.exp(self._log_p(coordinate_grid))
        # scale pdf accordingly
        # determinant of the affine transform from [-1, 1]^2 \to [0,xdim] x [0, ydim]
        # is xdim * ydim / 4, so we divide the pdf by that.
        scale_factor = self.arena_dims[units].prod() / 4
        return pdf_on_arbitrary_grid / scale_factor

class UniformOutput(ModelOutput):
    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.tensor(0.25))

class GaussianParameterization(enum.Enum):
    """
    Enum class storing different ways to parameterize a Gaussian response and
    the number of raw model outputs expected in each case.
    """
    SPHERICAL_FIXED_VARIANCE = 2
    SPHERICAL = 3
    DIAGONAL = 4
    CHOLESKY = 5


class GaussianOutput(ProbabilisticOutput):
    """
    Base class unifying various ways to parameterize a (batch of) Gaussian
    response distribution(s) and packaging model output together with its
    units.
    """

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Unit,
        parameterization: str | GaussianParameterization,
        variance: typing.Optional[torch.Tensor] = None,
        units: typing.Optional[Unit] = None,
        ):
        """
        Construct a GaussianOutput object given a raw Tensor outputted by a DNN,
        with shape dependent on the chosen parameterization.

        Note: if the "SPHERICAL_FIXED_VARIANCE" parameterization is chosen,
        this constructor requires the additional keyword arguments `variance`
        and `units` which determine the chosen fixed variance and in which
        units that variance is expressed.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        if isinstance(parameterization, GaussianParameterization):
            self.parameterization = parameterization
        else:
            try:
                self.parameterization = GaussianParameterization[parameterization]
            except KeyError:
                raise ValueError(
                    f'Invalid paramterization string {parameterization}! '
                    f'Expected one of {[param.name for param in GaussianParameterization]}.'
                    )
                                 
        if raw_output.shape[-1] != self.parameterization.value:
            raise ValueError(
                f'Given Gaussian parameterization {parameterization}, expected '
                f'output to have dimension {self.parameterization.value} '
                f'along final axis, but instead encountered dimension '
                f'{raw_output.shape[-1]}.'
                )

        match parameterization:
            case GaussianParameterization.SPHERICAL_FIXED_VARIANCE:
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

            case GaussianParameterization.SPHERICAL:
                # for now, just do a softplus so the diagonal entries of the
                # Cholesky (lower triangular) factor of the covariance matrix
                # are positive
                diagonal_entries = F.softplus(self.data[:, 2] * torch.ones(2))
                self.cholesky_covs = torch.diag_embed(diagonal_entries)

            case GaussianParameterization.DIAGONAL:
                diagonal_entries = F.softplus(self.data[:, 2:])
                self.cholesky_covs = torch.diag_embed(diagonal_entries)

            case GaussianParameterization.CHOLESKY:
                L = torch.zeros(self.batch_size, 2, 2, device=self.device)
                # embed the elements into the matrix
                idxs = torch.tril_indices(2, 2)
                L[:, idxs[0], idxs[1]] = self.data[:, 2:]
                # apply softplus to the diagonal entries to guarantee the resulting
                # matrix is positive definite
                new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
                L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
                # reshape y_hat so we can concatenate it to L
                self.cholesky_covs = L

    def point_estimate(self, units: Unit = Unit.ARBITRARY):
        """
        Return the mean of the Gaussian(s) in the specified units.
        """
        # first two values of model output are always interpreted as the mean
        mean = torch.clamp(self.data[:, :2], -1, 1)
        return self._convert(mean, Unit.ARBITRARY, units)

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the Gaussian parameterized by the model
        output, where x is an array-like with expected shape (..., 2). Importantly,
        x is expected to be provided in arbitrary units (i.e. on the square [-1, 1]^2).
        """
        distr = torch.distributions.MultivariateNormal(
            loc=self.point_estimate(),
            scale_tril=self.cholesky_covs
            )
        return distr.log_prob(x)
