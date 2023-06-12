import enum
import logging

from typing import List, Literal, Optional

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
    N_OUTPUTS_EXPECTED: int

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
        # jacobian determinant of the affine transform from [-1, 1]^2 \to
        # [0,xdim] x [0, ydim] is xdim * ydim / 4, so we divide the pdf by
        # that.
        scale_factor = self.arena_dims[units].prod() / 4
        return pdf_on_arbitrary_grid / scale_factor

class Mixture(ProbabilisticOutput):
    """
    Class providing ability to create/specify/learn mixture densities.
    """
    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: str | Unit,
        constituent_response_types: List[type[ProbabilisticOutput]],
        combination_mode: Literal['AVERAGE', 'MDN'],
        ):
        """Initialize a Mixture object."""
        super().__init__(raw_output, arena_dims, arena_dim_units)


class UniformOutput(ProbabilisticOutput):
    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.tensor(0.25))

