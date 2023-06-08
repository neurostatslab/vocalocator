import logging
import typing

import torch

Unit = typing.Literal['ARBITRARY', 'CM', 'MM']


class ModelOutput:
    """
    Base class packaging the raw Tensor output from a deep neural network
    with information about its unit and potentially the way it parameterizes
    a response distribution, if probabilistic.
    """

    def __init__(self, raw_output: torch.Tensor):
        """Initialize a ModelOutput object."""
        self.data = raw_output

    def point_estimate(self, units: Unit):
        """Return a single point estimate."""
        raise NotImplementedError


class ProbabilisticOutput(ModelOutput):
    """
    Base class packaging the raw Tensor output from a deep neural network that
    parameterizes a distribution with its unit and choice of parameterization.
    """
    def log_p(self, coordinate: torch.Tensor, units: Unit):
        """
        Return log p(coordinate) under the distribution p parameterized by the
        model output.
        """
        raise NotImplementedError

GaussianParameterization = typing.Literal[
    'SPHERICAL_FIXED_VARIANCE',
    'SPHERICAL',
    'DIAGONAL',
    'CHOLESKY',
]

class GaussianOutput(ProbabilisticOutput):
    """
    Base class unifying various ways to parameterize a Gaussian
    response distribution and packaging model output together
    with its units.
    """

    EXPECTED_SHAPES = {
        'SPHERICAL_FIXED_VARIANCE': 2,
        'SPHERICAL': 3,
        'DIAGONAL': 4,
        'CHOLESKY': 5
    }

    def __init__(
        self,
        raw_output: torch.Tensor,
        parameterization: GaussianParameterization,
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
        super().__init__(raw_output)

        self.parameterization = parameterization

        if not self.EXPECTED_SHAPES.get(parameterization):
            raise ValueError(
                f'Expected `parameterization` to be one of {list(self.EXPECTED_SHAPES.values())}. '
                f'Instead encountered {parameterization}.'
                )
        expected_shape = self.EXPECTED_SHAPES[parameterization]
        if raw_output.shape[-1] != expected_shape:
            raise ValueError(
                f'Given Gaussian parameterization {parameterization}, expected output to '
                f'have dimension {expected_shape} along final axis, but instead encountered '
                f'dimension {raw_output.shape[-1]}.'
                )

        if parameterization == 'SPHERICAL_FIXED_VARIANCE':
            if not variance or not units:
                raise ValueError('For spherical fixed variance parameterization, arguments `variance` and `units` are required!')
            if units == 'ARBITRARY':
                self.variance = variance
            # if not, convert variance to arbitrary units
