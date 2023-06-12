import enum
import logging

from typing import List, Literal, Optional, Union

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
        arena_dim_units: Union[str, Unit],
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


# class Ensemble(ProbabilisticOutput):
#     """
#     Util class allowing an arbitrary number of models to be ensembled together
#     into one mixture density.
#
#     Crucially this class has a DIFFERENT init pattern: instead of inputting a
#     torch tensor `raw_output`, this class requires a LIST of
#     ProbabilisticOutputs `model_outputs`. The reason for this change is so that
#     an arbitrary number of model output classes each accepting a potentially
#     instance-specific number of parameters can still be combined without
#     needing to know the details ahead of time.
#     """
#     def __init__(
#         self,
#         model_outputs: List[ProbabilisticOutput],
#         arena_dims: torch.Tensor,
#         arena_dim_units: str | Unit,
#         ):
#         """Initialize a Mixture object."""
#         # create a fake raw_output tensor so we can still
#         # use the base class's constructor to process arena dims and units
#         device = model_outputs[0].data.device
#         fake_raw_output = torch.tensor(0, device=device)
#         super().__init__(fake_raw_output,, arena_dims, arena_dim_units)


class MDNOutput(ProbabilisticOutput):
    """
    Class storing the output of a mixture density network, which
    assumes a mixture response and parameterizes both the constituent
    distributions of the mixture as well as the weights by which the constituent
    distributions should be combined.
    """
    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
        constituent_response_types: List[type[ProbabilisticOutput]],
        ):
        """
        ADD A BETTER DOCSTRING HERE LATER.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # sum the number of parameters each constituent response distribution expects
        total_parameters_for_constituents = sum(
            t.N_OUTPUTS_EXPECTED for t in constituent_response_types
            )
        # then add on the number of mixing weights expected (R for R
        # the number of response distributions)
        num_responses = len(constituent_response_types)
        n_parameters_expected = total_parameters_for_constituents + num_responses

        if raw_output.shape[-1] != n_parameters_expected:
            raise ValueError(
                f"Given constituent response types {constituent_response_types}, expected "
                f"to recieve {n_parameters_expected} parameters per stimulus, with "
                f"{total_parameters_for_constituents} params to specify the constituent "
                f"distributions, and {num_responses} params to specify the mixing weights. "
                f"Instead encountered {raw_output.shape[-1]}, with raw_output shape "
                f"of {raw_output.shape}."
                )

        curr_idx = 0
        self.responses = []
        for response_type in constituent_response_types:
            n_params = response_type.N_OUTPUTS_EXPECTED
            next_idx = curr_idx + n_params
            response = response_type(raw_output[:, curr_idx:next_idx], arena_dims, arena_dim_units)
            self.responses.append(response)
            curr_idx = next_idx

        # the parameters representing the weights are expected to be
        # passes as unnormalized logits. call these w_ij for i the batch index
        # and j the logit for response distribution j.
        unnormalized_logits = raw_output[:, curr_idx:]
        # the actual weights theta_i are given by
        # theta_ij = exp(w_ij) / [ \sum_k exp(w_ik) ]
        # but we're ultimately going to work on log scale, so
        # instead compute log theta)ij = w_ij - log(sum_k exp(w_ik))
        normalizer = torch.logsumexp(unnormalized_logits, -1, keepdim=True)
        self.log_weights = unnormalized_logits - normalizer

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        # with the log weights log(theta_ij) defined in the init
        # method, we can now compute the mixture density evaluated at a
        # certain (collection of) point(s)
        if x.shape[-2] != self.batch_size:
            raise ValueError(
                f'Incorrect shape for input! Since batch size is {self.batch_size}, '
                f'expected second-to-last dim of input `x` to have the same shape. Instead '
                f'found shape {x.shape}.'
                )
        # stacks R tensors each of shape (..., self.batch_size)
        # into one big tensor of shape (..., self.batch_size, R)
        # this is for compatibility with self.log_weights, which has
        # shape (..., self.batch_size)
        individual_log_probs = torch.stack([r._log_p(x) for r in self.responses], -1)
        # the result is given by p_i = \sum_j theta_ij p_ij
        # for p_i the mixture density for batch element i,
        # and p_ij the probability assigned from response j
        # calculating this using only the log probs and log weights,
        # the formula becomes
        # log p_i = log \sum_j exp( log theta_ij + log p_ij)

        # tensor shape wise, this operation reduces along the last dimension
        # across the different responses as expected
        #     (..., self.batch_size, R) --> (..., self.batch_size)
        return torch.logsumexp(self.log_weights + individual_log_probs, -1)

class UniformOutput(ProbabilisticOutput):

    N_OUTPUTS_EXPECTED = 0

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.batch_size:
            raise ValueError(
                f'Incorrect shape for input! Since batch size is {self.batch_size}, '
                f'expected second-to-last dim of input `x` to have the same shape. Instead '
                f'found shape {x.shape}.'
                )
        output_shape = x.shape[:-1]
        return torch.ones(output_shape) * torch.log(torch.tensor(0.25))

