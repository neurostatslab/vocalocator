from typing import Union

import torch
from vocalocator.outputs.base import EnsembleOutput, MDNOutput, ModelOutput, Unit


class ModelOutputFactory:
    """
    Util class passed to model instances that provides the number of expected
    parameters and a creator method taking a tensor and returning a ModelOutput
    object.
    """

    def __init__(
        self,
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
        output_type: type[ModelOutput],
        **kwargs,
    ):
        self.arena_dims = arena_dims
        self.arena_dim_units = arena_dim_units
        self.output_type = output_type
        self.type_specific_kwargs = kwargs

        # expected_param_count_known = hasattr(self.output_type, 'N_OUTPUTS_EXPECTED')
        if hasattr(self.output_type, "N_OUTPUTS_EXPECTED"):
            self.n_outputs_expected = self.output_type.N_OUTPUTS_EXPECTED
        elif self.output_type == MDNOutput:
            # if supposed to create a mixture density network,
            # we need to parse the kwargs to understand how many
            # outputs are expected
            constituents = self.type_specific_kwargs.get("constituent_response_types")
            if not constituents:
                raise ValueError(
                    "Cannot create MDNOutput without "
                    "keyword argument `constituent_response_types`, which "
                    "should be a list of the base distributions the mixture "
                    "density network should be interpreted as parameterizing."
                )
            # sum the number of parameters each constituent response distribution expects
            n_params_for_constituents = sum(t.N_OUTPUTS_EXPECTED for t in constituents)
            # and expect one value per constituent distribution representing a
            # log of the mixing weights
            n_mixing_params = len(constituents)
            self.n_outputs_expected = n_params_for_constituents + n_mixing_params
        elif self.output_type == EnsembleOutput:
            # ensemble is a bit of a weird exception
            # probably worth cleaning up and readjusting at some point
            # for now, though, it can take a variable number of arguments
            # since it expects an output of type List[ProbabilisticOutput]
            self.n_outputs_expected = None
        else:
            raise ValueError(
                f"ModelOutputFactory doesn't support creating objects of type {self.output_type}. "
                f"Maybe {self.output_type} is only meant to be used as an interface and not instantiated?"
            )

    def create_output(self, x) -> ModelOutput:
        """
        Package a batch of raw model output `x` into an appropriate
        `ModelOutput` instance.
        """
        return self.output_type(
            raw_output=x,
            arena_dims=self.arena_dims,
            arena_dim_units=self.arena_dim_units,
            **self.type_specific_kwargs,
        )
