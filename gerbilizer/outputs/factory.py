from typing import Union

import torch

from gerbilizer.outputs.base import (
    ModelOutput,
    MDNOutput,
    Unit
    )

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
        **kwargs
        ):
        self.arena_dims = arena_dims
        self.arena_dim_units = arena_dim_units
        self.output_type = output_type
        self.type_specific_kwargs = kwargs

        # expected_param_count_known = hasattr(self.output_type, 'N_OUTPUTS_EXPECTED')
        if hasattr(self.output_type, 'N_OUTPUTS_EXPECTED'):
            self.n_outputs_expected = self.output_type.N_OUTPUTS_EXPECTED
        elif self.output_type == MDNOutput:
            # if supposed to create a mixture density network,
            # we need to parse the kwargs to understand how many
            # outputs are expected
            constituents = kwargs.get('constituent_response_types')
            if not constituents:
                raise ValueError(
                    "Cannot create MDNOutput without "
                    "keyword argument `constituent_response_types`, which "
                    "should be a list of the base distributions the mixture "
                    "density network should be interpreted as parameterizing."
                    )
            # sum the number of parameters each constituent response distribution expects
            self.n_outputs_expected = sum(
                t.N_OUTPUTS_EXPECTED for t in constituents
                )
        else:
            raise ValueError(
                f'ModelOutputFactory doesn\'t support creating objects of type {self.output_type}. '
                f'Maybe {self.output_type} is only meant to be used as an interface and not instantiated?'
                )

    def create_output(self, x: torch.Tensor) -> ModelOutput:
        """
        Package a batch of raw model output `x` into an appropriate
        `ModelOutput` instance.
        """
        return self.output_type(
            raw_output=x,
            arena_dims=self.arena_dims,
            arena_dim_units=self.arena_dim_units,
            **self.type_specific_kwargs
            )
