from typing import Literal, overload

import torch

from gerbilizer.outputs import ModelOutput, ModelOutputFactory


class GerbilizerArchitecture(torch.nn.Module):
    defaults: dict

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(GerbilizerArchitecture, self).__init__()

        self.config = CONFIG
        self.output_factory = output_factory
        self.n_outputs = self.output_factory.n_outputs_expected

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    # add overload for nice unbatched functionality
    @overload
    def forward(self, x: torch.Tensor, unbatched: Literal[False]) -> ModelOutput:
        ...

    @overload
    def forward(self, x: torch.Tensor, unbatched: Literal[True]) -> list[ModelOutput]:
        ...

    def forward(self, x: torch.Tensor, unbatched: bool = False):
        """
        Run the model on input `x` and return an appropriate
        ModelOutput object, determined based on `self.output_factory`.
        """
        raw_outputs = self._forward(x)
        if unbatched:
            outputs = []
            for o in raw_outputs:
                # add batch dim back
                outputs.append(self.output_factory.create_output(o[None]))
            return outputs
        else:
            return self.output_factory.create_output(self._forward(x))
