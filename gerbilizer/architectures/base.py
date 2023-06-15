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

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Run the model on input `x` and return an appropriate
        ModelOutput object, determined based on `self.output_factory`.
        """
        return self.output_factory.create_output(self._forward(x))
