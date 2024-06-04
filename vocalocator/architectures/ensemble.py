from typing import Literal, overload

import torch
from torch import nn
from vocalocator.architectures.base import VocalocatorArchitecture
from vocalocator.outputs import ModelOutput, ModelOutputFactory, ProbabilisticOutput


# inference only for now
class VocalocatorEnsemble(VocalocatorArchitecture):
    """
    Wrapper class to help assess ensembles of models.

    The proper way to use this class currently involves training N
    models and getting their final configs with weights paths in each
    config file. Then assemble these configs into one ensemble
    json file with key "MODELS" and value a list of json objects,
    where each object is the config for one of the models in the ensemble.
    Once this is complete, the ensemble config can be passed to the assess.py
    script without issue.
    """

    def __init__(
        self,
        config,
        built_models: list[VocalocatorArchitecture],
        output_factory: ModelOutputFactory,
    ):
        """
        Inputs the loaded config dictionary object, as well as each submodel
        built using `build_model`.
        """
        super().__init__(config, output_factory)

        # check that each model has the correct output type
        for model in built_models:
            output_type = model.output_factory.output_type
            if not issubclass(output_type, ProbabilisticOutput):
                raise ValueError(
                    "Ensembling not available for models outputting non-probabilistic outputs! "
                    f"Encountered class: {output_type}, which is not a subclass of "
                    "`ProbabilisticOutput`."
                )

        self.models = nn.ModuleList(built_models)

    def load_weights(self, best_weights_path=None, use_final_weights: bool = False):
        """
        Load the weights for each constituent model of the ensemble.
        """
        # best_weights_path included for consistency of signature
        # with `VocalocatorArchitecture.load_weights`
        for model in self.models:
            model.load_weights(use_final_weights=use_final_weights)

    # add overload for nice unbatched functionality
    @overload
    def forward(self, x: torch.Tensor, unbatched: Literal[False]) -> ModelOutput: ...

    @overload
    def forward(
        self, x: torch.Tensor, unbatched: Literal[True]
    ) -> list[ModelOutput]: ...

    def forward(self, x: torch.Tensor, unbatched: bool = False):
        """
        Run the model on input `x` and return an appropriate
        ModelOutput object, determined based on `self.output_factory`.
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x, unbatched=unbatched))
        if unbatched:
            # outputs will be a list where each item
            # corresponds to a submodel,
            # and the item is a list of model outputs, one for each
            # input in the batch.
            # to return as expected, we need to transpose
            # this list of lists and run `create_output` on
            # each item of the result.
            return [self.output_factory.create_output(list(l)) for l in zip(*outputs)]
        else:
            return self.output_factory.create_output(outputs)
