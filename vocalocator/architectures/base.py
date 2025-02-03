from pathlib import Path
from typing import Literal, overload

import torch

from vocalocator.outputs import ModelOutput, ModelOutputFactory


class VocalocatorArchitecture(torch.nn.Module):
    defaults: dict

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(VocalocatorArchitecture, self).__init__()

        self.config: dict = CONFIG
        self.output_factory: ModelOutputFactory = output_factory
        self.n_outputs: int = self.output_factory.n_outputs_expected
        self.is_finetuning: bool = False

    def load_weights(self, best_weights_path=None, use_final_weights: bool = False):
        """
        Load in the weights of the model, where use of `best_weights.pt` or
        `final_weights.pt` is specified by `use_final_weights`.
        """
        if best_weights_path is None:
            # try checking config for a weights path
            print("No weights path provided, checking model config.")
            try:
                best_weights_path = self.config["WEIGHTS_PATH"]
            except KeyError:
                raise ValueError(
                    "Couldn't load model weights! No `best_weights_path` kwarg provided "
                    'to `load_weights`, and could not find "WEIGHTS_PATH" in model config.'
                )

        weights_path = best_weights_path
        if use_final_weights:
            filename = Path(weights_path).name
            if "best" not in filename:
                raise ValueError(
                    "Since best weights path isn't of the form /.../best_weights.pt, "
                    "cannot locate a corresponding final weights file!"
                )
            else:
                filename = filename.replace("best", "final")
                weights_path = Path(weights_path).parent / filename

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(device)

        print(f"Loading weights from path {weights_path}.")
        weights = torch.load(weights_path, map_location=device, weights_only=True)
        # First attempt a strict load. If the saved weights are the result of finetuning, this should fail.
        try:
            self.load_state_dict(weights, strict=True)
        except RuntimeError:
            print("Attempting to load model as finetuned...")
            try:
                self._make_finetuneable()
                self.is_finetuning = True
            except NotImplementedError as e:
                print(f"Error, mismatch between architecture and saved weights: {e}")
                raise
        print("Weights loaded successfully.")

        if not self.is_finetuning:
            print("Attempting to make model finetunable...")
            try:
                self._make_finetuneable()
                self.to(device)
            except NotImplementedError as e:
                print(f"{__class__}: Warning, model is not finetuneable: {e}")

    def _make_finetuneable(self):
        """Makes the model finetunable by wrapping the necessary child modules in LORA
        objects. Should be called after loading weights.
        """
        raise NotImplementedError(f"{__class__}: This model is not fine-tuneable!")

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

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
        raw_outputs = self._forward(x)
        if unbatched:
            outputs = []
            for o in raw_outputs:
                # add batch dim back
                outputs.append(self.output_factory.create_output(o[None]))
            return outputs
        else:
            return self.output_factory.create_output(self._forward(x))
