import pathlib

import torch
from torch import nn

# from torch.nn import functional as F
# from gerbilizer.training.configs import build_config

# inference only for now
class GerbilizerEnsemble(nn.Module):
    # config should just tell us model architectures
    def __init__(self, config, built_models):
        """
        Inputs the loaded config dictionary object, as well as each submodel
        built using `build_model`.
        """
        super().__init__()

        for model_config, model in zip(config["MODELS"], built_models):
            if "OUTPUT_COV" not in model_config:
                raise ValueError(
                    "Ensembling not yet available for models without uncertainty estimates."
                )

            if "WEIGHTS_PATH" not in model_config:
                raise ValueError(
                    "Cannot construct ensemble! Not all models have config paramter `WEIGHTS_PATH`."
                    )

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            weights = torch.load(model_config["WEIGHTS_PATH"], map_location=device)
            model.load_state_dict(weights, strict=False)

        self.models = nn.ModuleList(built_models)

#     @classmethod
#     def from_ensemble_dir(cls, ensemble_dir):
#         """
#         Load in an ensemble of models.
#         """
#         # load in jsons
#         config_paths = sorted(pathlib.Path(ensemble_dir).glob("*/config.json"))
#         # load in model weights
#         configs = []
#         for path in config_paths:
#             configs.append(build_config(str(path)))
#         # call regular constructor
#         ensemble = cls({"MODELS": configs})
#         # load state dict for each model
#         # for model, config in zip(ensemble.models, configs):
#         #     if "WEIGHTS_PATH" not in config:
#         #         raise ValueError(
#         #             "Cannot construct ensemble! Not all models have config paramter `WEIGHTS_PATH`."
#         #         )
#         #     model.load_state_dict(torch.load(config["WEIGHTS_PATH"]))
#         return ensemble

    def forward(self, x):
        # return mean and cholesky covariance of gaussian mixture
        # created from the ensemble of models
        batch_size = x.shape[0]
        means = torch.zeros(batch_size, len(self.models), 2)
        cholesky_covs = torch.zeros(batch_size, len(self.models), 2, 2)
        for i, model in enumerate(self.models):
            output = model(x)
            means[:, i] = output[:, 0]
            cholesky_covs[:, i] = output[:, 1:]

        # add extra dim so means is of shape (batch_size, len(self.models), 1, 2)
        return torch.cat((means[:, :, None], cholesky_covs), dim=-2)
