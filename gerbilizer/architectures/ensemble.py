import torch
from torch import nn

# from torch.nn import functional as F
# from gerbilizer.training.configs import build_config

# inference only for now
class GerbilizerEnsemble(nn.Module):
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
