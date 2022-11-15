import pathlib

import torch
from torch import nn
# from torch.nn import functional as F
from gerbilizer.training.configs import build_config

from gerbilizer.training.models import build_model

# inference only for now
class GerbilizerEnsemble(nn.Module):
    # config should just tell us model architectures
    def __init__(self, config):
        super().__init__()

        models = []

        # load in the models from config
        for model_config in config['MODELS']:
            if 'OUTPUT_COV' not in model_config:
                raise ValueError('Ensembling not yet available for models without uncertainty estimates.')
            model, _ = build_model(model_config)
            models.append(model)

        self.models = nn.ModuleList(models)

    @classmethod
    def from_ensemble_dir(cls, ensemble_dir):
        """
        Load in an ensemble of models.
        """
        # load in jsons
        config_paths = sorted(pathlib.Path(ensemble_dir).glob('*/config.json'))
        # load in model weights
        configs = []
        for path in config_paths:
            configs.append(build_config(str(path)))
        # call regular constructor
        ensemble = cls({'MODELS': configs})
        # load state dict for each model
        for model, config in zip(ensemble.models, configs):
            if 'WEIGHTS_PATH' not in config:
                raise ValueError('Cannot construct ensemble! Not all models have config paramter `WEIGHTS_PATH`.')
            model.load_state_dict(torch.load(config['WEIGHTS_PATH']))
        return ensemble

    def forward(self, x):
        # return mean and cholesky covariance of gaussian mixture
        # created from the ensemble of models
        batch_size = x.shape[0]
        means = torch.zeros(len(self.models), batch_size, 2)
        cholesky_covs = torch.zeros(len(self.models), batch_size, 2, 2)
        for i, model in enumerate(self.models):
            output = model(x)
            means[i] = output[:, 0]
            cholesky_covs[i] = output[:, 1:]
        # calculate the mean and covariance matrix of the mixture density
        # mean is easy, just the average of the means:
        mixture_means = means.mean(dim=0)
        # covariance is harder. by introducing a new variable z representing the mixture
        # assignment, we can use the law of total variance to decompose the resulting
        # covariance as
        # Var(Y) = E[Var(Y | Z = z)] + Var(E[Y | Z = z])
        # the following is the Cholesky factor of E[Var(Y|Z=z)]
        avg_cholesky_cov = cholesky_covs.mean(dim=0)
        # this is Var(E[Y | Z = z])
        cov_means_over_assigment = torch.zeros(batch_size, 2, 2)
        for i in range(batch_size):
            cov_means_over_assigment[i] = torch.cov(means[:, i])
        # take the cholesky decomposition
        chol_cov_over_assignment = torch.linalg.cholesky(cov_means_over_assigment)
        # sum the two to find the full cholesky mixture covariance
        chol_mixture_cov = avg_cholesky_cov + chol_cov_over_assignment
        # concatenate mixture_means and chol_mixture_cov to make a (batch, 3, 2) tensor
        mixture_means = mixture_means.reshape((-1, 1, 2))  # (batch, 1, 2)
        return torch.cat((mixture_means, chol_mixture_cov), dim=-2)
