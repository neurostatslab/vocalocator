"""Initialize model and loss function from configuration."""

from collections import namedtuple
from typing import Any, Callable, Union

import numpy as np
import torch

from ..architectures.attentionnet import GerbilizerSparseAttentionNet
from ..architectures.densenet import GerbilizerDenseNet
from ..architectures.perceiver import (
    GerbilizerPerceiver,
    GerbilizerCovPerceiver,
)
from ..architectures.reduced import (
    GerbilizerReducedAttentionNet,
    GerbilizerAttentionHourglassNet,
)
from ..architectures.simplenet import (
    GerbilizerSimpleNetwork,
    GerbilizerSimpleWithCovariance,
    # GerbilizerSimpleIsotropicCovariance
    )
from .losses import (
    se_loss_fn,
    map_se_loss_fn,
    # wass_loss_fn,
    gaussian_NLL_loss
    )

ModelType = namedtuple("ModelType", ("model", "loss_fn", "cov_loss_fn"))


LOOKUP_TABLE = {
    "GerbilizerDenseNet": ModelType(GerbilizerDenseNet, se_loss_fn, gaussian_NLL_loss),
    "GerbilizerSimpleNetwork": ModelType(GerbilizerSimpleNetwork, se_loss_fn, gaussian_NLL_loss),
    "GerbilizerSparseAttentionNet": ModelType(
        GerbilizerSparseAttentionNet, se_loss_fn, gaussian_NLL_loss
   ),
    "GerbilizerReducedSparseAttentionNet": ModelType(
        GerbilizerReducedAttentionNet, se_loss_fn, gaussian_NLL_loss
    ),
    "GerbilizerAttentionHourglassNet": ModelType(
        GerbilizerAttentionHourglassNet, map_se_loss_fn, None
    ),
    'GerbilizerSimpleWithCovariance': ModelType( 
        GerbilizerSimpleWithCovariance, gaussian_NLL_loss, None
    ),
    "GerbilizerPerceiver": ModelType(GerbilizerPerceiver, se_loss_fn, None),
    "GerbilizerCovPerceiver": ModelType(GerbilizerCovPerceiver, gaussian_NLL_loss, None)
}


def build_model(config: dict[str, Any]) -> tuple[torch.nn.Module, Callable]:
    """
    Specifies model and loss funciton.

    Parameters
    ----------
    config : dict
        Dictionary holding training hyperparameters.

    Returns
    -------
    model : torch.nn.Module
        Model instance with hyperparameters specified
        in config.

    loss_function : function
        Loss function mapping network output to a
        per-instance. That is, loss_function should
        take a torch.Tensor with shape (batch_size, ...)
        and map it to a shape of (batch_size,) holding
        losses.
    """
    arch = config["ARCHITECTURE"]
    if arch in LOOKUP_TABLE:
        model, loss_fn, cov_loss_fn = LOOKUP_TABLE[arch]
        model = model(config)
    else:
        raise ValueError(f'ARCHITECTURE {arch} not recognized.')

    if config["DEVICE"] == "GPU" and torch.cuda.is_available():
        model.cuda()

    loss = loss_fn

    # change the loss function depending on whether a model outputting covariance
    # was chosen
    if config.get('OUTPUT_COV'):
        # if the cov_loss_fn is None, the selected model architecture doesn't have an
        # ability to output a cov matrix. for example, a model that outputs a 2d map
        if cov_loss_fn is None:
            raise ValueError(
                f"Flag `OUTPUT_COV` was passed to a model architecture that can't output a covariance matrix ({arch})."
                )
        else:
            loss = cov_loss_fn

    return model, loss


def __apply_affine(locations: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Helper function to apply an affine transformation Ax + b
    to a list of vectors.

    Args:
        locations: Array of shape (n_samples, k)
        A: Array of shape (k, k)
        b: Array of shape (k,)
    """
    return (A @ locations[..., None]).squeeze() + b

def unscale_output(model_output: np.ndarray, arena_dims: Union[tuple[float, float], np.ndarray]) -> np.ndarray:
    """
    Transform model output from arbitrary units on the square [-1, 1]^2 to
    the size of the arena, accounting correctly for the shape of the input.

    Note: expects the first dimension of model_output to be the number of
    samples per vocalization.

    This transformation and recentering is given by the affine transform
    
        z_i = Ay_i + b,

    where z_i is the new location and

        b = 1/2 * (a_1, a_2),

        A = 1/2 * [[a_1 ,  0  ],
                   [0   ,  a_2]].
    """
    A = 0.5 * np.diag(arena_dims)  # rescaling matrix
    b = 0.5 * np.array(arena_dims)  # recentering vector

    unscaled = np.zeros(model_output.shape)

    # if the model outputs a mean y and a covariance matrix S, the transformed
    # mean is given by Ay + b, and the cov matrix is given by A @ S @ A.T
    # (result from the transformation of a random variable)
    if model_output.ndim == 3 and model_output.shape[1:] == (3, 2):
        means = model_output[:, 0]  # shape: (len(model_output), 2)
        cholesky = model_output[:, 1:]  # shape: (len(model_output), 2, 2)
        covs = cholesky @ cholesky.swapaxes(-1, -2)

        unscaled[:, 0] = __apply_affine(means, A, b)
        unscaled[:, 1:] = A @ covs @ A.T
    # similar if model outputs a batch of means + cholesky covariances
    # this is the case for ensemble models
    elif model_output.ndim == 4 and model_output.shape[2:] == (3,2):
        means = model_output[:, :, 0]  # shape: (len(model_output), 2)
        cholesky = model_output[:, :, 1:]  # shape: (len(model_output), 2, 2)
        covs = cholesky @ cholesky.swapaxes(-1, -2)

        unscaled[:, :, 0] = __apply_affine(means, A, b)
        unscaled[:, :, 1:] = A @ covs @ A.T
    # otherwise, just apply the affine transformation
    elif model_output.ndim == 2 and model_output.shape[1] == 2:
        unscaled = __apply_affine(model_output, A, b)
    else:
        raise ValueError(f'Unscaling not currently supported for output of shape {model_output.shape}!')

    return unscaled
