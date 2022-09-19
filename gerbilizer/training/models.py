from typing import NewType

import torch
from torch.nn import functional as F

from ..architectures.attentionnet import GerbilizerSparseAttentionNet
from ..architectures.densenet import GerbilizerDenseNet
from ..architectures.reduced import (
    GerbilizerReducedAttentionNet,
    GerbilizerAttentionHourglassNet,
)
from ..architectures.simplenet import GerbilizerSimpleNetwork


JSON = NewType("JSON", dict)


def build_model(CONFIG: JSON):
    """
    Specifies model and loss funciton.

    Parameters
    ----------
    CONFIG : dict
        Dictionary holding training hyperparameters.

    Returns
    -------
    model : torch.nn.Module
        Model instance with hyperparameters specified
        in CONFIG.

    loss_function : function
        Loss function mapping network output to a
        per-instance. That is, loss_function should
        take a torch.Tensor with shape (batch_size, ...)
        and map it to a shape of (batch_size,) holding
        losses.
    """

    if CONFIG["ARCHITECTURE"] == "GerbilizerDenseNet":
        model = GerbilizerDenseNet(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG["ARCHITECTURE"] == "GerbilizerSimpleNetwork":
        model = GerbilizerSimpleNetwork(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG["ARCHITECTURE"] == "GerbilizerSparseAttentionNet":
        model = GerbilizerSparseAttentionNet(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG["ARCHITECTURE"] == "GerbilizerReducedSparseAttentionNet":
        model = GerbilizerReducedAttentionNet(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG["ARCHITECTURE"] == "GerbilizerAttentionHourglassNet":
        model = GerbilizerAttentionHourglassNet(CONFIG)
        loss_fn = map_se_loss_fn
    else:
        raise ValueError("ARCHITECTURE not recognized.")

    if CONFIG["DEVICE"] == "GPU":
        model.cuda()

    return model, loss_fn


# MARK: Loss functions
# All loss functions are expected to accept the prediction as the first argument
# and the ground truth as the second argument
def se_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    # Assumes the inputs have shape (B, 2), representing a batch of `B` 2-dimensional coordinates
    return 2 * torch.mean(torch.square(target - pred))  # Should be a scalar tensor


def map_se_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    """MSE loss over a location map."""
    target = torch.flatten(target, start_dim=1)
    pred = torch.flatten(pred, start_dim=1)
    return torch.mean(torch.square(target - pred).mean(dim=1))


def wass_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    """Calculates the earth mover's distance between the target and predicted locations.
    This is done by element-wise multiplying the prediction map by a (pre-computed) matrix
    of distances, in which each element contains the distance of that bin from the target
    location.
    This implementation is not symmetric because it scales the input 'pred' via a softmax.
    What is returned is not the true distance, but a numerically stable analogue that shares
    minima with the true distance w.r.t model parameters.
    """
    flat_pred = torch.flatten(pred, start_dim=1)
    flat_target = torch.flatten(target, start_dim=1)
    # Add 1 to target to make the smallest value 0
    error_map = F.log_softmax(flat_pred, dim=1) + torch.log(flat_target + 1)
    return torch.logsumexp(error_map, dim=1)
