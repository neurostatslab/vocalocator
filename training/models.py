import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from architectures.attentionnet import GerbilizerAttentionNet, GerbilizerAttentionHourglassNet, GerbilizerSparseAttentionNet
from architectures.densenet import GerbilizerDenseNet
from architectures.reduced import GerbilizerReducedAttentionNet
from architectures.simplenet import GerbilizerSimpleNetwork, GerbilizerSimpleWithCovariance


def build_model(CONFIG):
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
    elif CONFIG['ARCHITECTURE'] == "GerbilizerSparseAttentionNet":
        model = GerbilizerSparseAttentionNet(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG['ARCHITECTURE'] == "GerbilizerReducedSparseAttentionNet":
        model = GerbilizerReducedAttentionNet(CONFIG)
        loss_fn = se_loss_fn
    elif CONFIG['ARCHITECTURE'] == "GerbilizerAttentionHourglassNet":
        model = GerbilizerAttentionHourglassNet(CONFIG)
        loss_fn = wass_loss_fn
    elif CONFIG['ARCHITECTURE'] == "GerbilizerSimpleWithCovariance":
        model = GerbilizerSimpleWithCovariance(CONFIG)
        loss_fn = gaussian_mle_loss_fn
    else:
        raise ValueError("ARCHITECTURE not recognized.")

    if CONFIG['DEVICE'] == 'GPU':
        model.cuda()

    return model, loss_fn


# MARK: Loss functions
# All loss functions are expected to accept the prediction as the first argument
# and the ground truth as the second argument
def se_loss_fn(pred, target):
    # Assumes the inputs have shape (B, 2), representing a batch of `B` 2-dimensional coordinates
    return 2 * torch.mean(torch.square(target - pred))  # Should be a scalar tensor

def map_se_loss_fn(pred, target):
    """ MSE loss over a location map.
    """
    target = torch.flatten(target, start_dim=1)
    pred = torch.flatten(pred, start_dim=1)
    return torch.mean(torch.square(target - pred).sum(dim=1))

def wass_loss_fn(pred, target):
    """ Calculates the earth mover's distance between the target and predicted locations.
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

def gaussian_mle_loss_fn(pred, target):
    """
    Gaussian MLE loss function as described in [Russell and Reale, 2021]. Assumes
    that the true conditional distribution of a label :math: `y \in \R^2` given
    an example :math: `\bold{x} \in \mathcal{X}` can be approximated by a
    multivariate gaussian

    .. math::
        p(y | \bold{x}) = \mathcal{N}(\hat{\mu}(\bold(x)), \hat{\Sigma}(x)),

    where :math:`\hat{mu}` and :math:`\hat{\Sigma}` are models of the mean and covariance.
    
    To train the models, simply minimize the negative log of the above likelihood
    function
    
    .. math::
        \ln |\hat{\Sigma}| + (y - \hat{\mu})^T \hat{\Sigma}^{-1} (y - \hat{\mu})

    Note: we expect pred to be a vector of length 5, where the first two entries
    represent the predicted :math: `\hat{\mu}` value and the remaining entries are
    used to determine the covariance matrix as follows:

    Place the three entries into a lower triangular matrix L. Since L is full
    rank, the matrix L @ L.T is symmetric positive definite. Interpret this resulting
    matrix as the estimated covariance :math: `\hat{\Sigma}`.
    """
    # assume that preds has shape: (B, 6) where B is the batch size
    y_hat = pred[:, :2]  # shape: (B, 2)
    # construct the lower triangular matrix
    reshaped = pred[:, 2:].reshape((-1, 2, 2))
    L = reshaped.tril()
    # and now the covariance
    S = torch.matmul(L, L.transpose(1, 2))
    # compute the loss
    # first term: `\ln |\hat{Sigma}|`
    _, logdet = torch.linalg.slogdet(S)
    # second term: `(y - \hat{y})^T \hat{\Sigma}^{-1} (y - \hat{y})`
    diff = (y_hat - target).unsqueeze(-1)  # shape (B, 2, 1)
    # use torch.linalg.solve(S, diff) instead of
    # torch.matmul(torch.inverse(S), diff), since it's faster and more
    # stable according to the docs for torch.linalg.inv
    right_hand_term = torch.linalg.solve(S, diff)
    quadratic_form = torch.matmul(diff.transpose(1, 2), right_hand_term) # (B, 1, 1) by default
    quadratic_form = quadratic_form.squeeze() # (B,)
    return quadratic_form + logdet
