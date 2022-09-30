import logging

from functools import partial
from typing import Any, Callable, Optional

import torch

from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Wishart

from architectures.attentionnet import GerbilizerAttentionNet, GerbilizerAttentionHourglassNet, GerbilizerSparseAttentionNet
from architectures.densenet import GerbilizerDenseNet
from architectures.reduced import GerbilizerReducedAttentionNet
from architectures.simplenet import (
    GerbilizerSimpleNetwork,
    GerbilizerSimpleWithCovariance,
    GerbilizerSimpleIsotropicCovariance
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
outfile = '/mnt/home/achoudhri/logs/train_model_loss_debugging_1.log'
fh = logging.FileHandler(outfile)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def build_model(CONFIG: dict[str, Any]) -> tuple[torch.nn.Module, Callable]:
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
    architecture = CONFIG['ARCHITECTURE']

    basic_configs = {
        'GerbilizerDenseNet': {
            'model': GerbilizerDenseNet,
            'loss_fn': se_loss_fn
        },
        'GerbilizerSimpleNetwork': {
            'model': GerbilizerSimpleNetwork,
            'loss_fn': se_loss_fn
        },
        'GerbilizerSparseAttentionNet': {
            'model': GerbilizerSparseAttentionNet,
            'loss_fn': se_loss_fn
        },
        'GerbilizerReducedSparseAttentionNet': {
            'model': GerbilizerReducedAttentionNet,
            'loss_fn': se_loss_fn
        },
        'GerbilizerAttentionHourglassNet': {
            'model': GerbilizerAttentionHourglassNet,
            'loss_fn': wass_loss_fn
        },
        'GerbilizerSimpleWithCovariance': {
            'model': GerbilizerSimpleWithCovariance,
            'loss_fn': conditional_gaussian_loss_fn
        },
        'GerbilizerSimpleIsotropicCovariance': {
            'model': GerbilizerSimpleIsotropicCovariance,
            'loss_fn': conditional_gaussian_loss_fn
        }
    }
    # Load in the model architecture
    if architecture_config := basic_configs.get(architecture):
        model = architecture_config['model'](CONFIG)
        loss_fn = architecture_config['loss_fn']
    else:
        raise ValueError('ARCHITECTURE not recognized.')

    # make any necessary modifications
    if architecture == 'GerbilizerSimpleWithCovariance' and CONFIG.get('GAUSSIAN_LOSS_MODE') == 'map':
        scale_matrix = CONFIG.get('WISHART_SCALE')
        deg_freedom = CONFIG.get('WISHART_DEG_FREEDOM')
        loss_fn = partial(
            conditional_gaussian_loss_fn,
            mode='map',
            scale_matrix=scale_matrix,
            deg_freedom=deg_freedom
            )

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

def conditional_gaussian_loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode='mle',
    scale_matrix: Optional[torch.Tensor] = None,
    deg_freedom: Optional[torch.Tensor] = None,
    ):
    """
    Gaussian MLE/MAP loss function as described in [Russell and Reale, 2021]. Assumes
    that the true conditional distribution of a label :math: `y \in \R^2` given
    an example :math: `\bold{x} \in \mathcal{X}` can be approximated by a
    multivariate gaussian

    .. math::
        p(y | \bold{x}) = \mathcal{N}(\hat{\mu}(\bold(x)), \hat{\Sigma}(x)),

    where :math:`\hat{mu}` and :math:`\hat{\Sigma}` are models of the mean and covariance.
    
    To train the models, simply minimize the negative log of the above likelihood
    function, optionally including a Wishart prior on the covariance matrix.
    
    .. math::
        \ln |\hat{\Sigma}| + (y - \hat{\mu})^T \hat{\Sigma}^{-1} (y - \hat{\mu}) + prior

    Note: we expect pred to be a vector of length 6, where the first two entries
    represent the predicted :math: `\hat{\mu}` value and the remaining entries are
    used to determine the covariance matrix as follows:

    Place the three entries into a lower triangular matrix L. Since L is full
    rank, the matrix L @ L.T is symmetric positive definite. Interpret this resulting
    matrix as the estimated covariance :math: `\hat{\Sigma}`.

    The argument `mode` refers to whether we should return a loss function
    for maximum likelihood estimation ('mle') or maximum a posteriori estimation
    ('map'), where we use a Wishart prior.
    """
    if mode not in ('mle', 'map'):
        raise ValueError(
            f'Invalid value for `mode` passed: {mode}. Valid values are: \'mle\', \'map\'.'
            )
    

    # make sure that preds has shape: (B, 3, 2) where B is the batch size
    if pred.ndim != 3:
        raise ValueError(
            'Expected `pred` to have shape (B, 3, 2) where B is the batch size, '
            f'but encountered shape: {pred.shape}'
            )

    y_hat = pred[:, 0]  # output, shape: (B, 2)
    L = pred[:, 1:]  # cholesky factor of covariance, shape: (B, 2, 2)

    # calculate covariance matrices
    S = torch.matmul(L, L.mT)

    if mode == 'mle' and ((scale_matrix is not None) or (deg_freedom is not None)):
        logger.warning(
            'Given that loss function in mode MLE, expected `scale_matrix`'
            'and `deg_freedom` to be None. Ignoring arguments: '
            f'`scale_matrix`: {scale_matrix} and `deg_freedom`: '
            f'{deg_freedom}'
        )
    # calculate covariance matrix
 #    logger.debug(f'covariance matrices: {S}')
    # # for now, print out the eigenvalues
    # eigvals = torch.linalg.eigvalsh(S)
    # determinants = eigvals.prod(1)
    # if not (eigvals > 0).all():
    #     logger.debug('EIGENVALUES NOT ALL POSITIVE')
    #     logger.debug(f'eigenvalues: {eigvals}')
    #     logger.debug(f'determinants: {determinants}')
    # # get the log determinant
    # logger.debug(f'gaussian_mle_loss_fn | product of log eigenvalues: {torch.log(determinants)}')
    # compute the loss

    # first term: `\ln |\hat{Sigma}|`

    # create the distribution corresponding to the outputted predictive
    # density
    multivariate_normal = MultivariateNormal(
        loc=y_hat,
        scale_tril=L
    )

    loss = -multivariate_normal.log_prob(target)

    if mode == 'map':
        if scale_matrix is None or deg_freedom is None:
            raise ValueError(
                'In MAP inference mode, you must specify the '
                'parameters for the Wishart prior!'
                )
        
        device = pred.device
        wishart = Wishart(
            df=torch.tensor(deg_freedom, device=device),
            covariance_matrix=torch.tensor(scale_matrix, device=device)
        )

        prior_term = -wishart.log_prob(S)

        logger.debug(f'prior term: {prior_term}')
        loss = loss + prior_term

#     logger.debug(f'LOSS: {loss}')

    return loss
