"""Loss functions for the DNNs."""

import logging

from typing import Literal, Optional

import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Wishart

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def gaussian_NLL_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: Literal['mle', 'map'] = 'mle',
    scale_matrix: Optional[torch.Tensor] = None,
    deg_freedom: Optional[torch.Tensor] = None,
    ):
    """
    Negative log likelihood of the Gaussian, optionally including an additive
    prior term parameterized by `scale_matrix` and `deg_freedom`.

    Assumes that the conditional distribution of a location y
    given an example x can be approximated by a multivariate gaussian

        p(y | x) = N(mu, Sigma)

    where mu(x) and Sigma(x) are models of the mean and covariance as a
    function of the audio.
    
    To train these models, we minimize the negative log of the above likelihood
    function, optionally including a Wishart prior on the covariance matrix.
    
    Note: we expect pred to be a tensor of shape (B, 3, 2), where pred[i][0]
    represents the predicted mean for example i and the entries pred[i][1:]
    and the remaining entries represent a lower triangular matrix L such that

        L @ L.T = Sigma(x)

    The argument `mode` refers to whether we should return a loss function
    for maximum likelihood estimation ('mle') or maximum a posteriori estimation
    ('map'), where we use a Wishart prior.
    """
    MAP = 'map'
    MLE = 'mle'

    if mode not in (MLE, MAP):
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

    # if in MLE mode but prior parameters provided, log a warning
    if mode == MLE and ((scale_matrix is not None) or (deg_freedom is not None)):
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

    # add prior penalty if specified
    if mode == MAP:
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
