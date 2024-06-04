"""Initialize model and loss function from configuration."""

import json
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch

from vocalocator.outputs import (
    GaussianOutputFixedVariance,
    ModelOutput,
    ModelOutputFactory,
    ProbabilisticOutput,
)
from vocalocator.outputs.base import BaseDistributionOutput, MDNOutput

from ..architectures.base import VocalocatorArchitecture
from ..architectures.ensemble import VocalocatorEnsemble
from .losses import negative_log_likelihood, squared_error


def __subclasses_recursive(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in __subclasses_recursive(c)]
    )


def subclasses(cls):
    return {
        c.config_name: c
        for c in list(__subclasses_recursive(cls))
        if hasattr(c, "config_name")
    }


ARCHITECTURES = {
    model.__name__.lower(): model for model in VocalocatorArchitecture.__subclasses__()
}

OUTPUT_TYPES = subclasses(ModelOutput)


def make_output_factory(config: dict[str, Any]) -> ModelOutputFactory:
    """
    Parse the provided config and return an appropriately configured
    ModelOutputFactory instance.
    """
    model_params = config["MODEL_PARAMS"]
    provided_type = model_params.get("OUTPUT_TYPE", "").upper()
    if provided_type not in OUTPUT_TYPES:
        raise ValueError(
            f"Unrecognized output type: {provided_type}! "
            f"Allowable types are: {list(OUTPUT_TYPES.keys())}"
        )
    output_type = OUTPUT_TYPES[provided_type]

    arena_dims = torch.tensor(config["DATA"]["ARENA_DIMS"])
    arena_dims_units = config["DATA"].get("ARENA_DIMS_UNITS")

    if not arena_dims_units:
        raise ValueError(
            "Need to specify units for arena dimensions under key "
            "`ARENA_DIMS_UNITS`!"
        )

    # now handle certain subclasses which require additional information
    # to be parsed from config
    additional_kwargs = {}
    if output_type == MDNOutput:
        # check for additional parameter 'CONSTITUENT_DISTRIBUTIONS'
        if not model_params.get("CONSTITUENT_DISTRIBUTIONS"):
            raise ValueError(
                "Given mixture density network output type (MDNOutput), "
                "expected parameter `CONSTITUENT_DISTRIBUTIONS` expressing "
                "the component distributions for the MDN. This argument "
                "should be in the `MODEL_PARAMS` subdict of the config "
                "and should contain a list of valid output type strings."
            )
        provided_constituents = model_params["CONSTITUENT_DISTRIBUTIONS"]

        # only allow subclasses of BaseDistributionOutput now
        base_distr_subclasses = subclasses(BaseDistributionOutput)
        constituent_types = []
        for constituent_type in provided_constituents:
            if constituent_type not in base_distr_subclasses:
                raise ValueError(
                    f"Unrecognized output type as constituent for MDN: {constituent_type}! "
                    f"Allowable types are: {list(base_distr_subclasses.keys())}"
                )
            constituent_types.append(base_distr_subclasses[constituent_type])

        additional_kwargs["constituent_response_types"] = constituent_types
        # for more details on this config param, see the MDNOutput constructor
        additional_kwargs["constituent_extra_kwargs"] = model_params.get(
            "CONSTITUENT_EXTRA_KWARGS", {}
        )

    elif output_type == GaussianOutputFixedVariance:
        # expect 'VARIANCE' and 'VARIANCE_UNITS'
        variance = model_params.get("VARIANCE", "")
        units = model_params.get("VARIANCE_UNITS", "")

        additional_kwargs["variance"] = torch.tensor(variance)
        additional_kwargs["units"] = units

    return ModelOutputFactory(
        arena_dims=arena_dims,
        arena_dim_units=arena_dims_units,
        output_type=output_type,
        **additional_kwargs,
    )


LossFunction = Callable[[ModelOutput, torch.Tensor], torch.Tensor]


def build_model(config: dict[str, Any]) -> tuple[VocalocatorArchitecture, LossFunction]:
    """
    Specifies model and loss funciton.

    Parameters
    ----------
    config : dict
        Dictionary holding training hyperparameters.

    Returns
    -------
    model : VocalocatorArchitecture
        Model instance with hyperparameters specified
        in config.

    loss_function : LossFunction
        Loss function mapping network output to a
        per-instance. That is, loss_function should
        take a torch.Tensor with shape (batch_size, ...)
        and map it to a shape of (batch_size,) holding
        losses.
    """
    # 1. parse desired output type
    output_factory = make_output_factory(config)

    # if the output is probabilistic in nature, use NLL
    if issubclass(output_factory.output_type, ProbabilisticOutput):
        loss = negative_log_likelihood
    # otherwise, just use squared error
    else:
        loss = squared_error

    arch: str = config["ARCHITECTURE"].lower()

    if arch == "vocalocatorensemble":
        # None out the other parameters
        built_submodels = []
        for sub_model_config in config["MODEL_PARAMS"]["CONSTITUENT_MODELS"]:
            if not isinstance(sub_model_config, dict):
                # A path to a config was passed instead of the config contents
                with open(sub_model_config, "r") as f:
                    if not Path(sub_model_config).exists():
                        raise ValueError(
                            f"Path to submodel config {sub_model_config} does not exist!"
                        )
                    sub_model_config = json.load(f)
            submodel, _ = build_model(sub_model_config)
            built_submodels.append(submodel)
        model = VocalocatorEnsemble(config, built_submodels, output_factory)
    elif arch.lower() in ARCHITECTURES:
        model = ARCHITECTURES[arch]
        model = model(config, output_factory)
    else:
        raise ValueError(f"ARCHITECTURE {arch} not recognized.")

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


def unscale_output(
    model_output: np.ndarray, arena_dims: Union[tuple[float, float], np.ndarray]
) -> np.ndarray:
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
    elif model_output.ndim == 4 and model_output.shape[2:] == (3, 2):
        means = model_output[:, :, 0]  # shape: (batch_size, n_models, 2)
        cholesky = model_output[:, :, 1:]  # shape: (batch_size, n_models, 2, 2)
        covs = cholesky @ cholesky.swapaxes(-1, -2)

        unscaled[:, :, 0] = __apply_affine(means, A, b)
        unscaled[:, :, 1:] = A @ covs @ A.T

    # otherwise, just apply the affine transformation
    elif model_output.ndim == 2 and model_output.shape[1] == 2:
        unscaled = __apply_affine(model_output, A, b)
    else:
        raise ValueError(
            f"Unscaling not currently supported for output of shape {model_output.shape}!"
        )

    return unscaled
