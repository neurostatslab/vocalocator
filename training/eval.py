"""Evaluate a DNN on the provided dataset, optionally testing its calibration."""

import argparse
import json
from os import path
from pathlib import Path

import h5py
import numpy as np
import torch

from calibrationtools.accumulator import (
        CalibrationAccumulator,
        InvalidSmoothingSpecError
        )
from torch.utils.data import DataLoader

from configs import build_config_from_file, build_config_from_name
from train import Trainer
from dataloaders import GerbilVocalizationDataset

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datafile",
        type=str,
        help="Path to vocalization data to evaluate.",
    )

    parser.add_argument(
        '--outdir',
        type=str,
        required=False,
        help='Directory to which output should be saved.'
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Used to specify model configuration via a hard-coded named "
            "configuration or json file."
            ),
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        help=(
            "Used to indicate location of model weights, if not already "
            "provided by a saved config file."
            )
    )

    parser.add_argument(
        '--calibration_config',
        type=str,
        required=False,
        help=(
            'Path to JSON file containing the `smoothing_specs_for_outputs` argument'
            'to the `CalibrationAccumulator` class.'
        )
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if not path.exists(args.datafile):
        raise ValueError(f"Error: could not find data at path {args.datafile}")

    if not args.outdir:
        # if no outdir was passed, put output in the same directory
        # as the data file by default. this agrees with the previous
        # implementation
        args.outdir = Path(args.datafile).parent

    elif not path.exists(args.outdir):
        raise ValueError(f"Error: could not find output directory at path {args.outdir}")

    valid_ext = ('.h5', '.hdf', '.hdf5')
    if not any(args.datafile.endswith(ext) for ext in valid_ext):
        raise ValueError("Datafile must be an HDF5 file (.h5, .hdf, .hdf5)")

    if args.config is None:
        raise ValueError("No config or model name provided.")

    if not path.exists(args.config):
        # Assuming a model name was provided rather than a path to a json file
        args.config_data = build_config_from_name(args.config, args.job_id)
    else:
        # Assuming a json file was provided
        args.config_data = build_config_from_file(args.config, args.job_id)


def run():
    args = get_args()
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    dest_path = Path(args.datafile).stem + '_preds.h5'
    dest_path = str(Path(args.outdir) / dest_path)

    logging.debug(f'destination path: {dest_path}')

    with h5py.File(dest_path, 'w') as dest:
        with h5py.File(args.datafile, 'r') as source:
            if 'len_idx' in source:
                n_vox = len(source['len_idx']) - 1
            else:
                n_vox = source['vocalizations'].shape[0]

            source.copy(source['locations'], dest['/'], 'locations')

        # Close the h5 here to reopen it in the Dataset obj
        arena_dims = (
            args.config_data['ARENA_WIDTH'],
            args.config_data['ARENA_LENGTH']
            )
        make_xcorr = args.config_data['COMPUTE_XCORRS']
        test_set = GerbilVocalizationDataset(
            args.datafile,
            segment_len=args.config_data['SAMPLE_LEN'],
            arena_dims=arena_dims,
            make_xcorrs=make_xcorr
            )
        test_set.samp_size = 30  # take 30 samples from each vocalization. Pass them to the model as if each were a full batch of inputs
        #test_set_loader = DataLoader(test_set, args.config_data['TEST_BATCH_SIZE'], shuffle=False)
        test_set_loader = DataLoader(test_set, 1, shuffle=False)  # Only using Dataloader here for the convenience of converting np.ndarray to torch.Tensor

        preds = dest.create_dataset(
            'predictions',
            shape=(n_vox, 2),
            dtype=np.float32
        )
        vars = dest.create_dataset(
            'std',
            shape=(n_vox,),
            dtype=np.float32
        )
        errs = dest.create_dataset(
            'errs',
            shape=(n_vox,),
            dtype=np.float32
        )

        model = Trainer.from_trained_model(
            args.config_data,
            job_id=args.job_id,
            device_override=device
        )

        model.model.eval()

        # convert arena dims to cm (same unit as the output)
        MM_TO_CM = 0.1
        arena_dims_cm = np.array(arena_dims) * MM_TO_CM

        calculate_calibration = args.calibration_config is not None

        try:
            ca = CalibrationAccumulator.from_JSON(
                args.calibration_config,
                arena_dims_cm
            )
        except InvalidSmoothingSpecError as e:
            logging.warning(
                'Setting up calibration accumulator failed with the'
                f'following error: {e}'
                )
            calculate_calibration = False

        with torch.no_grad():
            for idx, (audio, location) in enumerate(test_set_loader):
                audio = audio.squeeze()
                if device == 'gpu':
                    audio = audio.cuda()
                # output = model.model(audio)
                output: torch.Tensor = model.model(audio)

                def _unscale_helper(arbitrary_scaled: torch.Tensor):
                    """
                    Helper function that calls GerbilVocalizationDataset.unscale_features
                    with the correct arguments.
                    """
                    unscaled = GerbilVocalizationDataset.unscale_features(
                        arbitrary_scaled.cpu().numpy(), arena_dims=arena_dims
                        )
                    centered = unscaled + (arena_dims_cm / 2)
                    return centered

                cm_location = _unscale_helper(location)
                # if the model is outputting a location and a covariance matrix,
                # the output shape will be (BATCH, 3, 2)
                if output.shape[1:] == (3, 2):
                    # where v[..., 0] is the x coord and v[..., 1] is the y coord
                    cm_predicted_location = _unscale_helper(output[:, 0])
                    cm_predicted_cov = _unscale_helper(output[:, 1:])
                    
                    cm_output = np.concatenate(
                        (cm_predicted_location[:, None], cm_predicted_cov),
                        axis=1
                        )

                elif output.shape[1:] == 2:
                    # output = output.mean(dim=0, keepdims=True)  # Output should have shape (30, 2)
                    cm_output = _unscale_helper(output)
                    cm_predicted_location = cm_output

                else:
                    raise ValueError("Expected model output to be either of shape (BATCH, 3, 2)"
                                     f"or (BATCH, 2). Encountered: {output.shape}"
                                     )

                # occasionally log progress
                if idx % 10 == 0:
                    logging.info(f'Reached vocalization {idx}.')
                    if idx % 100 == 0:
                        logging.debug(
                            f'Vox {idx} -- cm_predicted_location: {cm_predicted_location} '
                            f'| cm_location: {cm_location} '
                            )

                # if calibration config was passed,
                # calculate one calibration step using those arguments
                if calculate_calibration:
                    save_path = None
                    # occasionally visualize the pmfs
                    if idx % 500 == 0:
                        save_path = Path(args.outdir) / 'pmfs' / f'vox_{idx}'
                        save_path.mkdir(parents=True, exist_ok=True)

                    output_name = ca.output_names[0]
                    ca.calculate_step(
                        {output_name: cm_output},
                        cm_location,
                        pmf_save_path=save_path,
                    )

                centroid = cm_predicted_location.mean(axis=0)
                # calculate distance from each estimate to the centroid
                distances = np.linalg.norm(centroid[None] - cm_predicted_location, axis=-1)
                dist_spread = distances.std()
                preds[idx] = centroid
                vars[idx] = dist_spread

                # calculate error
                errs[idx] = np.linalg.norm(centroid - cm_location)

        if args.calibration_config:
            # calculate the calibration curves + errors
            ca.calculate_curves_and_error(h5_file=dest)
            # plot the figures
            fig_path = Path(dest_path).parent
            ca.plot_results(fig_path)


if __name__ == '__main__':
    run()
