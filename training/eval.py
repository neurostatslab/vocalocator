import argparse
import multiprocessing
from os import path
from pathlib import Path
from sys import stderr

import h5py
import numpy as np
import torch

from calibrationtools import CalibrationAccumulator
from torch.utils.data import DataLoader

from configs import build_config_from_file, build_config_from_name
from train import Trainer
from dataloaders import GerbilVocalizationDataset

import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# ================ calibration constants ==================
# gaussian mixture smoothing:
# min and max variance used in the gaussian smoothing
# of point estimates, in cm
MIN_SIGMA = 0.1
MAX_SIGMA = 20
# number of sigma values at which to calculate
# calibration, up to MAX_SIGMA.
NUM_STEPS = 50
# dynamic spherical gaussian smoothing:
# minimium and maximum proportion of the observed
# mean distance between each point estimate and the
# centroid, used as variance of a gaussian placed
# at the centroid
MIN_FRAC = 0.01
MAX_FRAC = 10
# number of bins used to calculate each calibration curve
NUM_CALIBRATION_BINS = 20
# ========================================================


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
        '--min_std',
        type=float,
        required=False,
        default=MIN_SIGMA,
        help=(
            'Min smoothing std, in cm, to use in calibration curve calculation. '
            'Calibration curves will be calculated at `num_std_steps` values from '
            f'[min_std, max_std]. Default: {MIN_SIGMA}.'
        )
    )

    parser.add_argument(
        '--max_std',
        type=float,
        required=False,
        default=MAX_SIGMA,
        help=(
            'Max smoothing std, in cm, to use in calibration curve calculation. '
            'Calibration curves will be calculated at `num_std_steps` values from '
            f'[min_std, max_std]. Default: {MAX_SIGMA}.'
        )
    )

    parser.add_argument(
        '--num_std_steps',
        type=int,
        required=False,
        default=NUM_STEPS,
        help=(
            'Number of steps in range at which to calculate the calibration curve '
            'and error for the `gaussian_mixture` smoothing method. '
            f'Default: {NUM_STEPS}'
            )
    )

    parser.add_argument(
        '--min_frac',
        type=float,
        required=False,
        default=MIN_FRAC,
        help=(
            'Minimum fraction to use in calibration curve calculation for the '
            '`dynamic_spherical_gaussian` smoothing method. This smoothing '
            'method calculates the mean distance between each point estimate '
            'and their centroid, then uses various fractions of that value '
            'as the variance of a spherical Gaussian placed at the centroid.'
        )
    )

    parser.add_argument(
        '--max_frac',
        type=float,
        required=False,
        default=MAX_FRAC,
        help=(
            'Minimum fraction to use in calibration curve calculation for the '
            '`dynamic_spherical_gaussian` smoothing method. This smoothing '
            'method calculates the mean distance between each point estimate '
            'and their centroid, then uses various fractions of that value '
            'as the variance of a spherical Gaussian placed at the centroid.'
        )
    )

    parser.add_argument(
        '--num_frac_steps',
        type=int,
        required=False,
        default=NUM_STEPS,
        help=(
            'Number of steps in range at which to calculate the calibration '
            'curve and error for the `dynamic_spherical_gaussian` smoothing '
            f'method. Default: {NUM_STEPS}'
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
        test_set = GerbilVocalizationDataset(args.datafile, segment_len=args.config_data['SAMPLE_LEN'], arena_dims=arena_dims, make_xcorrs=make_xcorr)
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

        model = Trainer.from_trained_model(
            args.config_data,
            job_id=args.job_id,
            device_override=device
        )

        model.model.eval()

        # convert arena dims to cm (same unit as the output)
        MM_TO_CM = 0.1
        arena_dims_cm = np.array(arena_dims) * MM_TO_CM
        # set up calibration accumulator
        PMF_GRID_RESOLUTION = 1  # 1 cm grid resolution for pmfs
        sigma_values = np.linspace(
            args.min_std, args.max_std, args.num_std_steps
        )
        frac_values = np.linspace(
            args.min_frac, args.max_frac, args.num_frac_steps
        )
        smoothing_specs = {
            'model_output': {
                'gaussian_mixture': {
                    'std_values': sigma_values,
                    'desired_resolution': PMF_GRID_RESOLUTION,
                },
                'dynamic_spherical_gaussian': {
                    'fracs_of_est_variance': frac_values,
                    'desired_resolution': PMF_GRID_RESOLUTION,
                }
            }
        }

        ca = CalibrationAccumulator(
            arena_dims_cm,
            smoothing_specs
        )

        with torch.no_grad():
            for idx, (audio, location) in enumerate(test_set_loader):
                audio = audio.squeeze()
                if device == 'gpu':
                    audio = audio.cuda()
                # output = model.model(audio)
                output = model.model(audio)
                # output = output.mean(dim=0, keepdims=True)  # Output should have shape (30, 2)
                centimeter_output = GerbilVocalizationDataset.unscale_features(
                    output.cpu().numpy(), arena_dims=arena_dims
                    )
                centimeter_location = GerbilVocalizationDataset.unscale_features(
                    location.cpu().numpy(), arena_dims=arena_dims
                    )

                # move origin from center of room to bottom left corner
                centered_output = centimeter_output + (arena_dims_cm / 2)
                centered_location = centimeter_location + (arena_dims_cm / 2)

                # occasionally log progress
                if idx % 10 == 0:
                    logging.info(f'Reached vocalization {idx}.')
                    if idx % 100 == 0:
                        logging.debug(
                            f'Vox {idx} -- centimeter_output: {centimeter_output} '
                            f'| centimeter_location: {centimeter_location} '
                            f'| centered_output: {centered_output}'
                            f'| centered_location: {centered_location}'
                            )
                save_path = None
                # occasionally visualize the pmfs
                if idx % 500 == 0:
                    save_path = Path(args.outdir) / 'pmfs' / f'vox_{idx}'
                    save_path.mkdir(parents=True, exist_ok=True)

                ca.calculate_step(
                    {'model_output': centered_output},
                    centered_location,
                    pmf_save_path=save_path,
                )

                centroid = centimeter_output.mean(axis=0)
                distances = np.sqrt( ((centroid[None, ...] - centimeter_output)**2).sum(axis=-1) )  # Should have shape (30,)
                dist_spread = distances.std()
                # preds[idx:idx+n_added] = centimeter_output
                preds[idx] = centroid
                vars[idx] = dist_spread

        # calculate the calibration curves + errors
        ca.calculate_curves_and_error(h5_file=dest)
        # plot the figures
        fig_path = Path(dest_path).parent
        ca.plot_results(fig_path)


if __name__ == '__main__':
    run()
