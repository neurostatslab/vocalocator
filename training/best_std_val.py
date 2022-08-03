import argparse
from os import path
from pathlib import Path

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
MIN_SIGMA = 0.8
MAX_SIGMA = 7
# number of sigma values at which to calculate
# calibration, up to MAX_SIGMA.
NUM_STEPS = 50
# number of bins used to calculate each calibration curve
NUM_CALIBRATION_BINS = 20
# ========================================================


DATAFILE = Path('/mnt/home/atanelus/ceph/iteration/finetune_split/val_set.h5')
OUTDIR = Path.home() / 'ceph' / 'poster'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
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

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
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

    model_name = args.config_data['CONFIG_NAME']

    dest_path = DATAFILE.stem + '_preds.h5'
    dest_path = str(OUTDIR / model_name / dest_path)

    logging.debug(f'destination path: {dest_path}')

    with h5py.File(dest_path, 'w') as dest:
        with h5py.File(DATAFILE, 'r') as source:
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
        val_set = GerbilVocalizationDataset(DATAFILE, segment_len=args.config_data['SAMPLE_LEN'], arena_dims=arena_dims, make_xcorrs=make_xcorr)
        val_set.samp_size = 30  # take 30 samples from each vocalization. Pass them to the model as if each were a full batch of inputs
        #test_set_loader = DataLoader(test_set, args.config_data['TEST_BATCH_SIZE'], shuffle=False)
        val_set_loader = DataLoader(val_set, 1, shuffle=False)  # Only using Dataloader here for the convenience of converting np.ndarray to torch.Tensor

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
        PMF_GRID_RESOLUTION = 0.4  # 1 cm grid resolution for pmfs
        sigma_values = np.linspace(
            args.min_std, args.max_std, args.num_std_steps
        )


        smoothing_specs = {
            model_name: {
                'gaussian_mixture': {
                    'std_values': sigma_values,
                    'desired_resolution': PMF_GRID_RESOLUTION,
                },
            }
        }

        ca = CalibrationAccumulator(
            arena_dims_cm,
            smoothing_specs
        )

        with torch.no_grad():
            for idx, (audio, location) in enumerate(val_set_loader):
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
                if idx % 200 == 0:
                    save_path = Path(OUTDIR) / 'pmfs' / f'vox_{idx}'
                    save_path.mkdir(parents=True, exist_ok=True)

                ca.calculate_step(
                    {model_name: centered_output},
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
