import argparse
import multiprocessing
from os import path
from pathlib import Path
from sys import stderr
import json

import h5py
import numpy as np
import torch

from matplotlib import pyplot as plt

from calibrationtools import CalibrationAccumulator
from calibrationtools.smoothing import gaussian_mixture
from calibrationtools.util import make_xy_grids
from calibrationtools.calculate import assign_to_bin_2d

from torch.utils.data import DataLoader

from configs import build_config_from_file, build_config_from_name
from train import Trainer
from dataloaders import GerbilVocalizationDataset

import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


TEST_SET_PATH = Path('/mnt/home/atanelus/ceph/iteration/finetune_split/test_set.h5')
OUTDIR = Path.home() / 'ceph' / 'poster' / 'eval'
OPTIMAL_STDS = Path.home() / 'ceph' / 'poster' / 'optimal_stds.json'

def confidence_set(pmf, alpha):
    """
    Return the region around the mean to which the model assigns `alpha` proportion
    of the probability mass.
    """
    sorted_bins = pmf.flatten().argsort()[::-1]
    sorted_masses = pmf.flatten()[sorted_bins]
    total_mass_by_bins = sorted_masses.cumsum()
    desired_bins = sorted_bins[total_mass_by_bins <= alpha]
    return desired_bins

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

    fh = logging.FileHandler(Path.home() / 'logs' / model_name / 'eval_calib.log')
    logger = logging.getLogger('eval_calib')
    logger.addHandler(fh)

    dest_path = Path(TEST_SET_PATH).stem + '_preds.h5'
    dest_path = str(Path(OUTDIR) / model_name / dest_path)

    logger.debug(f'destination path: {dest_path}')

    with h5py.File(dest_path, 'w') as dest:
        with h5py.File(TEST_SET_PATH, 'r') as source:
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
        test_set = GerbilVocalizationDataset(TEST_SET_PATH, segment_len=args.config_data['SAMPLE_LEN'], arena_dims=arena_dims, make_xcorrs=make_xcorr)
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
        PMF_GRID_RESOLUTION = 0.5  # 1 cm grid resolution for pmfs
        with open(OPTIMAL_STDS, 'r') as f:
            stds = json.load(f)
            std_to_use = np.array([stds[model_name]])

        smoothing_specs = {model_name: {}}

        ca = CalibrationAccumulator(
            arena_dims_cm,
            smoothing_specs,
            use_multiprocessing=False,
        )

        cal_sets = []
        cal_set_areas = []
        loc_bins = []
        loc_in_calib_set = np.zeros(len(test_set_loader))
        distances_to_furthest_point = np.zeros(len(test_set_loader))
        furthest_points = []
        centered_preds = np.zeros((len(test_set_loader), 2))
        centered_locs = np.zeros((len(test_set_loader), 2))
        errs = np.zeros(len(test_set_loader))

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
                    logger.info(f'Reached vocalization {idx}.')
                    if idx % 100 == 0:
                        logger.debug(
                            f'Vox {idx} -- centimeter_output: {centimeter_output} '
                            f'| centimeter_location: {centimeter_location} '
                            f'| centered_output: {centered_output}'
                            f'| centered_location: {centered_location}'
                            )
                save_path = None
                # occasionally visualize the pmfs
                if idx % 100 == 0:
                    save_path = Path(OUTDIR)  / model_name / 'pmfs' / f'vox_{idx}'
                    save_path.mkdir(parents=True, exist_ok=True)

                smoothed_output = gaussian_mixture(
                    centered_output,
                    std_to_use,
                    arena_dims_cm,
                    PMF_GRID_RESOLUTION
                )

                ca.calculate_step(
                    {model_name: smoothed_output},
                    centered_location,
                    pmf_save_path=save_path,
                )

                # calculate error stats
                centroid = centimeter_output.mean(axis=0)
                distances = np.sqrt( ((centroid[None, ...] - centimeter_output)**2).sum(axis=-1) )  # Should have shape (30,)
                dist_spread = distances.std()
                # preds[idx:idx+n_added] = centimeter_output
                preds[idx] = centroid
                vars[idx] = dist_spread

                # centered centroid
                centered_centroid = centered_output.mean(axis=0)
                centered_preds[idx] = centered_centroid
                centered_locs[idx] = centered_location
                errs[idx] = np.linalg.norm(centered_centroid - centered_location)

                # get the confidence set
                grid_edges_shape = np.array(smoothed_output[0].shape) + 1
                edge_xgrid, edge_ygrid = make_xy_grids(
                    arena_dims_cm,
                    shape=grid_edges_shape
                    )
                cal_set = confidence_set(smoothed_output, 0.95)
                loc_bin = assign_to_bin_2d(
                    centered_location.reshape(1, 2),
                    edge_xgrid,
                    edge_ygrid
                    )
                loc_bins.append(loc_bin)

                loc_in_calib_set[idx] = loc_bin in cal_set

                total_area = arena_dims_cm[0] * arena_dims_cm[1]

                cal_set_area = (len(cal_set) / smoothed_output[0].size) * total_area
                cal_set_areas.append(cal_set_area)

                # store the distance between the true location and the furthest
                # point contained in the confidence set
                center_xgrid, center_ygrid = make_xy_grids(
                    arena_dims_cm,
                    shape=smoothed_output[0].shape,
                    return_center_pts=True
                )
                coords = np.dstack((center_xgrid, center_ygrid))
                distances = np.linalg.norm(coords - centered_location, axis=-1)
                max_dist = distances.flatten()[cal_set].max()
                distances_to_furthest_point[idx] = max_dist

                idxs = np.unravel_index(cal_set, smoothed_output[0].shape)
                confid_set = np.zeros(smoothed_output[0].shape)
                confid_set[idxs] = 1
                cal_sets.append(confid_set)
                distances_in_set = np.where(confid_set == 1, distances, 0)
                furthest_points.append(distances_in_set.argmax())

        # calculate the calibration curves + errors
        ca.calculate_curves_and_error(h5_file=dest)
        # plot the figures
        fig_path = Path(dest_path).parent
        ca.plot_results(fig_path)

        r = dest.create_group('results')

        r.create_dataset(
            'centered_preds', data=centered_preds
        )
        r.create_dataset(
            'centered_locs', data=centered_locs
        )
        r.create_dataset(
            'loc_bins', data=np.array(loc_bins)
        )
        r.create_dataset(
            'loc_in_calib_set', data=loc_in_calib_set
        )
        r.create_dataset(
            'errs', data=errs
        )
        r.create_dataset(
            'cal_set_areas', data=np.array(cal_set_areas)
        )
        r.create_dataset(
            'cal_sets', data=np.array(cal_sets)
        )
        r.create_dataset(
            'dists_to_furthest_point', data=distances_to_furthest_point
        )
        r.create_dataset(
            'furthest_point_idxs', data=np.array(furthest_points)
        )

if __name__ == '__main__':
    run()
