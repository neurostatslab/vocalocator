import argparse
from os import path

import h5py
import numpy as np
import torch

from configs import build_config_from_file, build_config_from_name
from train import Trainer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datafile",
        type=str,
        help="Path to vocalization data to evaluate.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Used to specify model configuration via a hard-coded named configuration or json file.",
    )

    parser.add_argument(
        "--job_id",
        type=int,
        required=False,
        help="Used to indicate location of model weights, if not already provided by a saved config file.",
    )


    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if not path.exists(args.datafile):
        raise ValueError(f"Error: could not find data at path {args.datafile}")
    
    valid_ext = ('.h5', '.hdf', '.hdf5')
    if not any(args.datafile.endswith(ext) for ext in valid_ext):
        raise ValueError(f"Datafile must be an HDF5 file (.h5, .hdf, .hdf5)")
    
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

    with h5py.File(args.datafile, 'r+') as source:
        n_vox = source['vocalizations'].shape[0]

        if 'predictions' in source:
            del source['predictions']

        preds = source.create_dataset(
            'predictions',
            shape=(n_vox, 2),
            dtype=np.float32
        )

        idx = 0

        model = Trainer.from_trained_model(
            args.config_data,
            job_id=args.job_id,
            device_override=device
        )

        for block in model.eval_on_dataset(source):
            n_added = block.shape[0]
            preds[idx:idx+n_added] = block
            idx += n_added


if __name__ == '__main__':
    run()
