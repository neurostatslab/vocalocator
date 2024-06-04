import argparse
from os import path
from pathlib import Path

from vocalocator.training.configs import build_config
from vocalocator.training.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # Configs can be provided as either a name, which then references an entry in the dictionary
    # located in configs.py, or as a path to a JSON file, when then uses the entries in that file
    # to override the default configuration entries.

    parser.add_argument(
        "--config",
        type=str,
        help="Used to specify model configuration via a JSON file.",
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Path to directory containing train, test, and validation datasets or single h5 file for inference",
    )

    parser.add_argument(
        "--save-path",
        type=Path,
        required=False,
        default=path.abspath("."),
        help="Directory for trained models' weights",
    )

    parser.add_argument(
        "--indices",
        type=Path,
        required=False,
        default=None,
        help="Path to directory containing train/val indices. Should be provided as train_set.npy and val_set.npy",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if args.config is None:
        raise ValueError("No config file provided.")

    if not path.exists(args.config):
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    if args.save_path is None:
        raise ValueError("No save path (trained model storage location) provided.")

    if args.data is None:
        raise ValueError("No data path provided.")

    args.config_data = build_config(args.config)

    Path(args.save_path).mkdir(parents=True, exist_ok=True)


def run(args):
    weights = args.config_data.get("WEIGHTS_PATH", None)

    # This modifies args.config_data['WEIGHTS_PATH']
    trainer = Trainer(
        data_dir=args.data,
        model_dir=args.save_path,
        config_data=args.config_data,
        eval_mode=False,
        index_dir=args.indices,
    )

    if weights is not None:
        trainer.model.load_weights(best_weights_path=weights)

    trainer.train()


if __name__ == "__main__":
    args = get_args()
    run(args)
