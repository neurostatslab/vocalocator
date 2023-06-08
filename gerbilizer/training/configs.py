"""
File to configure all hyperparameters (model architecture and training).
"""
from sys import stderr
from typing import NewType

try:
    # Attempt to use json5 if available
    import pyjson5 as json
except ImportError:
    print("Warning: json5 not available, falling back to json.", file=stderr)
    import json


JSON = NewType("JSON", dict)
DEFAULT_CONFIG = {
    "OPTIMIZATION": {
        "NUM_EPOCHS": 30,
        "OPTIMIZER": "SGD",
        "MOMENTUM": 0.9,
        "MAX_LEARNING_RATE": 0.0005,
        "MIN_LEARNING_RATE": 0.0000,
        "CLIP_GRADIENTS": False,
    },
    "ARCHITECTURE": "GerbilizerSimpleNetwork",
    "GENERAL": {
        "CONFIG_NAME": "simple_network",
        "DEVICE": "GPU",  # 'GPU' or 'CPU'
        "TORCH_SEED": 888,  # rng seeds for reproducibility
        "NUMPY_SEED": 777,
        "LOG_INTERVAL": 3,  # Amount of time between consecutive log messages
        "SAVE_SAMPLE_OUTPUT": False,
        "SAVE_LOSS_PLOT": False,
    },
    "DATA": {
        "NUM_MICROPHONES": 4,
        "AUDIO_SAMPLE_RATE": 125000,
        "COMPUTE_XCORRS": False,
        "TRAIN_BATCH_SIZE": 1024,
        "TRAIN_BATCH_MAX_SAMPLES": 200_000,
        "VAL_BATCH_SIZE": 64,
        "TEST_BATCH_SIZE": 64,
        "CROP_LENGTH": 2048,
        # controls mirroring and channel permutation augmentations
        "AUGMENT_LABELS": False,
        "AUGMENT_DATA": True,  # controls noise addition
        "ARENA_DIMS": [558.9, 355.6],
    },
    "AUGMENTATIONS": {
        "AUGMENT_STRETCH_MIN": 0.95,  # currently unimpleemented
        "AUGMENT_STRETCH_MAX": 1.1,
        "AUGMENT_STRETCH_PROB": 1e-6,
        "AUGMENT_SNR_PROB": 0.5,
        "AUGMENT_SNR_MIN": 5,
        "AUGMENT_SNR_MAX": 45,
        "AUGMENT_PITCH_MIN": -1.0,  # currently unimplemented
        "AUGMENT_PITCH_MAX": 1.0,
        "AUGMENT_PITCH_PROB": 1e-6,
        "AUGMENT_SHIFT_MIN": -0.1,  # currently unimplemented
        "AUGMENT_SHIFT_MAX": 0.1,
        "AUGMENT_SHIFT_PROB": 1.0,
        "AUGMENT_INVERSION_PROB": 0.5,  # currently unimplemented
        # Label augmentations: involve mirroring sounds within the arena
        "AUGMENT_FLIP_HORIZ": True,  # contingent on AUGMENT_LABELS
        "AUGMENT_FLIP_VERT": True,  # contingent on AUGMENT_LABELS
    },
}


def keys_to_uppercase(dictionary: dict) -> dict:
    """Converts all the string keys in a dictionary to uppercase for consistency"""
    new_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict):  # recurse through subdictionaries if present
            value = keys_to_uppercase(value)
        if isinstance(key, str):
            new_dict[key.upper()] = value
        else:
            new_dict[key] = value
    return new_dict


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def build_config(filepath: str) -> JSON:
    with open(filepath, "r") as ctx:
        try:
            config = json.load(ctx)
        except:
            raise ValueError(
                f"Could not parse JSON file at {filepath}. Perhaps a JSON5 file was provided without the necessary libraries installed?"
            )

    if "CONFIG_NAME" not in config["GENERAL"]:
        raise ValueError(
            "Configurations provided as JSON files should include a 'CONFIG_NAME' string."
        )

    config = update_recursively(config, DEFAULT_CONFIG)
    return keys_to_uppercase(config)
