"""
File to configure all hyperparameters (model architecture and training).
"""
import json
import numpy as np

from typing import NewType


JSON = NewType('JSON', dict)
DEFAULT_CONFIG = {
    "DEVICE": "CPU",
    "TORCH_SEED": 888,
    "NUMPY_SEED": 777,
    "LOG_INTERVAL": 3, # seconds
    "NUM_MICROPHONES": 4,  # not including xcorrs
    "COMPUTE_XCORRS": False,
    "NUM_AUDIO_SAMPLES": 10000,
    "AUDIO_SAMPLE_RATE": 125000,
    "SAVE_SAMPLE_OUTPUT": True,  # Will save part of the validation predictions at every epoch
    "SAVE_LOSS_PLOT": True,  # Will plot loss at every epoch
    "ARENA_WIDTH": 600,  # Size of enclosure along x axis
    "ARENA_LENGTH": 400,  # Size of enclosure along y axis

    # Training hyperparameters.
    "NUM_EPOCHS": 20,
    "TRAIN_BATCH_SIZE": 64,
    "MAX_LEARNING_RATE": 1e-2,
    "MIN_LEARNING_RATE": 1e-6,
    "WEIGHT_DECAY": 1e-5,
    "CLIP_GRADIENTS": False,
    "OPTIMIZER": "SGD",
    "MOMENTUM": 0.5,

    "VAL_BATCH_SIZE": 64,
    "TEST_BATCH_SIZE": 64,

    # Architecture hyperparameters.
    "ARCHITECTURE": "GerbilizerSimpleNet",
    "USE_BATCH_NORM": False,
    "POOLING": "AVG",

    "CONV_NUM_CHANNELS": [11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    "CONV_FILTER_SIZES": [51, 51, 51, 51, 51, 51, 51, 51, 49, 37, 19, 9],
    "CONV_DILATIONS": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

    # Data Augmentations present during training.
    "AUGMENT_DATA": True,
    "AUGMENT_STRETCH_MIN": 0.95,
    "AUGMENT_STRETCH_MAX": 1.1,
    "AUGMENT_STRETCH_PROB": 1e-6,
    "AUGMENT_GAUSS_NOISE": 0.005,
    "AUGMENT_PITCH_MIN": -1.0,
    "AUGMENT_PITCH_MAX": 1.0,
    "AUGMENT_PITCH_PROB": 1e-6,
    "AUGMENT_SHIFT_MIN": -0.1,
    "AUGMENT_SHIFT_MAX": 0.1,
    "AUGMENT_SHIFT_PROB": 1.0,
    "AUGMENT_INVERSION_PROB": 0.5,
    
    # Label augmentations: involve mirroring sounds within the arena
    "AUGMENT_LABELS": True,
    "AUGMENT_FLIP_HORIZ": True,
    "AUGMENT_FLIP_VERT": True,
}


def keys_to_uppercase(dictionary: dict) -> dict:
    """ Converts all the string keys in a dictionary to uppercase
    """
    new_dict = dict()
    for key, value in dictionary.items():
        if isinstance(key, str):
            new_dict[key.upper()] = value
        else:
            new_dict[key] = value
    return new_dict


def build_config_from_name(config_name, job_id):
    """
    Returns dictionary of hyperparameters.
    """
    # Use job_id to seed any random hyperparameters.
    rs = np.random.RandomState(job_id)

    # Specify default hyperparameters

    # Specify custom CONFIG dicts below. Note that
    # any unspecified hyperparameters will default
    # to the values held in DEFAULT_CONFIG.
    if config_name == "default":
        CONFIG = dict()

    elif config_name == "alex":
        CONFIG = {
            "NUM_EPOCHS": 100,
            "ARCHITECTURE": "GerbilizerSimpleNetwork",
            "MAX_LEARNING_RATE": 1e-1,
            "MIN_LEARNING_RATE": 1e-4,
            "MOMENTUM": 0.9,
            "DEVICE": "CPU",
            "AUGMENT_DATA": True,

            "CONV_NUM_CHANNELS": [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128],
            "CONV_FILTER_SIZES": [11, 11, 11, 7,  7,  7,  7,  7,  7,  7,   7,   7],
            "CONV_DILATIONS":    [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,   2,   2],
        }
    else:
        raise ValueError(
            f"'{config_name}' was not recognized as a "
            "valid 'config_name' parameter."
            )
    
    CONFIG['JOB_ID'] = job_id
    CONFIG['CONFIG_NAME'] = config_name
    # Fill in any default values.
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in CONFIG.keys():
            CONFIG[key] = default_value

    return keys_to_uppercase(CONFIG)


def build_config_from_file(filepath: str, job_id: int) -> JSON:
    with open(filepath, 'r') as ctx:
        config = json.load(ctx)
    
    if 'CONFIG_NAME' not in config:
        raise ValueError("Configurations provided as JSON files should include a 'CONFIG_NAME' string.")

    if job_id is not None:
        config['JOB_ID'] = job_id
    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config.keys():
            config[key] = default_value

    return keys_to_uppercase(config)
