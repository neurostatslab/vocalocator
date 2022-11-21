import copy

SIMPLENET_BASE = {
    "MOMENTUM": 0.9,
    "TRAIN_BATCH_SIZE": 64,
    "VAL_BATCH_SIZE": 64,
    "WEIGHT_DECAY": 1e-05,
    "USE_BATCH_NORM": True,
    "MAX_LEARNING_RATE": 1e-03,
    "MIN_LEARNING_RATE": 1e-05,
    "SAMPLE_LEN": 8192,
    "SHOULD_DOWNSAMPLE": [False, True, True, True, True, True, False],
    "CONV_FILTER_SIZES": [
        19,
        7,
        39,
        41,
        23,
        29,
        33
    ],

    "NUM_MICROPHONES": 4,
    "NUM_EPOCHS": 20,
    "ARCHITECTURE": "GerbilizerSimpleNetwork",
    "COMPUTE_XCORRS": True,
    "AUGMENT_DATA": False,
    "CLIP_GRADIENTS": True,
    "CONV_NUM_CHANNELS": [
        16,
        16,
        16,
        32,
        32,
        32,
        64,
        64,
        64
    ],
    "CONV_DILATIONS": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
    ],

    "ARENA_WIDTH": 600,
    "ARENA_LENGTH": 400,

    "CONFIG_NAME": "simplenet_no_cov",
    "DEVICE": "CPU",
    "TORCH_SEED": 888,
    "NUMPY_SEED": 777,
    "LOG_INTERVAL": 3,
    "NUM_AUDIO_SAMPLES": 10000,
    "AUDIO_SAMPLE_RATE": 125000,
    "SAVE_SAMPLE_OUTPUT": False,
    "SAVE_LOSS_PLOT": True,
    "OPTIMIZER": "SGD",
    "TEST_BATCH_SIZE": 64,
    "POOLING": "AVG",
    "AUGMENT_STRETCH_MIN": 0.95,
    "AUGMENT_STRETCH_MAX": 1.1,
    "AUGMENT_STRETCH_PROB": 1e-06,
    "AUGMENT_GAUSS_NOISE": 0.005,
    "AUGMENT_PITCH_MIN": -1.0,
    "AUGMENT_PITCH_MAX": 1.0,
    "AUGMENT_PITCH_PROB": 1e-06,
    "AUGMENT_SHIFT_MIN": -0.1,
    "AUGMENT_SHIFT_MAX": 0.1,
    "AUGMENT_SHIFT_PROB": 1.0,
    "AUGMENT_INVERSION_PROB": 0.5,
    "AUGMENT_LABELS": False,
    "AUGMENT_FLIP_HORIZ": True,
    "AUGMENT_FLIP_VERT": True,
    "DATAFILE_PATH": "/mnt/home/atanelus/ceph/iteration/small_room_4/",
}

SIMPLENET_COV = copy.deepcopy(SIMPLENET_BASE)
SIMPLENET_COV['OUTPUT_COV'] = True

ENSEMBLE = {
    'MODELS': [SIMPLENET_COV, SIMPLENET_COV, SIMPLENET_COV]
}

ENSEMBLE_AVG = {
    'MODELS': [SIMPLENET_COV, SIMPLENET_COV, SIMPLENET_COV],
    'AVERAGE_OUTPUTS': True
}

ENSEMBLE_MISSING_COV = {
    'MODELS': [SIMPLENET_COV, SIMPLENET_BASE, SIMPLENET_COV],
}
