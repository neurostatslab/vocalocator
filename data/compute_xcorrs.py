import numpy as np
import h5py
from tqdm import trange
import math
from itertools import combinations


with h5py.File('preprocessed_vocalization_dataset.h5') as f_src:

    # All vocalizations.
    all_vocs = np.array(f_src["vocalizations"])
    n_samples = all_vocs.shape[0]
    n_mics = all_vocs.shape[1]
    nt_audio = all_vocs.shape[2]

    # Add microphone cross-correlations to each sample.
    vocs_xcorr = np.empty((n_samples, math.comb(n_mics, 2), nt_audio))
    for s in trange(n_samples):
        for k, (i, j) in enumerate(combinations(range(n_mics), 2)):
            vocs_xcorr[s, k] = np.correlate(all_vocs[s, i], all_vocs[s, j], mode="same")

    # Save new datafile.
    with h5py.File('preprocessed_vocalization_dataset_with_xcorrs.h5', 'w') as f_dest:
        f_dest["vocalizations"] = np.concatenate((all_vocs, vocs_xcorr), axis=1)
        f_dest["locations"] = np.array(f_src["locations"])

