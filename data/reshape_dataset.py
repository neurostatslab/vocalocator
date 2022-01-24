import numpy as np
import h5py
from tqdm import trange
import math
from itertools import combinations

nt_new = 10
random_state = np.random.RandomState(123)

with h5py.File('preprocessed_vocalization_dataset.h5') as f_src:

    # All vocalizations.
    all_vocs = np.array(f_src["vocalizations"])[np.load("filtered_idx.npy")]
    n_samples = all_vocs.shape[0]
    n_mics = all_vocs.shape[1]
    nt_audio = all_vocs.shape[2]

    # # Add microphone cross-correlations to each sample.
    # vocs_xcorr = np.empty((n_samples, math.comb(n_mics, 2), nt_audio))
    # for s in trange(n_samples):
    #     for k, (i, j) in enumerate(combinations(range(n_mics), 2)):
    #         vocs_xcorr[s, k] = np.correlate(all_vocs[s, i], all_vocs[s, j], mode="same")

    # # Concatenate cross-correlations
    # all_vocs = np.concatenate((all_vocs, vocs_xcorr), axis=1)

    # Create random train, val, test set (80-10-10 split)
    idx = random_state.permutation(n_samples)
    i0 = int(n_samples * .8)
    i1 = int(n_samples * .9)
    train_idx = idx[:i0]
    val_idx = idx[i0:i1]
    test_idx = idx[i1:]

    # Take average location as the target.
    all_locs = np.mean(
        np.array(f_src["locations"])[np.load("filtered_idx.npy")],
        axis=1
    )
    # nt_old = loc_src.shape[1]
    # n_keypoints = loc_src.shape[2]
    # n_spatial_dims = loc_src.shape[3]
    # all_locs = np.zeros(
    #     (n_samples, nt_new, n_keypoints, n_spatial_dims)
    # ).astype("float32")
    # for i in trange(n_samples):
    #     for j in range(n_keypoints):
    #         for k in range(n_spatial_dims):
    #             all_locs[i, :, j, k] = np.interp(
    #                 np.linspace(0, 1, nt_new),
    #                 np.linspace(0, 1, nt_old),
    #                 loc_src[i, :, j, k],
    #             )

    # Save train set.
    with h5py.File('train_set.h5','w') as f_dest:
        f_dest["vocalizations"] = 1e3 * all_vocs[train_idx]
        f_dest["locations"] = 1e-3 * all_locs[train_idx]
    
    # Save validation set.
    with h5py.File('val_set.h5','w') as f_dest:
        f_dest["vocalizations"] = 1e3 * all_vocs[val_idx]
        f_dest["locations"] = 1e-3 * all_locs[val_idx]
    
    # Save test set.
    with h5py.File('test_set.h5','w') as f_dest:
        f_dest["vocalizations"] = 1e3 * all_vocs[test_idx]
        f_dest["locations"] = 1e-3 * all_locs[test_idx]
