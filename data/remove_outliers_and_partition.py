import numpy as np
import h5py
from tqdm import trange
import math
from itertools import combinations

random_state = np.random.RandomState(123)

with h5py.File("preprocessed_vocalization_dataset_with_xcorrs.h5") as f_src:

    # All vocalizations.
    vocs = np.array(f_src["vocalizations"]).astype("float32")

    # Normalize each channel by the standard deviation.
    vocs /= np.std(vocs, axis=(0, 2), keepdims=True)

    # Take average location as the target.
    locs = np.mean(f_src["locations"], axis=1)

    # Compute pairwise distances between SLEAP points.
    from scipy.spatial.distance import pdist

    pds = np.array([pdist(x, metric="euclidean") for x in locs])

    # Assume 5% of the points are mislabeled outliers.
    from sklearn.covariance import EllipticEnvelope

    cov = EllipticEnvelope(contamination=0.05).fit(pds)

    # Detect outliers and remove them
    good_idx = np.argwhere(cov.predict(pds) > 0).ravel()
    random_state.shuffle(good_idx)

    # Create random train, val, test set (80-10-10 split)
    i0 = int(len(good_idx) * 0.8)
    i1 = int(len(good_idx) * 0.9)
    train_idx = good_idx[:i0]
    val_idx = good_idx[i0:i1]
    test_idx = good_idx[i1:]

    # Save train set.
    with h5py.File("train_set.h5", "w") as f_dest:
        f_dest["vocalizations"] = vocs[train_idx]
        f_dest["locations"] = 1e-3 * locs[train_idx]

    # Save validation set.
    with h5py.File("val_set.h5", "w") as f_dest:
        f_dest["vocalizations"] = vocs[val_idx]
        f_dest["locations"] = 1e-3 * locs[val_idx]

    # Save test set.
    with h5py.File("test_set.h5", "w") as f_dest:
        f_dest["vocalizations"] = vocs[test_idx]
        f_dest["locations"] = 1e-3 * locs[test_idx]
