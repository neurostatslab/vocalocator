import multiprocessing as mp

import numpy as np
import scipy.stats


def assign_to_bin_2d(locations, xgrid, ygrid):
    """
    Return an array of indices of the 2d bins to which each input in
    `locations` corresponds.
    
    The indices correspond to the "flattened" version of the grid. In essence,
    for a point in bin (i, j), the output is i * n_y_pts + j, where n_y_pts
    is the number of gridpoints in the y direction.
    """
    # locations: (NUM_SAMPLES, 2)
    # xgrid: (n_y_pts, n_x_pts)
    # xgrid: (n_y_pts, n_x_pts)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    # 1d array of numbers representing x coord of each bin
    x_bins = xgrid[0]
    # same for y coord
    y_bins = ygrid[:, 0]
    # assign each coord to a bin in one dimension
    # note: subtract one to ignore leftmost bin (0)
    x_idxs = np.digitize(x_coords, x_bins) - 1
    y_idxs = np.digitize(y_coords, y_bins) - 1
    # NOTE: we expect model output to have shape (NUM_SAMPLES, n_x_pts, n_y_pts)
    # so when we flatten, the entry at coordinate (i, j) gets mapped to
    # (n_y_pts * i) + j
    n_y_pts = len(y_bins)
    return (n_y_pts * x_idxs) + y_idxs


def min_mass_containing_location(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
    ):
    # maps: (NUM_SAMPLES, n_x_pts, n_y_pts)
    # locations: (NUM_SAMPLES, 2)
    # coord_bins: (n_y_pts, n_x_pts, 2)  ( output of meshgrid then dstack ) 
    # reshape maps to (NUM_SAMPLES, N_BINS)
    num_samples = maps.shape[0]
    flattened_maps = maps.reshape((num_samples, -1))
    idx_matrix = flattened_maps.argsort(axis=1)[:, ::-1]
    # bin number for each location
    loc_idxs = assign_to_bin_2d(locations, xgrid, ygrid)
    # bin number for first interval containing location
    bin_idxs = (idx_matrix == loc_idxs[:, np.newaxis]).argmax(axis=1)
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = xgrid.shape[0] * xgrid.shape[1]
    x_idx = np.arange(num_bins)[np.newaxis, :].repeat(num_samples, axis=0)
    condition = x_idx > bin_idxs[:, np.newaxis]
    sorted_maps = np.take_along_axis(flattened_maps, idx_matrix, axis=1)
    s = np.where(condition, 0, sorted_maps).sum(axis=1)
    return s


def min_mass_containing_location_single(pmf, loc, xgrid, ygrid):
    # flatten the pmf
    flattened = pmf.flatten()
    # argsort in descending order
    argsorted = flattened.argsort()[::-1]
    # assign the true location to a coordinate bin
    # reshape loc to a (1, 2) array so the vectorized function
    # assign_to_bin_2d still works
    loc_idx = assign_to_bin_2d(loc[np.newaxis, :], xgrid, ygrid)
    # bin number for first interval containing location
    bin_idx = (argsorted == loc_idx).argmax()
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = xgrid.shape[0] * xgrid.shape[1]
    sorted_maps = flattened[argsorted]
    s = np.where(np.arange(num_bins) > bin_idx, 0, sorted_maps).sum()
    return s


def min_mass_containing_location_mp(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
):
    def arg_iter():
        for pmf, loc in zip(maps, locations):
            yield (pmf, loc, xgrid, ygrid)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        arg_iterator = arg_iter()
        masses = pool.starmap(min_mass_containing_location_single, arg_iterator)
    
    return np.array(masses)
        

def calibration_step(
    model_output,
    true_location,
    arena_dims,
    std_values,
    n_calibration_bins: int = 10,
    grid_resolution: float = 0.1,
    ) -> np.ndarray:
    """
    Perform one step of the calibration process on `model_output`,
    vectorized over multiple different smoothing parameters `sigma_values`.

    Args:
        model_output: Array of location estimates (in centimeters) from a model,
            for one audio sample. Expected shape: (n_predictions, 2).
        true_location: Array contianing the true location of the audio sample, in
            centimeters. Expected shape: (1, 2).
        arena_dims: Config parameter storing the dimensions of the arena, in
            millimeters.
        sigma_values: Array of variances, in cm, used to smooth the predictions.
        n_calibration_bins: integer representing how fine the calibration curve should
            be. Default: 10.
        grid_resolution: Desired resolution (in centimeters) to use when creating
            the discrete probability distributions representing the model output.

    Returns:
        An array `arr` of shape (len(sigma_values),), where each entry arr[i]
        represents the calibration mass output for the given sample when predictions
        were smoothed with variance sigma_values[i].

    Essentially, we place a spherical Gaussian with variance $sigma^2$ at
    each location estimate in `model_output`, then sum and average over all location 
    estimates for the given sample to create a new probability mass function.
    
    Then, we calculate $m$, the probability assigned to the smallest region in the xy 
    plane containing the true location, where these regions are defined by
    progressively taking the location bins to which the model assigns the highest probability mass.

    We repeat this process for each value of sigma in `sigma_vals`, and return
    the resulting array.
    """
    # check to make sure that we have a collection of point estimates
    # rather than a full probability map
    if model_output.ndim != 2 or model_output.shape[1] != 2:
        raise TypeError(
            'Expected `model_output` to have shape (n_estimates, 2)' \
            f'but encountered: {model_output.shape}. Maybe the model is' \
            'outputting a probability distribution instead of a collection' \
            'of point estimates?'
       )

    # setup grids on which to smooth the precitions and create the pmfs
    MM_TO_CM = 10
    arena_dims_cm = np.array(arena_dims) / MM_TO_CM
    get_coords = lambda dim_cm: np.linspace(0, dim_cm, int(dim_cm / grid_resolution))

    xs, ys = map(get_coords, arena_dims_cm)
    xgrid, ygrid = np.meshgrid(xs, ys)
    coord_grid = np.dstack((xgrid, ygrid))

    # recenter origin for the location estimates (by default,
    # origin is placed in the middle of the room)
    model_output += arena_dims_cm / 2
    true_location += arena_dims_cm / 2

    # now assemble an array of probability mass functions by smoothing
    # the location estimates with each value of sigma
    pmfs = np.zeros((len(std_values), *xgrid.shape))

    for i, std in enumerate(std_values):
        for loc_estimate in model_output:
            # place a spherical gaussian with variance sigma
            # at the individual location estimate
            distr = scipy.stats.multivariate_normal(
                mean=loc_estimate,
                cov= (std ** 2)
            )
            # and add it to the corresponding entry in pmfs
            pmfs[i] += distr.pdf(coord_grid)

    # renormalize
    # get total sum over grid, adding axes so broadcasting works
    sum_per_sigma_val = pmfs.sum(axis=(1, 2)).reshape((-1, 1, 1))
    pmfs /= sum_per_sigma_val

    # repeat location so we can use the vectorized min_mass_containing_location fn
    true_loc_repeated = true_location.repeat(len(std_values), axis=0)

    # perform the calibration calculation step
    m_vals = min_mass_containing_location(
        pmfs,
        true_loc_repeated,
        xgrid,
        ygrid
    )

    # transform to the bin in [0, 1] to which each value corresponds,
    # essentially iteratively building a histogram with each step
    bins = np.linspace(0, 1, n_calibration_bins)
    # subtract one to track the left hand side of the bin
    bin_idxs = np.digitize(m_vals, bins) - 1

    return bin_idxs

def calibration_from_steps(cal_step_bulk: np.array):
    """
    Calculate calibration curves and error from the collected
    results of `calibration_step`.
    
    Args:
        cal_step_bulk: Results from `calibration_step`, should have
            shape (n_sigma_values, n_bins).
    """
    # calculate the calibration curve by taking the cumsum
    # and dividing by the total sum (adding extra axis so broadcasting works)
    calibration_curves = cal_step_bulk.cumsum(axis=1) / cal_step_bulk.sum(axis=1)[:, None]
    # next, calculate the errors
    # get probabilities the model assigned to each region
    # note: these are also the bucket edges in the histogram
    n_bins = calibration_curves.shape[1]
    assigned_probabilities = np.arange(1, n_bins + 1) / n_bins
    # get the sum of residuals between the assigned probabilities
    # and the true observed proportions for each value of sigma
    residuals = calibration_curves - assigned_probabilities
    abs_err = np.abs(residuals).sum(axis=1)
    signed_err = residuals.sum(axis=1)
    return calibration_curves, abs_err, signed_err
