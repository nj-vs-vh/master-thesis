"""
ndepdf: efficient calculation of empirical n-dimensional PDF used for Monte-Carlo likelihood estimation
"""

import numpy as np

from collections import defaultdict

from typing import Union, Any
from nptyping import NDArray


# for type hints
N_sample = Any
N_dim = Any


def ndepdf(
    sample: NDArray[(N_dim, N_sample), float],
    point: NDArray[(N_dim,), float],
    bins: Union[NDArray[(N_sample,), int], int],
    check_bin_count: bool = False,
) -> float:
    """Calculate empirical n-dimensional PDF (EPDF) from a sample from the distiribution at a given point

    Args:
        sample: NDArray[(N_dim, N_sample), float]: sample from distribution, row of columns-vectors
        point: NDArray[(N_dim,), float]: vector-point to estimate EPDF at
        bins: Union[NDArray[(N_dim,), int], int]: number of bins along each dimension or a single number
                                                  to use along all of them
        check_bin_count: bool: if set to True, function will check if bin count was set lowe enough. if the bin count is
                               too high, all bins wil contain 1-2 points and ePDF estimation is useless.
    """
    N_dim, N_sample = sample.shape
    point = np.broadcast_to(point, (N_dim,))
    bins = np.broadcast_to(np.array(bins), (N_sample,))

    # if the point lies outside distiribution's bounding n-dimensional rectangle, ePDF is 0
    mins = np.min(sample, axis=1)
    maxes = np.max(sample, axis=1)
    if not np.all(np.logical_and(mins <= point, point <= maxes)):
        return 0
    # creating binnings for each dimension
    # using list because binnings along each dimension may have different size
    binnings = []
    cell_volume = 1.0
    for min_in_dim, max_in_dim, bin_count_in_dim in zip(mins, maxes, bins):
        binning, step = np.linspace(min_in_dim, max_in_dim, bin_count_in_dim + 1, retstep=True)
        binnings.append(binning)
        cell_volume *= step
    # binning the sample and the point with numpy.digitize
    sample_binned = np.zeros_like(sample, dtype=int)
    point_binned = np.zeros_like(point, dtype=int)
    for i_dim, (sample_i, point_i, binning) in enumerate(zip(sample, point, binnings)):
        point_binned[i_dim] = np.digitize(point_i, binning)
        sample_binned[i_dim, :] = np.digitize(sample_i, binning)
    # creating histogram with sparse N_dim-dimensional array represented as dictionary of keys
    ndhist = defaultdict(lambda: 0)
    for sample_binned_i in sample_binned.T:
        ndhist[tuple(sample_binned_i)] += 1
    if check_bin_count:
        s = sorted(ndhist.values(), reverse=True)
        max_points_per_bin = s[0]
        if max_points_per_bin < 10:
            raise ValueError(
                f"Beens seem to be very small: there's only as much as {max_points_per_bin} points per bin. "
                + "Try lowering bins parameter"
            )
    # finally, estimate PDF as a number of points from sample in the same bin relative to the sample size
    return ndhist[tuple(point_binned)] / (N_sample * cell_volume)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from itertools import product
    from numpy.random import default_rng

    rng = default_rng()

    N_sample = 100000
    sample = rng.multivariate_normal(mean=[0, 0], cov=np.array([[0.5, 0.1], [0.1, 0.5]]), size=N_sample).T

    N_bins = 30
    # using hist2d for comparison
    counts, xedges, yedges, hist_img = plt.hist2d(sample[0, :], sample[1, :], bins=N_bins, density=True)

    plt.colorbar(mappable=hist_img)
    plt.savefig('test.png')

    xmeans = 0.5 * (xedges[1:] + xedges[:-1])
    ymeans = 0.5 * (yedges[1:] + yedges[:-1])

    residual_sqsum = 0
    for (i, x), (j, y) in product(enumerate(xmeans), enumerate(ymeans)):
        if counts[i, j] < 0.1:
            continue
        residual = ndepdf(sample, np.array([x, y]), bins=N_bins) - counts[i, j]
        residual_sqsum += residual ** 2
        # if abs(residual) > abs(max_residual):
        #     max_residual = residual

    residual_sqsum /= xmeans.size * ymeans.size
    print(np.sqrt(residual_sqsum))
