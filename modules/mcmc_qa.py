"""
mcmc_qa: functions for mcmc sampling quality assessment
"""

import numpy as np

from numbers import Real
from typing import Any
from nptyping import NDArray


N_sample = Any
N_dim = Any


def learn_burn_in(sample: NDArray[(N_dim, N_sample), Real], trial_points_n: int = 10, i=0) -> int:
    """Heuristically determine burn-in period in a given sample produced by MCMC sampler. Uses marginal distributions'
    dispersions to find a point where they stabilize -- this is our estimated burn-in point.

    Args:
        sample (NDArray[(N_dim, N_sample), Real]): sample drawn from some distribution
    """
    n_dim, sample_n = sample.shape

    burn_in_points = np.linspace(0, sample_n, num=trial_points_n + 1).astype(int)[1:-1]
    dim_avg_std_diff = []
    for burn_in_point in burn_in_points:
        burn_in_std = np.std(sample[:, :burn_in_point], axis=1)
        good_std = np.std(sample[:, burn_in_point:], axis=1)
        dim_avg_std_diff.append(np.sqrt(np.mean(np.square(good_std - burn_in_std))))
    dim_avg_std_diff = np.array(dim_avg_std_diff)

    # subsamples = np.split(sample, np.linspace(0, sample.shape[1], num=subsample_n+1).astype(int)[1:-1], axis=1)
    # subsample_means = [np.mean(ss, axis=1) for ss in subsamples]
    # subsample_stds = [np.std(ss, axis=1) for ss in subsamples]

    # # averaged over dimensions (assuming they are of the same order and unit -- not very universal!)
    # subsample_mean_deviations_from_last = [
    #     np.sqrt(np.mean(np.square(ssm - subsample_means[-1]))) for ssm in subsample_means
    # ]
    # subsample_std_deviations_from_last = [
    #     np.sqrt(np.mean(np.square(sss - subsample_stds[-1]))) for sss in subsample_stds
    # ]
    # return subsample_mean_deviations_from_last, subsample_std_deviations_from_last
    return dim_avg_std_diff


if __name__ == "__main__":
    sample = np.load('notebooks/test.npy')
    print(learn_burn_in(sample.T, 20, 18))
