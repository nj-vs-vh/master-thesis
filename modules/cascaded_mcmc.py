"""
cascaded_mcmc: several MCMC sampling processes run consequently with results of a given mcmc being starting points
               for the next
"""

import numpy as np
import dataclasses

from typing import Callable, List
from nptyping import NDArray

from . import mcmc


def run_cascaded_mcmc(
    loglikes: List[Callable[[NDArray], float]],
    n_walkers_list: List[int],
    n_samples_list: List[int],
    n_vec_estimation: NDArray,
    L: int,
    generic_sampling_config: mcmc.SamplingConfig,
) -> mcmc.SamplingResult:
    """Run several MCMC samplers in sequence

    Args:
        loglikes (List[Callable[[NDArray], float]]): loglike functions to sample on each step
        n_walkers_list (List[int]): n_walkers for each step
        n_samples_list (List[int]): n_samples for each step
        n_vec_estimation (NDArray): initial point for the first sampler
        L (int): same as everywhere else (rir.L, rireff.L)
        generic_sampling_config (SamplingConfig): used to specify parameters other than n walkers, n samples:
                                                  multiprocessing, progress bar etc; they are the same for all samplers

    Returns:
        SamplingResult: final sampler's result
    """
    n_vec_min = np.zeros_like(n_vec_estimation, dtype=float)
    n_vec_max = n_vec_estimation * 100
    current_init_state = n_vec_estimation
    current_starting_points_strategy = 'around_estimation'

    for i, (loglike, n_walkers, n_samples) in enumerate(zip(loglikes, n_walkers_list, n_samples_list)):

        # cannot create this in separate function because then multiprocessing can't pickle it :(
        def current_logposterior(n_vec):
            if np.any(np.logical_or(n_vec < n_vec_min, n_vec > n_vec_max)):
                return -np.inf
            return loglike(n_vec)

        sampling_config = dataclasses.replace(
            generic_sampling_config,
            n_samples=n_samples,
            n_walkers=n_walkers,
            starting_points_strategy=current_starting_points_strategy,
        )

        result = mcmc.run_mcmc(current_logposterior, current_init_state, L, config=sampling_config)

        # extracting the next sampler's init state
        if i != len(loglikes) - 1:
            current_starting_points_strategy = 'given'
            current_init_state = mcmc.extract_independent_sample(
                result.sampler, desired_sample_size=n_walkers_list[i + 1]
            )
            del result  # saving memory :)

    return result
