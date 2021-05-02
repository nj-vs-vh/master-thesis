"""
mcmc: wrapper around emcee module creating repeatable MCMC sampling routine
"""

import numpy as np
import emcee
from multiprocessing import Pool

from typing import Any, Callable
from nptyping import NDArray

from modules.utils import generate_poissonian_ns, slice_edge_effects


rng = np.random.default_rng()


class SamplingConfig:
    multiprocessing = False
    n_samples = 10 ** 3
    starting_points_strategy = 'around_estimation'  # 'around_estimation' or 'prior', see Foreman-Mackey et al. (2013)
    moves = [  # see https://emcee.readthedocs.io/en/stable/user/moves/#moves-user
        (emcee.moves.StretchMove(), 1),
        # (emcee.moves.DEMove(), 1),
        # (emcee.moves.DESnookerMove(), 1),
        # (emcee.moves.KDEMove(), 1),  # BAD -- low acceptance, very slow
    ]


def get_posterior_sample(
    logposterior: Callable[[NDArray[(Any,), float]], float],
    n_vec_estimation: NDArray[(Any,), float],
    L: int,
    progress: bool = False,
) -> NDArray[(Any, Any), float]:
    """High-level routine to sample posterior probability, automatically estimating burn-in and thinning

    Args:
        logposterior (Callable[[NDArray[(Any,), float]], float]): function to draw sample from
        n_vec_estimation (NDArray[(Any,), float]): initial guess for n_vec (= model params)
        progress (bool): flag to print progress bar while sampling, default is False
    """
    n_walkers = 2048
    N = n_vec_estimation.size

    starting_points = starting_points_from_estimation(
        n_vec_estimation, n_walkers, SamplingConfig.starting_points_strategy
    )

    pool = Pool() if SamplingConfig.multiprocessing else None

    sampler = emcee.EnsembleSampler(
        n_walkers,
        N,
        logposterior,
        moves=SamplingConfig.moves,
        pool=pool,
    )

    # roughly estimates target sampling error of each parameter (n in bin)
    # see for details: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr
    autocorr_is_good_if_less_than = 1 / 100  # fraction of number of samples drawn

    autocorr_estimation_each = 10
    samples_count = []
    autocorr_estimates = []
    # prev_tau = np.inf  # for relative tau drift
    for sample in sampler.sample(starting_points, iterations=SamplingConfig.n_samples, progress=progress):
        if sampler.iteration % autocorr_estimation_each:
            continue

        tau = sampler.get_autocorr_time(tol=0)

        tau = slice_edge_effects(tau, L, N)

        print(np.median(sampler.acceptance_fraction))

        autocorr_estimates.append(np.mean(tau))
        samples_count.append(sampler.iteration)

        if (
            np.all(tau < autocorr_is_good_if_less_than * sampler.iteration)
            # & np.all(np.abs(prev_tau - tau) / tau < 0.01)
        ):
            break
        # prev_tau = tau

    samples = sampler.get_chain(flat=True)

    if SamplingConfig.multiprocessing:
        pool.close()

    return samples_count, autocorr_estimates, samples


def starting_points_from_estimation(
    n_vec_estimation: NDArray[(Any,), float], n_walkers: int, strategy: str
) -> NDArray[(Any, Any), float]:
    """Use a single estimation to initialize a given number of walkers"""
    N = n_vec_estimation.size
    if strategy == 'prior':
        starting_points = np.zeros((n_walkers, N))
        for i_dim, n_vec_estimation_in_dim in enumerate(n_vec_estimation):
            starting_points[:, i_dim] = generate_poissonian_ns(
                n_vec_estimation_in_dim if n_vec_estimation_in_dim > 1 else 1, n_walkers
            )
    elif strategy == 'around_estimation':
        sigma_from_estimation = np.std(n_vec_estimation) / 10
        starting_points = np.tile(n_vec_estimation, (n_walkers, 1)) + rng.normal(
            scale=sigma_from_estimation, size=(n_walkers, n_vec_estimation.size)
        )
        starting_points = np.abs(starting_points)  # inverting clipping n
    else:
        raise ValueError(f"Unknown starting points generation strategy '{strategy}'")
    return starting_points


if __name__ == "__main__":

    from random import random
    from modules.randomized_ir import RandomizedIr, RandomizedIrEffect

    L_true = 3.5
    ir_x = np.linspace(0, L_true, int(L_true * 100))
    ir_y = np.exp(-ir_x)
    rir = RandomizedIr(ir_x, ir_y, factor=lambda: 0.5 + random() * 0.5)

    N = 10
    n_vec_mean = 20
    n_vec = generate_poissonian_ns(n_vec_mean, N)

    s_vec = rir.convolve_with_n_vec(n_vec)

    stats = RandomizedIrEffect(rir, samplesize=10 ** 5)
    n_vec_estimation = stats.estimate_n_vec(s_vec)
    loglike = stats.get_loglikelihood_mvn(s_vec)

    get_posterior_sample(loglike, n_vec_estimation, stats.L)
