"""
mcmc: wrapper around emcee module creating repeatable MCMC sampling routine
"""

import numpy as np
import emcee
from multiprocessing import Pool

from dataclasses import dataclass, field

from typing import Any, Callable, List, Tuple, Optional, Union
from nptyping import NDArray
from emcee.moves import Move
from emcee.ensemble import EnsembleSampler

from modules.utils import generate_poissonian_ns, slice_edge_effects


rng = np.random.default_rng()


@dataclass
class SamplingConfig:
    n_walkers: int = 512
    n_samples: int = 5000
    # 'given', 'around_estimation' or 'prior', see Foreman-Mackey et al. (2013)
    # if 'given', n_vec_estimation must be a (n_walkers, N) matrix with ready-to use starting points
    starting_points_strategy: str = 'around_estimation'
    # see https://emcee.readthedocs.io/en/stable/user/moves/#moves-user
    moves: List[Tuple[Move, float]] = field(default_factory=lambda: [(emcee.moves.StretchMove(), 1.0)])
    multiprocessing: bool = False
    autocorr_estimation_each: Optional[int] = None  # None to avoid estimation
    debug_acceptance_fraction_each: Optional[int] = None  # None to not debug
    progress_bar: bool = False


@dataclass
class SamplingResult:
    sampler: EnsembleSampler
    sample: Optional[NDArray] = None
    N_tau: Optional[Tuple[NDArray, NDArray]] = None


def run_mcmc(
    logposterior: Callable[[NDArray[(Any,), float]], float],
    init_point: Union[NDArray[(Any,), float], NDArray[(Any, Any), float]],
    L: int,
    config: SamplingConfig,
) -> NDArray[(Any, Any), float]:
    """High-level routine to sample posterior probability, automatically estimating burn-in and thinning

    Args:
        logposterior (Callable[[NDArray[(Any,), float]], float]): function to draw sample from
        init_point (NDArray[(Any,), float]): initial guess for n_vec (= model params) OR ini
        L (int): rir.L
        config (SamplingConfig): see SamplingConfig class for params
    """
    if config.starting_points_strategy == 'given':
        N = init_point.shape[1]
    else:
        if len(init_point.shape) > 1:
            raise ValueError(
                "2D init points array only allowed with SamplingCongig.starting_points_strategy='given', "
                + f"but '{config.starting_points_strategy}' is passed."
            )
        N = init_point.size
        init_point = starting_points_from_estimation(
            init_point, n_walkers=config.n_walkers, strategy=config.starting_points_strategy
        )

    pool = Pool() if config.multiprocessing else None

    try:
        sampler = emcee.EnsembleSampler(
            config.n_walkers,
            N,
            logposterior,
            moves=config.moves,
            pool=pool,
        )

        # roughly estimates target sampling error of each parameter (n in bin)
        # see for details: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr
        autocorr_is_good_if_less_than = 1 / 100  # fraction of number of samples drawn

        autocorr_estimated_at = []
        autocorr_estimates = []
        # prev_tau = np.inf  # for relative tau drift
        for sample in sampler.sample(init_point, iterations=config.n_samples, progress=config.progress_bar):
            if (
                config.debug_acceptance_fraction_each is not None
                and sampler.iteration % config.debug_acceptance_fraction_each == 0
            ):
                print('\n Current acc. frac.:' + str(np.median(sampler.acceptance_fraction)))

            if config.autocorr_estimation_each is None or sampler.iteration % config.autocorr_estimation_each != 0:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            tau = slice_edge_effects(tau, L, N)

            autocorr_estimates.append(np.mean(tau))
            autocorr_estimated_at.append(sampler.iteration)
            if (
                np.all(tau < autocorr_is_good_if_less_than * sampler.iteration)
                # & np.all(np.abs(prev_tau - tau) / tau < 0.01)
            ):
                break
            # prev_tau = tau

        sample = sampler.get_chain(flat=True)
    finally:
        if config.multiprocessing:
            pool.close()

    return SamplingResult(
        sampler=sampler,
        sample=sample,
        N_tau=(autocorr_estimated_at, autocorr_estimates) if config.autocorr_estimation_each is not None else None,
    )


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


def extract_independent_sample(
    sampler: EnsembleSampler, desired_sample_size: Optional[int] = None, debug: bool = False
) -> NDArray[(Any, Any), float]:
    tau = sampler.get_autocorr_time(quiet=True, tol=0)
    tau = tau[np.logical_not(np.isnan(tau))]

    burnin = int(2 * np.max(tau))
    thin = int(0.9 * np.min(tau))

    if debug:
        print(f'Autocorrelation time is estimated at {np.mean(tau)} (ranges from {np.min(tau)} to {np.max(tau)})')
        print(f'Burn-in = {burnin} samples')
        print(f'Thinning = {thin} samples')

    min_number_of_burnins_in_chain = 3
    if min_number_of_burnins_in_chain * burnin > sampler.iteration:
        raise ValueError(
            f"Chain seems too short! Length is {sampler.iteration}, but must be at least "
            + f"{min_number_of_burnins_in_chain}x longer than burn in time = {burnin}"
        )

    independent_sample = sampler.get_chain(discard=burnin, thin=thin, flat=True)

    if desired_sample_size is not None:
        independent_sample = independent_sample[-desired_sample_size:, :]
        if independent_sample.shape[0] < desired_sample_size:
            raise ValueError(
                f"Cannot extract {desired_sample_size} from sampling result, "
                + f"only {independent_sample.shape[0]} are available. "
                + "Lower desired_sample_size or increase n_samples parameter to get long enough chain."
            )

    return independent_sample


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

    rireff = RandomizedIrEffect(rir, N, samplesize=10 ** 4)
    n_vec_estimation = rireff.estimate_n_vec(s_vec)
    loglike = rireff.get_loglikelihood_mvn(s_vec)

    result = run_mcmc(
        loglike,
        n_vec_estimation,
        rireff.L,
        SamplingConfig(
            n_samples=1000,
            n_walkers=128,
            starting_points_strategy='around_estimation',
            progress_bar=True,
            autocorr_estimation_each=10,
        ),
    )

    print(result)
