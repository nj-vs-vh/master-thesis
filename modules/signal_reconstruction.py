"""
Loglikelihood and initial point for MCMC-based signal-noise separation
"""

import numpy as np
from numba import njit

from modules import utils

from typing import Any
from nptyping import NDArray


N_ = Any
Samplesize = Any
Signal = NDArray[(Samplesize,), float]
SignalSample = NDArray[(Samplesize, N_), float]


def estimate_theta_from_sample(sample: SignalSample, signal_t: Signal, n_walkers: int) -> NDArray:  # ready for init pts
    means = sample.mean(axis=0)
    t_est_index = np.argmax(means)
    t_init = np.random.normal(loc=signal_t[t_est_index], scale=2.0, size=(n_walkers, 1))
    n_est = np.sum(means[t_est_index - 1 : t_est_index + 1])  # noqa
    n_init = np.abs(np.random.normal(loc=n_est, scale=1.0, size=(n_walkers, 1)))
    sigma_est = 1
    sigma_init = np.abs(np.random.normal(loc=sigma_est, scale=1.0, size=(n_walkers, 1)))

    return np.concatenate((n_init, t_init, sigma_init), axis=1)


def get_logprior(signal_sample: SignalSample, signal_t: Signal, mean_n_phels: float):
    t_mean_min, t_mean_max = signal_t[0], signal_t[-1]

    signal_mean = signal_sample.mean(axis=0)
    signal_std = signal_sample.std(axis=0)
    n_eas_min = 0.0
    # n_eas_max = np.sum(signal_mean + signal_std) - mean_n_phels * (t_mean_max - t_mean_min)
    n_eas_max = np.sum(signal_mean + 5 * signal_std)

    sigma_min = 0.0
    sigma_max = (t_mean_max - t_mean_min) / 2

    theta_min = np.array([n_eas_min, t_mean_min, sigma_min])
    theta_max = np.array([n_eas_max, t_mean_max, sigma_max])

    def logprior(theta):
        if not ((theta_min <= theta).all() and (theta <= theta_max).all()):
            return -np.inf
        return (
            # exp decaying prior on n_EAS
            # see /media/njvh/last-child/Science archive/1маг/пакеты фотонов/pics/N_distribution.png
            - theta[0] / 40
            # gaussian prior on arrival time sigma
            # see std_range_stats.png in the same dir
            - (theta[2] - 2.4) / (2)
        )

    return logprior


def get_signal_reconstruction_loglike(
    signal_sample: SignalSample,
    signal_t: Signal,
    mean_n_phels: float,
    simulate_packets: bool = False,
    njitted: bool = True,
):
    t_first_bin = signal_t[0] - 1
    bin_edges = np.arange(signal_sample.shape[1] + 1)
    samplesize, N = signal_sample.shape

    def loglike(theta):
        n_eas, t_mean, sigma_t = theta
        t_relative = t_mean - t_first_bin
        if not simulate_packets:
            cdf_at_bin_edges = utils.norm_cdf(bin_edges, mu=t_relative, sigma=sigma_t)
            n_eas_per_bin = n_eas * np.diff(cdf_at_bin_edges)  # diff(cdf) gives probabilities in bins
        else:
            packets = np.random.normal(loc=t_relative, scale=sigma_t, size=(samplesize, int(n_eas)))
            n_eas_per_bin = np.zeros_like(signal_sample, dtype='int')
            for i_sample in range(samplesize):
                n_eas_per_bin[i_sample, :] = np.histogram(packets[i_sample, :], bins=bin_edges)[0]
        only_noise_sample = signal_sample - n_eas_per_bin

        pmfs = utils.poisson_pmf(only_noise_sample, lmb=mean_n_phels)

        log_p = 0
        for pmf_in_bin in pmfs.T:
            log_p += np.log(np.mean(pmf_in_bin))
        return log_p if not np.isnan(log_p) else -np.inf

    if njitted:
        loglike = njit(loglike)

    return loglike


if __name__ == "__main__":
    from pathlib import Path

    # import modules.experiment.rir as exprir
    import modules.experiment.events as expevents

    # import modules.plots.deconvolution as dec_plots
    # import modules.plots.experimental_data as exp_plots
    # from modules import utils, mcmc
    # import modules.signal_reconstruction as sigrec

    N = 45
    # _, rireff = exprir.get_rireffs(N)

    DECONV_RESULTS_DIR = Path('./temp-data/deconvolution-results')

    def read_deconv_result(event_id, i_ch):
        deconv_path = DECONV_RESULTS_DIR / f"{event_id}/{i_ch}.deconv.npz"
        if not deconv_path.exists():
            raise FileNotFoundError(f"No saved doconvolution for {event_id} event {i_ch} channel")
        data = np.load(deconv_path)
        return data['sample'], data['signal_t']

    def read_deconv_frame(event_id):
        # frame = np.zeros(109, N):
        frame = []
        channels = np.arange(109)

        nan_signal = np.zeros((N,))
        nan_signal[:] = np.nan
        for i_ch in channels:
            try:
                sample, _ = read_deconv_result(event_id, i_ch)
                frame.append(sample.mean(axis=0))
            except FileNotFoundError:
                frame.append(nan_signal)
        return np.array(frame).T, channels

    event_id = 10675
    i_ch = 1

    event = expevents.Event(event_id)
    sample, signal_t = read_deconv_result(event_id, i_ch)
    mean_n_phels = event.mean_n_photoelectrons[i_ch]
    loglike = get_signal_reconstruction_loglike(sample, signal_t, mean_n_phels, simulate_packets=False)

    print(loglike(np.array([1.06872767e02, 4.21386480e02, 1.09149194e-01])))
