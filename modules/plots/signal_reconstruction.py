import numpy as np
from corner import corner

from typing import Optional, List
from nptyping import NDArray


# def plot_


def plot_theta_sample(
    theta_sample: NDArray, n_sigmas_cut: Optional[float] = None, cut_along: Optional[List[bool]] = None
):
    if n_sigmas_cut is not None:
        if cut_along is None:
            cut_along = [True] * 3
        # cleaning theta sample before display assuming marginal distributions are somewhat close to normal
        theta_mean = theta_sample.mean(axis=0)
        theta_std = theta_sample.std(axis=0)
        theta_min = theta_mean - n_sigmas_cut * theta_std
        theta_max = theta_mean + n_sigmas_cut * theta_std

        theta_sample = theta_sample[
            np.logical_and(
                (theta_sample > theta_min)[:, cut_along].all(axis=1),
                (theta_sample < theta_max)[:, cut_along].all(axis=1),
            ),
            :,
        ]

    corner(
        theta_sample,
        labels=['$n_{{EAS}}$', '$\\mu_t$', '$\\sigma_t$'],
        show_titles=True,
        bins=20,
        hist_kwargs={
            'histtype': 'bar',
        },
    )
