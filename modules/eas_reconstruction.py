"""
Fitting signal arrival times with plane EAS front (bonus: adaptive point exclusion)
"""

import numpy as np
from scipy.optimize import curve_fit
from numba import njit

from typing import Tuple
from nptyping import NDArray

import modules.mcmc as mcmc
from modules.utils import angle_between, norm_cdf, apply_mask


def arrival_time_plane(xdata, theta, phi, z00):
    x = xdata[:, 0]
    y = xdata[:, 1]
    return z00 - np.tan(theta) * np.cos(phi) * x - np.tan(theta) * np.sin(phi) * y


def adaptive_excluding_fit(
    x_fov,
    y_fov,
    t_means,
    t_stds,
    min_points_to_leave: int = 6,
    acceptable_angle_between: float = 0.1,
    absolute_distance_exclusion: bool = True,
):
    """
    Perform linear fit of EAS arrival times, excluding points one by one until min_points_to_leave left OR until
    an angle between consecutive fits is less than acceptable_angle_between degrees.

    The point to be excluded is the furthest from the current plane either by absolute delta or by z-value
    (scaled down by standard dev), as controlled by absolute_distance_exclusion argument.
    """
    popt = [np.pi / 3, 0, 0]
    x_y = np.concatenate((np.expand_dims(x_fov, 1), np.expand_dims(y_fov, 1)), axis=1)
    in_fit_mask = np.ones_like(t_means, dtype=bool)

    acceptable_angle_between *= np.pi / 180

    points_left = x_fov.size
    while points_left > min_points_to_leave:
        new_popt, new_pcov = curve_fit(
            f=arrival_time_plane,
            xdata=x_y[in_fit_mask],
            ydata=t_means[in_fit_mask],
            p0=popt,
            sigma=t_stds[in_fit_mask],
            absolute_sigma=True,
            bounds=([0, -np.pi, -np.inf], [np.pi / 2, np.pi, np.inf]),
        )
        if angle_between(*popt[:2], *new_popt[:2]) < acceptable_angle_between:
            break
        popt = new_popt
        pcov = new_pcov

        scale = 1 if absolute_distance_exclusion else t_stds
        abs_residuals = np.ma.masked_array(
            np.abs(arrival_time_plane(x_y, *popt) - t_means) / scale, mask=np.logical_not(in_fit_mask)
        )
        excluded_point_i = abs_residuals.argmax(fill_value=0)
        in_fit_mask[excluded_point_i] = False
        points_left -= 1

    perr = np.sqrt(np.diag(pcov))

    return popt, perr, in_fit_mask


def get_axis_position_logprior_and_loglike(x, y, n_mean, n_std):
    ch_x_y = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)

    max_ch_r = np.max(np.sqrt(np.sum(ch_x_y ** 2, axis=1)))

    n_disp = n_std ** 2
    N_pts = len(x)

    @njit
    def logprior(ax_x_y):
        r = np.sqrt(np.sum(ax_x_y ** 2))
        return 0 if r < 3 * max_ch_r else -np.inf

    @njit
    def loglike(ax_x_y):
        r = np.sqrt(np.sum((ch_x_y - ax_x_y) ** 2, axis=1))
        sort_is = np.argsort(r)  # sorting from shower axis to the side
        r = r[sort_is]
        n_mean_ = n_mean[sort_is]
        n_disp_ = n_disp[sort_is]

        # @njit
        def is_greater_prob(i, j):  # P(i>j)
            mu = n_mean_[i] - n_mean_[j]
            sigma = np.sqrt(n_disp_[i] + n_disp_[j])
            return 1 - norm_cdf(np.array([0.0]), mu=mu, sigma=sigma)

        logp = 0.0
        for i in range(N_pts):
            for j in range(i + 1, N_pts):
                logp += np.log(is_greater_prob(i, j))[0]

        return logp

    return logprior, loglike


def estimate_axis_position(x_fov, y_fov, n_means, n_stds, in_fit_mask):
    """Return max-likelihood axis position"""
    logprior, loglike = get_axis_position_logprior_and_loglike(
        *apply_mask(x_fov, y_fov, n_means, n_stds, mask=in_fit_mask)
    )

    def logposterior(ax):
        logp = logprior(ax)
        return logp if np.isinf(logp) else logp + loglike(ax)

    n_walkers = 128
    i_max_ch = np.argmax(n_means[in_fit_mask])
    max_ch_xy = np.array([x_fov[in_fit_mask][i_max_ch], y_fov[in_fit_mask][i_max_ch]]).reshape(1, 2)
    # max_ch_xy = np.array([0, 0], dtype=float)  # test of convergence

    sigma_estimation = 10
    init_point = np.tile(max_ch_xy, (n_walkers, 1)) + np.random.normal(scale=sigma_estimation, size=(n_walkers, 2))

    tau = 200
    result = mcmc.run_mcmc(
        logposterior=logposterior,
        init_point=init_point,
        config=mcmc.SamplingConfig(
            n_walkers=n_walkers,
            n_samples=10 * tau,
            starting_points_strategy='given',
            progress_bar=True,
            autocorr_estimation_each=500,
            debug_acceptance_fraction_each=1000,
        ),
    )
    axis_xy_sample = mcmc.extract_independent_sample(result.sampler, tau_override=tau, debug=True)
    i_max_in_sample = np.argmax(np.array([loglike(axis_xy) for axis_xy in axis_xy_sample]))

    return axis_xy_sample[i_max_in_sample, :]


def project_on_shower_plane(x: NDArray, y: NDArray, theta: float, phi: float) -> Tuple[NDArray, NDArray]:
    r_d = np.sqrt(x ** 2 + y ** 2)
    phi_d = np.arctan(y / x)
    phi_d[x < 0] += np.pi
    phi_d[np.logical_and(x > 0, y < 0)] += 2 * np.pi

    eta = np.pi / 2 - (phi - phi_d)
    x_pl = - r_d * np.cos(eta)
    y_pl = - r_d * np.sin(eta) * np.cos(theta)
    return x_pl, y_pl
