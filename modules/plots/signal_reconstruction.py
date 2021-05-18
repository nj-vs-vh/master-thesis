import numpy as np
import matplotlib.pyplot as plt
from corner import corner

from typing import Optional, List
from nptyping import NDArray

from modules.plots.experimental_data import plot_signals_frame, CHANNEL_NO_LABEL
from modules.plots._shared import Color, Figsize, _save_or_show, TIME_LABEL
from modules.experiment.events import Event, EventProcessor


def plot_signal_reconstruction(event: Event, processor: EventProcessor):
    fig = plt.figure(figsize=Figsize.TWOPANEL_VERT.value)
    gs = plt.GridSpec(ncols=1, nrows=5, height_ratios=[4, 4, 2, 2, 2], figure=fig)

    ax_orig = fig.add_subplot(gs[0])
    plot_signals_frame(event, fig_ax=(fig, ax_orig), window=processor.N)
    # ax_orig.set_xticklabels([])
    ax_orig.set_xlabel('')

    ax_deconv = fig.add_subplot(gs[1], sharex=ax_orig)
    deconv_frame, signal_t, channels = processor.read_deconv_frame(event.id_)
    mesh = ax_deconv.pcolormesh(channels, signal_t, deconv_frame, shading='nearest')
    cbar = plt.colorbar(mesh)
    cbar.set_label('$n_{{ph.el}}$')
    ax_deconv.set_ylabel(TIME_LABEL)

    ax_n_eas = fig.add_subplot(gs[2], sharex=ax_orig)
    ax_t_mean = fig.add_subplot(gs[3], sharex=ax_orig)
    ax_sigma_t = fig.add_subplot(gs[4], sharex=ax_orig)

    channels_with_signals = []
    n_eas = []  # two columns: value and it's std error
    t_mean = []
    sigma_t = []
    for i_ch in range(109):  # TODO: unhardcode!
        try:
            theta_sample = processor.read_signal_reconstruction(event.id_, i_ch)
        except FileNotFoundError:
            continue
        channels_with_signals.append(i_ch)
        theta_mean = theta_sample.mean(axis=0)
        theta_std = theta_sample.std(axis=0)
        for storage, i_dim in zip([n_eas, t_mean, sigma_t], range(3)):
            storage.append([theta_mean[i_dim], theta_std[i_dim]])

    _, _, ref_width, _ = ax_orig.get_position().bounds

    channels_with_signals = np.array(channels_with_signals)
    signal_significances = processor.read_signal_significances(event.id_)
    signal_significances = signal_significances[channels_with_signals]
    STRONG_SIGNIFICANCE = 6
    weak_mask = np.logical_and(signal_significances > 0, signal_significances <= STRONG_SIGNIFICANCE)
    strong_mask = signal_significances > STRONG_SIGNIFICANCE

    for storage, ax, ylabel in zip(
        [n_eas, t_mean, sigma_t],
        [ax_n_eas, ax_t_mean, ax_sigma_t],
        ['$n_{{EAS}}$', '$\\mu_t$, бин', '$\\sigma_t$, бин'],
    ):
        storage = np.array(storage)
        ax.errorbar(
            channels_with_signals[strong_mask] + 1,
            storage[strong_mask, 0],
            yerr=storage[strong_mask, 1],
            fmt='.',
            color=Color.THETA.value,
        )
        ax.errorbar(
            channels_with_signals[weak_mask] + 1,
            storage[weak_mask, 0],
            yerr=storage[weak_mask, 1],
            fmt='.',
            color=Color.THETA.value,
            alpha=0.3,
        )
        ax.set_xlim(0, 109)
        ax.set_ylabel(ylabel)
        left, bottom, _, height = ax.get_position().bounds
        ax.set_position([left, bottom, ref_width, height])

    ax.set_xticks([1, 30, 60, 90, 109])
    ax.set_xlabel(CHANNEL_NO_LABEL)

    _save_or_show(None)
    return fig, [ax_orig, ax_deconv, ax_n_eas, ax_t_mean, ax_sigma_t]


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
        hist_bin_factor=1,
        plot_contours=True,
        color=Color.THETA.value,
        hist_kwargs={
            'histtype': 'bar',
        },
    )
