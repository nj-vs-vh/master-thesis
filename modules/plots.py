import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import corner

from scipy.stats import norm

from enum import Enum

from typing import Any, Optional, List
from nptyping import NDArray


matplotlib.rcParams.update({'font.size': 12})


TIME_LABEL = 'Время, бины'


class Figsize(Enum):
    NORMAL = (7, 5)
    TRIPANEL = (7, 2)


class Color(Enum):
    N = '#0477DC'
    S = '#DC6904'
    N_ESTIMATION = '#e33b65'
    S_APPROX = '#db040b'


def _save_or_show(filename: Optional[str]):
    if filename:
        plt.savefig(f'../doc/pic/{filename}.pdf')
    else:
        plt.show()


def plot_convolution(n_vec: NDArray[(Any,), int], s_vec: NDArray[(Any,), float], filename=None):
    N = n_vec.size
    L = s_vec.size - N

    fig, ax1 = plt.subplots(figsize=Figsize.NORMAL.value)

    # bin stripes
    ax1.axhline(0, color='black')
    for i in range(N):
        ax1.axvspan(i, i + 1, facecolor=([0, 0, 0] if i % 2 == 0 else [0.3, 0.3, 0.3]), alpha=0.15, edgecolor=None)

    # counts
    ax1.bar(np.arange(N) + 0.5, n_vec, width=0.7, color=Color.N.value)

    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel('$n$', color=Color.N.value)
    ax1.tick_params(axis='y', labelcolor=Color.N.value)

    ax2 = ax1.twinx()

    t_S = np.arange(1, N + L + 1)

    def plot_signal_part(start, end, dashed):
        ax2.plot(t_S[start:end], s_vec[start:end], ('.-' if not dashed else '.:'), color=Color.S.value)

    plot_signal_part(0, L + 1, dashed=True)  # t in [1, L] with +1 point to the right (connection)
    plot_signal_part(L, N, dashed=False)  # t in [L+1, N]
    plot_signal_part(N - 1, N + L, dashed=True)  # t in [N+1, N+L] with +1 point to the left (connection)

    ax2.set_ylabel('$s$', color=Color.S.value)
    ax2.tick_params(axis='y', labelcolor=Color.S.value)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_xlim(0, N + L)

    _save_or_show(filename)
    return fig, [ax1, ax2]


def plot_mean_n_estimation(n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int], L: int, filename=None):
    N = n_vec.size

    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    bin_indices = np.arange(N)
    ax.bar(bin_indices + 0.5, n_vec, width=0.7, color=Color.N.value, label='$\\vec{n}$')

    ax.hlines(
        *[
            np.concatenate((vec[:L], vec[-L:])) for vec in (n_vec_estimation, bin_indices, bin_indices + 1)
        ],
        colors=[Color.N_ESTIMATION.value],
        linewidths=[1],
        linestyles=['dotted'],
    )
    ax.hlines(
        *[
            vec[L:-L] for vec in (n_vec_estimation, bin_indices, bin_indices + 1)
        ],
        colors=[Color.N_ESTIMATION.value],
        linewidths=[3],
        label='Оценка $\\vec{n}$ в предположении $\\mathbb{E} \\; \\vec{S} = \\vec{s}$',
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('$n$')
    ax.legend()

    ax.set_ylim(bottom=0)
    ax.set_xlim(0, N)

    _save_or_show(filename)
    return fig, ax


def plot_mean_n_estimation_assessment(
    n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int], L: int, filename=None
):
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    n_vec_estimation = n_vec_estimation[L:-L]

    bin_edges = np.arange(n_vec.min() - 1, n_vec.max() + 1)
    for data, color, label in (
        (n_vec, Color.N.value, '$\\vec{n}$'),
        (np.round(n_vec_estimation), Color.N_ESTIMATION.value, 'Оценка $\\vec{n}$'),
    ):
        _, _, histogram = ax.hist(data, bins=bin_edges, color=color, alpha=0.4, density=True, label=label)
        ax.axvline(data.mean(), color=histogram[0]._facecolor, alpha=1)

    _, top = ax.get_ylim()
    ax.set_ylim([0, top * 1])

    ax.set_xlabel('$n$')
    ax.set_ylabel('Плотность вероятности')
    ax.legend()

    _save_or_show(filename)
    return fig, ax


def plot_bayesian_mean_estimation(
    n_vec: NDArray[(Any,), int], sample: NDArray[(Any, Any), float], L: int, filename=None
):
    N = n_vec.size
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    bin_centers = np.arange(N) + 0.5
    means = np.mean(sample, axis=0)
    stds = np.std(sample, axis=0)

    ax.bar(bin_centers, n_vec, width=0.9, color=Color.N.value, label='$\\vec{n}$')

    # cutting edges with low quality poserior
    bin_centers = bin_centers[L + 1 : -L]  # noqa
    means = means[L + 1 : -L]  # noqa
    stds = stds[L + 1 : -L]  # noqa
    ax.hlines(
        means,
        bin_centers - 0.5,
        bin_centers + 0.5,
        colors=[Color.N_ESTIMATION.value],
        linewidths=[2],
    )

    std_line_halfwidth = 0.3

    ax.hlines(
        means - stds,
        bin_centers - std_line_halfwidth,
        bin_centers + std_line_halfwidth,
        colors=[Color.N_ESTIMATION.value],
        linewidths=[1],
    )
    ax.hlines(
        means + stds,
        bin_centers - std_line_halfwidth,
        bin_centers + std_line_halfwidth,
        colors=[Color.N_ESTIMATION.value],
        linewidths=[1],
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, N)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('n')

    ax.legend()

    _save_or_show(filename)
    return fig, ax


N_plus_L = Any
N_sample = Any


def plot_S_j_marginal_normality_assessment(
    S_samples: List[NDArray[(N_plus_L, N_sample), float]], n_means: List[int], L: int, filename=None
):
    n_rows = len(n_means)
    figsize = list(Figsize.TRIPANEL.value)
    figsize[1] *= n_rows
    fig, axes = plt.subplots(ncols=3, nrows=n_rows, figsize=figsize, gridspec_kw={'hspace': 0.3})

    if n_rows == 1:
        axes = [axes]

    def normality_marker(S_sample):
        marker = np.square(np.mean(S_sample, axis=1) - np.median(S_sample, axis=1))
        marker[np.mean(S_sample, axis=1) < 1e-2] = np.nan
        return marker

    for i_row, (S_sample, n_mean, axes_row) in enumerate(zip(S_samples, n_means, axes)):
        axes_row[0].set_ylabel(f"$\\bar{{n}} = {n_mean}$")

        N = S_sample.shape[0] - L
        S_sample = S_sample[L + 1 : N, :]  # noqa

        marker = normality_marker(S_sample)
        marker_i_sorted = np.argsort(marker)
        marker_i_sorted = marker_i_sorted[np.isfinite(marker[marker_i_sorted])]
        indices = marker_i_sorted[[0, len(marker_i_sorted) // 2, len(marker_i_sorted) - 1]]

        for i_bin, ax, description_ind in zip(
            indices, axes_row, ['best', 'median', 'worst']
        ):
            sample = S_sample[i_bin, :]

            epdf, bins, _ = ax.hist(sample, bins=30, density=True, color=Color.S.value)
            bincenters = 0.5 * (bins[1:] + bins[:-1])
            normpdf = norm.pdf(bincenters, loc=sample.mean(), scale=sample.std())
            ax.plot(bincenters, normpdf, color=Color.S_APPROX.value)

            # ax.set_xlim([bins[0], bins[-1]])

            ax.set_yticks([])

            if i_row == n_rows - 1:
                ax.set_xlabel(f'$S_{{{description_ind}}}$')

            rms_error = np.sqrt(np.mean(np.square(epdf - normpdf)))
            relative_rms_error = 100 * rms_error / np.max(epdf)
            # print(f'{description} RMS error ({i_bin} bin) = {relative_rms_error:.2f}% from peak')
            ax.table(cellText=[[f'{relative_rms_error:.2f}%']], loc='best', edges='', colWidths=[0.5])

    _save_or_show(filename)
    return fig, axes


def plot_S_j_pairwise_normality_assessment(S_sample: NDArray[(N_plus_L, N_sample), float], L: int, filename=None):
    start_bin = 5
    end_bin = start_bin + min(L, 5)

    fig = corner.corner(
        S_sample[start_bin:end_bin, :].T,
        labels=[f'$S_{{{bin_i}}}$' for bin_i in np.arange(start_bin, end_bin + 1)],
        show_titles=True,
        bins=30,
        color=Color.S.value,
        hist_kwargs={
            'histtype': 'bar',
        },
    )

    _save_or_show(filename)
    return fig
