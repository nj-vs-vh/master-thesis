import numpy as np
import matplotlib.pyplot as plt
import corner

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy.stats import norm

from typing import Any, Optional, List
from nptyping import NDArray

from ._shared import TIME_LABEL, Figsize, Color, _save_or_show


def plot_data_similarity_test(data_x, data_y, label_1, label_2):
    """For comparison of presumably similar datasets with scatterplot"""
    fig, ax = plt.subplots(figsize=Figsize.SMALL.value)

    min_ = min(data_x.min(), data_y.min())
    max_ = min(data_x.max(), data_y.max())
    range_ = np.linspace(min_, max_)

    ax.scatter(data_x, data_y, c='k', marker='.')
    ax.plot(range_, range_, 'r-')

    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)

    return fig, ax


def plot_convolution(
    n_vec: NDArray[(Any,), int],
    s_vec: NDArray[(Any,), float],
    delta: Optional[float] = None,
    filename=None,
    fig_ax=None,
    end_x_axis_on_N=False,
):
    """Problem setup plot. If delta is None, plot signal as precise, othewise as delta-wide error bars"""
    N = n_vec.size
    L = s_vec.size - N

    fig, ax1 = fig_ax or plt.subplots(figsize=Figsize.NORMAL.value)

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

    if delta is not None:
        for i in range(int(s_vec.max() / delta) + 1):
            ax2.axhspan(
                delta * i,
                delta * (i + 1),
                facecolor=([1, 1, 1, 0] if i % 2 == 0 else [*Color.S.as_rgb(), 1]),
                alpha=0.05,
                edgecolor=None,
            )

    t_S = np.arange(1, N + L + 1)

    def plot_signal_part(start, end, dashed):
        if delta is None:
            ax2.plot(t_S[start:end], s_vec[start:end], ('.-' if not dashed else '.:'), color=Color.S.value)
        else:
            ax2.errorbar(
                t_S[start:end],
                s_vec[start:end] + delta / 2,
                yerr=delta / 2,
                fmt=('.-' if not dashed else '.:'),
                elinewidth=(2 if not dashed else 0.5),
                color=Color.S.value,
            )

    plot_signal_part(0, L + 1, dashed=True)  # t in [1, L] with +1 point to the right (connection)
    plot_signal_part(L, N, dashed=False)  # t in [L+1, N]
    plot_signal_part(N - 1, N + L, dashed=True)  # t in [N+1, N+L] with +1 point to the left (connection)

    ax2.set_ylabel('$s$', color=Color.S.value)
    ax2.tick_params(axis='y', labelcolor=Color.S.value)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.set_xlim(0, N + L if not end_x_axis_on_N else N)

    _save_or_show(filename)
    return fig, [ax1, ax2]


def _plot_n_estimation(ax: plt.Axes, n_vec_estimation: NDArray[(Any,), int], L: int):
    N = n_vec_estimation.size
    bin_indices = np.arange(N)

    ax.hlines(
        *[np.concatenate((vec[: L + 1], vec[-L:])) for vec in (n_vec_estimation, bin_indices, bin_indices + 1)],
        colors=[Color.N_ESTIMATION.value],
        linewidths=[2],
        linestyles=['dotted'],
    )
    est_hlines = ax.hlines(
        *[vec[L + 1 : -L] for vec in (n_vec_estimation, bin_indices, bin_indices + 1)],  # noqa
        colors=[Color.N_ESTIMATION.value],
        linewidths=[3],
        label='Оценка $\\vec{n}$ в предположении $\\mathbb{E} \\; \\vec{S} = \\vec{s}$',
    )
    return est_hlines


def plot_mean_n_estimation(n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int], L: int, filename=None):
    N = n_vec.size

    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    bin_indices = np.arange(N)
    ax.bar(bin_indices + 0.5, n_vec, width=0.7, color=Color.N.value, label='$\\vec{n}$')

    _plot_n_estimation(ax, n_vec_estimation, L)

    ax.set_ylim(bottom=0, top=np.maximum(n_vec, n_vec_estimation).max() * 1.3)
    ax.set_xlim(0, N)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('$n$')
    ax.legend()

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
    n_vec: NDArray[(Any,), int],
    sample: NDArray[(Any, Any), float],
    L: int,
    n_vec_estimation: Optional[NDArray[(Any,), float]] = None,
    filename=None,
    fig_ax=None,
):
    fig, ax = fig_ax or plt.subplots(figsize=Figsize.NORMAL.value)
    N = n_vec.size
    legend_handles = []

    bin_centers = np.arange(N) + 0.5

    N_BARS_ALPHA = 0.3
    n_bars = ax.bar(
        bin_centers,
        n_vec,
        width=0.7,
        color=[*Color.N.as_rgb(), N_BARS_ALPHA],
        edgecolor=Color.N.value,
        linewidth=2,
        label='$\\vec{n}$',
    )
    legend_handles.append(n_bars)

    if n_vec_estimation is not None:
        est_hlines = _plot_n_estimation(ax, n_vec_estimation, L)
        legend_handles.append(est_hlines)

    inferred_color_rgb = Color.N_INFERRED.as_rgb()

    bin_hists_cm = ListedColormap(
        colors=np.array(
            [[*inferred_color_rgb, alpha] for alpha in np.linspace(0, 1, 100)],
        ),
        name='N_estimation_alphamap',
    )

    for i_bin, sample_in_bin in enumerate(sample.T):
        if i_bin <= L or i_bin >= N - L:
            continue
        hist_in_bin, n_value_bin_edges = np.histogram(sample_in_bin, bins=10, density=True)
        pcm_Y = 0.5 * (n_value_bin_edges[:-1] + n_value_bin_edges[1:])
        pcm_X = np.array([i_bin, i_bin + 1])
        pcm_C = np.tile(hist_in_bin, (2, 1)).T
        ax.pcolormesh(
            pcm_X, pcm_Y, pcm_C, shading='flat', edgecolors=[(0, 0, 0, 0)], cmap=bin_hists_cm, antialiased=True
        )

    legend_handles.append(
        Patch(
            facecolor=bin_hists_cm(0.75),
            label='Результат байесовской деконволюции $\\vec{{n}}$',
        )
    )

    top_defining_values = np.maximum(n_vec, n_vec_estimation) if n_vec_estimation is not None else n_vec
    ax.set_ylim(bottom=0, top=top_defining_values.max() * 1.7)
    ax.set_xlim(0, N)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('n')

    ax.legend(handles=legend_handles)

    _save_or_show(filename)
    return fig, ax


def plot_bayesian_mean_estimation_in_bin(
    n_vec, samples: List, sample_names: List, ibin: int, n_vec_estimation: NDArray, filename=None
):
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    samples_in_bin = [s[:, ibin] for s in samples]
    min_overall = min(s.min() for s in samples_in_bin)
    max_overall = max(s.max() for s in samples_in_bin)
    bin_edges = np.linspace(min_overall, max_overall, num=30)

    for i, (sib, name) in enumerate(zip(samples_in_bin, sample_names)):
        color_i = np.array(Color.N_INFERRED.as_rgb())
        color_i *= ((1 + i) / (len(samples))) ** 2
        ax.hist(sib, bins=bin_edges, density=True, alpha=0.7, color=color_i, label=name)

    ax.axvline(n_vec[ibin], color=Color.N.value, label='Истинное значение')
    ax.axvline(n_vec_estimation[ibin], color=Color.N_ESTIMATION.value, label='Грубая оценка')

    ax.set_xlabel(f'$n_{{{ibin}}}$')
    ax.legend()

    _, top = ax.get_ylim()
    ax.set_ylim(top=1.5 * top)

    _save_or_show(filename)

    return fig, ax


N_plus_L = Any
N_sample = Any


def plot_S_j_marginal_normality_assessment(
    S_samples: List[NDArray[(N_plus_L, N_sample), float]], n_means: List[int], L: int, filename=None
):
    n_rows = len(n_means)
    figsize = list(Figsize.TRIPANEL_HORIZ.value)
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

        for i_bin, ax, description_ind in zip(indices, axes_row, ['best', 'median', 'worst']):
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
