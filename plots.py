import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Any, Optional
from nptyping import NDArray


matplotlib.rcParams.update({'font.size': 12})


N_COLOR = '#0477DC'
N_ESTIMATION_COLOR = '#e33b65'
S_COLOR = '#DC6904'


TIME_LABEL = 'Время, бины'


# fine-tuned for notebook
MID_FIGSIZE = (7, 5)
LARGE_FIGSIZE = (10, 7)


def _save_or_show(filename: Optional[str]):
    if filename:
        plt.savefig(f'./doc/pic/{filename}.pdf')
    else:
        plt.show()


def plot_convolution(n_vec: NDArray[(Any,), int], s_vec: NDArray[(Any,), float], filename=None):
    N = n_vec.size
    L = s_vec.size - N

    fig, ax1 = plt.subplots(figsize=MID_FIGSIZE)

    # bin stripes
    ax1.axhline(0, color='black')
    for i in range(N):
        ax1.axvspan(i, i + 1, facecolor=([0, 0, 0] if i % 2 == 0 else [0.3, 0.3, 0.3]), alpha=0.15, edgecolor=None)

    # counts
    ax1.bar(np.arange(N) + 0.5, n_vec, width=0.7, color=N_COLOR)

    ax1.set_xlabel(TIME_LABEL)
    ax1.set_ylabel('$n$', color=N_COLOR)
    ax1.tick_params(axis='y', labelcolor=N_COLOR)

    ax2 = ax1.twinx()

    t_S = np.arange(1, N + L + 1)

    def plot_signal_part(start, end, dashed):
        ax2.plot(t_S[start:end], s_vec[start:end], ('.-' if not dashed else '.:'), color=S_COLOR)

    plot_signal_part(0, L + 1, dashed=True)  # t in [1, L] with +1 point to the right (connection)
    plot_signal_part(L, N, dashed=False)  # t in [L+1, N]
    plot_signal_part(N - 1, N + L, dashed=True)  # t in [N+1, N+L] with +1 point to the left (connection)

    ax2.set_ylabel('$s$', color=S_COLOR)
    ax2.tick_params(axis='y', labelcolor=S_COLOR)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    _save_or_show(filename)


def plot_mean_n_estimation(n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int], L: int, filename=None):
    N = n_vec.size

    fig, ax = plt.subplots(figsize=MID_FIGSIZE)

    bin_indices = np.arange(N)
    ax.bar(bin_indices + 0.5, n_vec, width=0.7, color=N_COLOR, label='$\\vec{n}$')

    ax.hlines(
        n_vec_estimation[:L],
        bin_indices[:L],
        bin_indices[:L] + 1,
        colors=[N_ESTIMATION_COLOR],
        linewidths=[2],
        linestyles=['dotted'],
    )
    ax.hlines(
        n_vec_estimation[L:],
        bin_indices[L:],
        bin_indices[L:] + 1,
        colors=[N_ESTIMATION_COLOR],
        linewidths=[2],
        label='Оценка $\\vec{n}$ в предположении $\\mathbb{E} \\; \\vec{S} = \\vec{s}$',
    )

    ax.set_ylim(bottom=0)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('$n$')
    ax.legend()

    _save_or_show(filename)


def plot_mean_n_estimation_assessment(
    n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int], L: int, filename=None
):
    fig, ax = plt.subplots(figsize=MID_FIGSIZE)

    n_vec_estimation = n_vec_estimation[L:-L]

    bin_edges = np.arange(n_vec.min() - 1, n_vec.max() + 1)
    for data, color, label in (
        (n_vec, N_COLOR, '$\\vec{n}$'),
        (np.round(n_vec_estimation), N_ESTIMATION_COLOR, 'Оценка $\\vec{n}$'),
    ):
        _, _, histogram = ax.hist(data, bins=bin_edges, color=color, alpha=0.4, density=True, label=label)
        ax.axvline(data.mean(), color=histogram[0]._facecolor, alpha=1)

    _, top = ax.get_ylim()
    ax.set_ylim([0, top * 1])

    ax.set_xlabel('$n$')
    ax.set_ylabel('Плотность вероятности')
    ax.legend()

    _save_or_show(filename)


def plot_bayesian_mean_estimation(
    n_vec: NDArray[(Any,), int], sample: NDArray[(Any, Any), float], L: int, filename=None
):
    N = n_vec.size
    fig, ax = plt.subplots(figsize=MID_FIGSIZE)

    bin_centers = np.arange(N) + 0.5
    means = np.mean(sample, axis=0)
    stds = np.std(sample, axis=0)

    ax.bar(bin_centers, n_vec, width=0.9, color=N_COLOR, label='$\\vec{n}$')

    # cutting edges with low quality poserior
    bin_centers = bin_centers[L + 1 : -L]  # noqa
    means = means[L + 1 : -L]  # noqa
    stds = stds[L + 1 : -L]  # noqa
    ax.hlines(
        means,
        bin_centers - 0.5,
        bin_centers + 0.5,
        colors=[N_ESTIMATION_COLOR],
        linewidths=[2],
    )

    std_line_halfwidth = 0.4

    ax.hlines(
        means - stds,
        bin_centers - std_line_halfwidth,
        bin_centers + std_line_halfwidth,
        colors=[N_ESTIMATION_COLOR],
        linewidths=[1],
    )
    ax.hlines(
        means + stds,
        bin_centers - std_line_halfwidth,
        bin_centers + std_line_halfwidth,
        colors=[N_ESTIMATION_COLOR],
        linewidths=[1],
    )
    ax.set_ylim(bottom=0)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('n')

    ax.legend()
