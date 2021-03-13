import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Optional
from nptyping import NDArray


N_COLOR = '#0477DC'
N_ESTIMATION_COLOR = '#e667b3'
S_COLOR = '#DC6904'


# fine-tuned for notebook
MID_FIGSIZE = (7, 5)
LARGE_FIGSIZE = (10, 7)


def plot_convolution(
    n_vec: NDArray[(Any,), int], S_vec: NDArray[(Any,), float], save_to_filename: Optional[str] = None
):
    N = n_vec.size
    L = S_vec.size - N

    fig, ax1 = plt.subplots(figsize=MID_FIGSIZE)

    # bin stripes
    ax1.axhline(0, color='black')
    for i in range(N):
        ax1.axvspan(i, i + 1, facecolor=([0, 0, 0] if i % 2 == 0 else [0.3, 0.3, 0.3]), alpha=0.15, edgecolor=None)

    # counts
    ax1.bar(np.arange(N) + 0.5, n_vec, width=0.7, color=N_COLOR)

    ax1.set_xlabel('time, bin')
    ax1.set_ylabel('$n_i$', color=N_COLOR)
    ax1.tick_params(axis='y', labelcolor=N_COLOR)

    ax2 = ax1.twinx()

    t_S = np.arange(1, N + L + 1)
    ax2.step(t_S, S_vec, '.-', where='post', color=S_COLOR)

    ax2.set_ylabel('$S_j$', color=S_COLOR)
    ax2.tick_params(axis='y', labelcolor=S_COLOR)

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    if save_to_filename:
        plt.savefig(f'./doc/pic/{save_to_filename}.pdf')
    else:
        plt.show()


def plot_mean_n_estimation(n_vec: NDArray[(Any,), int], n_vec_estimation: NDArray[(Any,), int]):
    N = n_vec.size

    fig, ax = plt.subplots(figsize=LARGE_FIGSIZE)

    bin_indices = np.arange(N)
    ax.bar(bin_indices + 0.5, n_vec, width=0.7, color=N_COLOR, label='$\\vec{n}$')
    ax.hlines(
        n_vec_estimation,
        bin_indices,
        bin_indices + 1,
        colors=[N_ESTIMATION_COLOR],
        label='$\\vec{n}$ estimation when $S_j = \\mathbb{E} \\; S_j$',
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.show()
