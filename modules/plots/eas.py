from mpl_toolkits.mplot3d import Axes3D

from typing import Any, List
from nptyping import NDArray


Vector = NDArray[(Any,), float]
MaskArr = NDArray[(Any,), bool]


def plot_scatter_with_errorbars(
    ax: Axes3D,
    x: Vector,
    y: Vector,
    means: Vector,
    stds: Vector,
    masks: List[MaskArr],
    markers: List[str],
    colors: List[str],
    ylabel: str,
):
    for mask, marker, color in zip(masks, markers, colors):
        ax.scatter(x[mask], y[mask], means[mask], marker=marker, color=color)
        for x_i, y_j, mu, sigma in zip(x[mask], y[mask], means[mask], stds[mask]):
            ax.plot([x_i, x_i], [y_j, y_j], [mu - sigma, mu + sigma], color=color)

    ax.view_init(azim=50, elev=30)
    ax.set_xlabel('$x_{{\\mathrm{{запад-восток}}}}$, м')
    ax.set_ylabel('$y_{{\\mathrm{{юг-север}}}}$, м')
    ax.set_zlabel(ylabel)
