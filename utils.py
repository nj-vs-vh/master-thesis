import numpy as np

from typing import Any
from nptyping import NDArray
from numbers import Real


rng = np.random.default_rng()


def generate_poissonian_ns(n_mean: Real, count: int) -> NDArray[Real]:
    return rng.poisson(n_mean, count)


def slice_edge_effects(mat: NDArray[(Any,), float], L: int, N: int, axis: int = 0):
    """See \\ref{sec:edge-effects}"""
    # t in [L + 1, N]; accounting for zero-based indexing of one-based time
    start_i, end_i = L-1, N-1
    # and for the fact the slice does not include last point!
    end_i += 1

    return np.take(mat, range(start_i, end_i), axis=axis)
