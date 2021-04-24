import numpy as np
import timeit
from functools import wraps

from typing import Any
from nptyping import NDArray
from numbers import Real


rng = np.random.default_rng()


def generate_poissonian_ns(n_mean: Real, count: int) -> NDArray[Real]:
    return rng.poisson(n_mean, count)


def slice_edge_effects(mat: NDArray[(Any,), float], L: int, N: int, axis: int = 0):
    """See \\ref{sec:edge-effects}"""
    # t in [L + 1, N]; accounting for zero-based indexing of one-based time
    start_i, end_i = L, N - 1
    # and for the fact the slice does not include last point!
    end_i += 1

    # WTF?
    start_i -= 1

    return np.take(mat, range(start_i, end_i), axis=axis)


def timer(args_formatter=None):
    if not args_formatter:
        args_formatter = (
            lambda *args, **kwargs: (
                ", ".join(
                    [
                        s
                        for s in [", ".join(str(a) for a in args), ", ".join(f"{k}={w}" for k, w in kwargs.items())]
                        if s
                    ]
                )
            )
        )  # noqa

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            elapsed = timeit.default_timer() - start_time
            print(f'\n\n{func.__name__}({args_formatter(*args, **kwargs)}) took {elapsed:.3f} seconds to complete.')
            return result

        return wrapper

    return decorator
