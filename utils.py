import numpy as np

from nptyping import NDArray
from numbers import Real


rng = np.random.default_rng()


def generate_poissonian_ns(n_mean: Real, count: int) -> NDArray[Real]:
    return rng.poisson(n_mean, count)
