"""
mcmc_qa: functions for mcmc sampling quality assessment
"""

import numpy as np

from numbers import Real
from typing import Any
from nptyping import NDArray


N_sample = Any
N_dim = Any


def learn_burn_in(sample: NDArray[(N_dim, N_sample), Real]):
    """Heuristically determine burn-in period in a given sample produced by MCMC sampler. Uses marginal distributions'
    dispersions to find a point where they stabilize -- this is our estimated burn-in point.

    Args:
        sample (NDArray[(N_dim, N_sample), Real]): sample drawn from some distribution
    """
    pass
