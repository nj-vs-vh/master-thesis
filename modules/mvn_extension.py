"""
mvn_extension: custom extension to scipy.stats.multivariate_normal
"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import mvn

from nptyping import NDArray
from typing import Any
from scipy.stats._multivariate import multivariate_normal_frozen


rng: np.random.Generator = default_rng()


def integrate_pdf(
    rv: multivariate_normal_frozen, min_: NDArray[(Any,), float], delta: float, debug: bool = False
) -> float:
    """Integrate a given multivariate random variable rv's PDF in n-dimensional box from min_ to max_. Equivalent to
    rv.cdf(max_) - rv.cdf(min_) but avoids integrating up to infinity two times

    Modeled after:
    https://github.com/scipy/scipy/blob/4ec4ab8d6ccc1cdb34b84fdcb66fde2cc0210dbf/scipy/stats/_multivariate.py#L531
    """
    min_ = rv._dist._process_quantiles(min_, rv.dim)
    max_ = rv._dist._process_quantiles(min_ + delta, rv.dim)
    integral, info = mvn.mvnun(min_, max_, rv.mean, rv.cov, rv.maxpts, rv.abseps, rv.releps)
    if debug:
        if info == 0:
            print('error < eps, probability integral converged')
        elif info == 1:
            print('error > eps, all maxpts used')
    return integral


def integrate_pdf_fast(
    rv: multivariate_normal_frozen,
    min_: NDArray[(Any,), float],
    delta: float,
    n_pts: int = 100,
    debug: bool = False,
) -> float:
    """Faster equivalent of P_between for integration over a small n-dim box with a simple Monte-Carlo method"""
    pdf_vals = np.empty((n_pts,))
    for i, qp in enumerate(rng.uniform(low=min_, high=min_ + delta, size=(n_pts, rv.dim))):
        pdf_vals[i] = rv.pdf(qp)
    return np.mean(pdf_vals) * (delta ** rv.dim)
