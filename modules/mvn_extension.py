"""
mvn_extension: custom extension to scipy.stats.multivariate_normal
"""

from scipy.stats import mvn

from nptyping import NDArray
from typing import Any
from scipy.stats._multivariate import multivariate_normal_frozen


def P_between(
    rv: multivariate_normal_frozen, min_: NDArray[(Any,), float], max_: NDArray[(Any,), float], debug: bool = False
) -> float:
    """Integrate a given multivariate random variable rv's PDF in n-dimensional box from min_ to max_. Equivalent to
    rv.cdf(max_) - rv.cdf(min_) but avoids integrating up to infinity two times

    Modeled after:
    https://github.com/scipy/scipy/blob/4ec4ab8d6ccc1cdb34b84fdcb66fde2cc0210dbf/scipy/stats/_multivariate.py#L531
    """
    min_ = rv._dist._process_quantiles(min_, rv.dim)
    max_ = rv._dist._process_quantiles(max_, rv.dim)
    integral, info = mvn.mvnun(min_, max_, rv.mean, rv.cov, rv.maxpts, rv.abseps, rv.releps)
    if debug:
        if info == 0:
            print('error < eps, probability integral converged')
        elif info == 1:
            print('error > eps, all maxpts used')
    return integral
