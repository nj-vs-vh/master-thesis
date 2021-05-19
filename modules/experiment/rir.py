"""
Experimental Randomized IR reading and preprocessing
"""


import numpy as np
import numdifftools as nd
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from pathlib import Path

from typing import Tuple, Any, Optional
from nptyping import NDArray

from modules.randomized_ir import RandomizedIr, RandomizedIrEffect


CUR_DIR = Path(__file__).parent
PMT_DATA_DIR = CUR_DIR / '../../experimental-data/pmt-characteristics'


FloatVector = NDArray[(Any,), float]

rng: np.random.Generator = np.random.default_rng()


def _normalized_ir_shape(ir_t, ir_shape) -> FloatVector:
    integral = np.sum(ir_shape) * (ir_t[1] - ir_t[0])
    return ir_shape / integral


def read_ir_shape() -> Tuple[FloatVector, FloatVector]:
    ir_shape = np.loadtxt(PMT_DATA_DIR / 'ir-shape.txt')
    ir_t = np.arange(0, ir_shape.size, dtype=float)  # ns, ir shape is sampled at 1 ns
    ir_t /= 12.5  # ns -> bin

    return ir_t, _normalized_ir_shape(ir_t, ir_shape)


def cut_ir_shape(
    ir_t: FloatVector,
    ir_shape: FloatVector,
    forced_L: Optional[int] = None,
    excluded_integral_percentile: Optional[float] = None,
) -> Tuple[FloatVector, FloatVector]:
    """Cut IR shape length either at fixed L or at the point where we are cutting less than
    a given percentile of IR's integral. If both are given, the shortest cut from two is applied.
    """
    if forced_L is not None:
        cut_index_forced = np.argmin(np.abs(ir_t - forced_L))
    else:
        cut_index_forced = ir_t.size
    if excluded_integral_percentile is not None:
        integration_step = ir_t[1] - ir_t[0]
        ir_cumintegral = np.cumsum(ir_shape) * integration_step
        cut_index_integral = np.argmin(np.abs(ir_cumintegral - (1 - excluded_integral_percentile)))
    else:
        cut_index_integral = ir_t.size
    cut_index = min(cut_index_integral, cut_index_forced)
    ir_t = ir_t[:cut_index]
    ir_shape = ir_shape[:cut_index]
    return ir_t, _normalized_ir_shape(ir_t, ir_shape)


# for regular ФЭУ 84/3 channels

Cpmt_invcdf_data = np.loadtxt(PMT_DATA_DIR / 'ir-amplification-invcdf.dat')
Cpmt_values_lookup = Cpmt_invcdf_data[:, 1] / 1.723  # normalizing to mean = 1
Cpmt_cdf_lookup = Cpmt_invcdf_data[:, 0]
Cpmt_invcdf_func = interp1d(Cpmt_cdf_lookup, Cpmt_values_lookup, kind='cubic', fill_value="extrapolate")
Cpmt_cdf_func = interp1d(Cpmt_values_lookup, Cpmt_cdf_lookup, kind='cubic', bounds_error=False, fill_value=(0, 1))


def read_C_pmt_cdf_pdf(n_sample_pts: int = 50) -> Tuple[FloatVector, FloatVector, FloatVector]:
    Cpmt_values = np.linspace(0, 1.01 * Cpmt_values_lookup.max(), n_sample_pts)
    Cpmt_pdf_func = nd.Derivative(Cpmt_cdf_func)
    return Cpmt_values, Cpmt_pdf_func(Cpmt_values), Cpmt_cdf_func(Cpmt_values)


def generate_C_pmt(n: int = 1) -> FloatVector:
    return Cpmt_invcdf_func(rng.uniform(size=n))


# for Hamamatsu Hamamatsu R3886, see Fig. 9 in
# Antonov, R. A., Bonvech, E. A., Chernov, D. V., Podgrudkov, D. A., & Roganova, T. M. (2016).
# The LED calibration system of the SPHERE-2 detector. Astroparticle Physics, 77, 55–65.
# https://doi.org/10.1016/j.astropartphys.2016.01.004

Ham_Cpmt_cut_gauss_mean = 0.499  # pC
Ham_Cpmt_cut_gauss_sigma = 0.23  # pC
truncnorm_a = - Ham_Cpmt_cut_gauss_mean / Ham_Cpmt_cut_gauss_sigma  # 0 expressed as Z value
Ham_Cpmt_rv = truncnorm(a=truncnorm_a, b=np.inf)
Ham_Cpmt_rv_mean = Ham_Cpmt_rv.mean()


def generate_Ham_C_pmt(n: int) -> FloatVector:
    return (Ham_Cpmt_rv.rvs(size=n) - truncnorm_a) / (Ham_Cpmt_rv_mean - truncnorm_a)


# Convinience function to create RandomizedIrEffects from real IR data

def get_rireffs(N: int) -> Tuple[RandomizedIrEffect, RandomizedIrEffect]:
    """
    Return RandomizedIrEffects for Hamamatsu and ФЭУ 84/3
    """
    ir_t, ir_shape = read_ir_shape()
    ir_t, ir_shape = cut_ir_shape(ir_t, ir_shape, excluded_integral_percentile=0.02)  # fine-tuned for reasonable length

    samplesize = 10 ** 5

    rir = RandomizedIr(ir_x=ir_t, ir_y=ir_shape, factor=generate_C_pmt)
    rireff = RandomizedIrEffect(rir, N, samplesize=samplesize)

    ham_rir = RandomizedIr(ir_x=ir_t, ir_y=ir_shape, factor=generate_Ham_C_pmt)
    ham_rireff = RandomizedIrEffect(ham_rir, N, samplesize=samplesize)
    return ham_rireff, rireff


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = generate_Ham_C_pmt(1000000)

    print(c.mean())

    plt.hist(c, bins=30)
    plt.savefig('hamamatsu_C_pmt_test.png')
