import numpy as np
import numdifftools as nd
from scipy.interpolate import interp1d

from typing import Tuple, Any, Optional
from nptyping import NDArray


FloatVector = NDArray[(Any,), float]

rng: np.random.Generator = np.random.default_rng()


def _normalized_ir_shape(ir_t, ir_shape) -> FloatVector:
    integral = np.sum(ir_shape) * (ir_t[1] - ir_t[0])
    return ir_shape / integral


def read_ir_shape() -> Tuple[FloatVector, FloatVector]:
    ir_shape = np.loadtxt('../experimental-data/pmt-characteristics/ir-shape.txt')
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


Cpmt_invcdf_data = np.loadtxt('../experimental-data/pmt-characteristics/ir-amplification-invcdf.dat')
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
