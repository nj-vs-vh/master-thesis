import math
import numpy as np
from matplotlib import pyplot as plt

from scipy.interpolate import interp1d

from typing import Union, Callable
from nptyping import NDArray


rng = np.random.default_rng()


class RandomizedIR:

    def __init__(self, ir_x: NDArray, ir_y: Union[NDArray, Callable[[], NDArray]], binsize: float = 1.0):
        """Create randomized impulse response

        Args:
            ir_x (NDArray): impulse response sampling times. Assumed to be in bins unless `binsize` is specified
            ir_y (NDArray or callable outputting NDArray): impulse response's actual values, must be the same
                                                           length as `ir_x`. Units are opaque and propagate to signal.
            binsize (float, optional): If `ir_x` is in some units other than bins, specify this for conversion.
                                       Defaults to 1.0.
        """
        if not isinstance(ir_x, NDArray) or ir_x.ndim != 1:
            raise ValueError("ir_x must be one-dimensional numpy array")
        self.ir_x = ir_x / binsize

        if self.ir_x[1] - self.ir_x[0] >= 1:
            raise ValueError("ir_x seem too spread out, haven't you forgot to set binsize?")

        self.nbins = math.ceil(ir_x[-1])

        self.base_ir_generator: Callable[[], NDArray] = ir_y if callable(ir_y) else lambda: ir_y
        ir_y_realization = self.base_ir_generator()
        if not isinstance(ir_y_realization, NDArray) or ir_y_realization.shape != ir_x.shape:
            raise ValueError("ir_y must be or return numpy array of the same shape as ir_x")

    def __call__(self, x: NDArray) -> NDArray:
        """Return realization of randomized IR at given points

        Args:
            x (NDArray): query points for IR

        Returns:
            NDArray: realization of randomized IR
        """
        return interp1d(
            self.ir_x, self.base_ir_generator(), kind='linear', copy=False, fill_value=0, bounds_error=False
        )(x)

    def plot_realization(self):
        ax = plt.subplot(111)
        ax.plot(self.ir_x, self.base_ir_generator())
        plt.show()

    def convolve_with_deltas(
        self, delta_ns: NDArray, inbin_invcdf: Callable[[float], float] = lambda x: x, debug_inbin_times: bool = False
    ) -> NDArray:
        """Given a number of delta function in each bin, return their convolution with the RIR. Delta times are assumed
        to be equally distributed in each bin.

        Args:
            delta_ns (NDArray): number of delta functions in each bin
            inbin_invcdf (Callable[[float], float], optional): inverse CDF of delta time distribution inside one bin.
                                                               Must have the followind properties: inbin_invcdf(0) = 0,
                                                               inbin_invcdf(1) = 1, monotonous growth.
                                                               Default is inbin_invcdf(x) = x, i.e. uniform.
            debug_inbin_times (bool, optional): if True, print mean and std of inbin time distribution. Useful for
                                                debugging inbin_invcdf. Default is False

        Returns:
            NDArray: convoluted signal
        """
        if not isinstance(delta_ns, NDArray) or delta_ns.ndim != 1 or delta_ns.dtype != int:
            raise ValueError("delta_ns must be one dimensional numpy array of integers")

        if debug_inbin_times:
            n_test_sample = 10000
            sample = np.vectorize(inbin_invcdf)(rng.random(size=(n_test_sample,)))
            print(f"Inbin times are distributed with mean = {sample.mean():.3f} and sigma={sample.std():.3f}")

        N_bins = delta_ns.size

        convoluted_bins = N_bins + self.nbins
        out_x = np.arange(0, convoluted_bins) + 1  # lag is because deltas from bin 0 (i.e. [0 1]) only appear in bin 1
        out_y = np.zeros((convoluted_bins,))

        ir_x_whole_bins = np.arange(0, self.nbins, step=1.0)

        for i, n_i in enumerate(delta_ns):
            for _ in range(n_i):
                inbin_time = inbin_invcdf(rng.random())
                out_y[i:i+self.nbins] += self(ir_x_whole_bins + (1 - inbin_time))
        return out_x, out_y

    # def 


if __name__ == "__main__":
    from utils import generate_poissonian_ns

    n = generate_poissonian_ns(50, 50)

    ir_x = np.arange(0, 1000) / 50
    ir_y = np.zeros_like(ir_x) + 0.1

    rir = RandomizedIR(ir_x, ir_y)

    x, y = rir.convolve_with_deltas(n)

    ax = plt.subplot(111)
    for i in range(len(n)):
        ax.axvspan(i, i+1, facecolor=([0, 0, 0] if i % 2 == 0 else [0.3, 0.3, 0.3]), alpha=0.1, edgecolor=None)
    ax.axhline(0, color='black')
    ax.step(x, y, '.-', where='post')
    plt.show()
