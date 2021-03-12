import math
import numpy as np
from matplotlib import pyplot as plt
import numdifftools as nd

from functools import partial
from scipy.interpolate import interp1d

from typing import Union, Callable
from nptyping import NDArray


rng = np.random.default_rng()


class RandomizedIr:
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

    def _realization(self):
        # later other add coeffs and shifts
        return self.base_ir_generator()

    def __call__(self, x: NDArray) -> NDArray:
        """Evaluate randomized IR (i.e. its random realization) at given points

        Args:
            x (NDArray): query points for IR

        Returns:
            NDArray: realization of randomized IR
        """
        return interp1d(
            self.ir_x, self._realization(), kind='linear', copy=False, fill_value=0, bounds_error=False
        )(x)

    def plot_realization(self, ax: plt.Axes = None):
        ax = ax or plt.subplot(111)
        ax.plot(self.ir_x, self._realization())
        plt.show()

    def convolve_with_deltas(
        self, delta_ns: NDArray, inbin_invcdf: Callable[[float], float] = None, debug_inbin_times: bool = False
    ) -> NDArray:
        """Given a number of delta function in each bin, return their convolution with the RIR. Delta times are assumed
        to be equally distributed in each bin.

        Args:
            delta_ns (NDArray): number of delta functions in each bin
            inbin_invcdf (Callable[[float], float], optional): inverse CDF of delta time distribution inside one bin.
                                                               Must have the followind properties: inbin_invcdf(0) = 0,
                                                               inbin_invcdf(1) = 1, monotonous growth.
                                                               Defaults to None, interpreted as uniform distribution.
            debug_inbin_times (bool, optional): if True, print mean and std of inbin time distribution. Useful for
                                                debugging inbin_invcdf. Defaults to False.

        Returns:
            NDArray: convoluted signal
        """
        if not isinstance(delta_ns, NDArray) or delta_ns.ndim != 1 or delta_ns.dtype != int:
            raise ValueError("delta_ns must be one dimensional numpy array of integers")

        if debug_inbin_times and inbin_invcdf is not None:
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
                uniform_sample = rng.random()
                inbin_time = inbin_invcdf(uniform_sample) if inbin_invcdf else uniform_sample
                out_y[i:i+self.nbins] += self(ir_x_whole_bins + (1 - inbin_time))
        return out_x, out_y


class RandomizedIrStats:
    def __init__(self, rir: RandomizedIr, samplesize: int = 100000, inbin_invcdf: Callable[[float], float] = None):
        """Calculate and store statistical representation of randomized IR

        Args:
            rir (RandomizedIr): RandIR for calculation
            samplesize (int, optional): amount of sample delta functions for each IR bin. Defaults to 100000.
            inbin_invcdf (Callable[[float], float], optional): See RanodmizedIr's convolve_with_deltas method.
        """
        self.rir = rir
        self.samples = np.zeros((rir.nbins, samplesize))

        if inbin_invcdf is not None:
            inbin_invcdf = np.vectorize(inbin_invcdf)
        for ibin in range(rir.nbins):
            uniform_sample = rng.random(size=(1, samplesize))
            inbin_times_sample = inbin_invcdf(uniform_sample) if inbin_invcdf else uniform_sample
            self.samples[ibin, :] = rir(ibin + inbin_times_sample)
        self.sample_means = np.mean(self.samples, axis=1)

    @property
    def nbins(self) -> int:
        return self.rir.nbins

    def plot_samples(self):
        fig, ax = plt.subplots(figsize=(8, 7))
        for ibin, sample in enumerate(self.samples):
            ax.hist(sample, label=f"values at lag {ibin + 1}", alpha=0.3)
        ax.legend()
        plt.show()

    def mgf(self, t: float, n: int, lag: int) -> float:
        """Calculate MGF (moment generating function) at argument t for contrbution of deltas after lag bins

        Args:
            t (float): mgf internal argument
            n (int): number of delta functions in bin
            lag (int): contribution lag for bin. Minimum value is 1, because bins are numbered at lower bound.

        Returns:
            NDArray: MGF(t) value
        """
        single_delta_mgf = np.mean(np.exp(t * self.samples[lag-1, :]))
        return np.power(single_delta_mgf, n)

    def mgf_moment(self, i: int, n: int, lag: int) -> float:
        """Compute ith moment of contrbution of deltas after lag bins using MGF

        Args:
            i (int): [description]
            n (int): [description]
            lag (int): [description]

        Returns:
            float: [description]
        """
        derivative = nd.Derivative(partial(self.mgf, n=n, lag=lag), n=i, full_output=True)
        val, info = derivative(0)
        return val

    def plot_moments(self, n: int, lag: int):
        fig, ax = plt.subplots(figsize=(8, 7))

        sample_1 = self.samples[lag-1, :]
        sample_n = np.zeros_like(sample_1)
        for _ in range(n):
            sample_n += rng.permutation(sample_1)

        ax.hist(sample_n, alpha=0.5, label=f'sample for {n} delta(s) at lag {lag}')

        mgf_mean = self.mgf_moment(1, n, lag)
        mgf_std = np.sqrt(self.mgf_moment(2, n, lag) - mgf_mean ** 2)
        ax.axvline(mgf_mean, color='red', label='MGF mean')
        ax.axvspan(mgf_mean - mgf_std, mgf_mean + mgf_std, color='red', alpha=0.3, label='MGF sigma')
        ax.legend()
        plt.show()

    def plot_mgf(self, tmax: float, n: int = 1, lag: int = 1):
        t = np.linspace(0, tmax, 100)
        mgf = np.zeros_like(t)
        for i, t_i in enumerate(t):
            mgf[i] = self.mgf(t_i, n, lag)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(t, mgf)
        plt.show()
