import math
import numpy as np
from matplotlib import pyplot as plt
import numdifftools as nd

from functools import partial
from scipy.interpolate import interp1d
from numpy.linalg import pinv

from typing import Union, Callable, Optional, Any
from nptyping import NDArray


rng = np.random.default_rng()


class RandomizedIr:
    def __init__(
        self,
        ir_x: NDArray,
        ir_y: Union[NDArray, Callable[[], NDArray]],
        factor: Optional[Callable[[], float]] = None,
        binsize: float = 1.0,
    ):
        """Create randomized impulse response

        Args:
            ir_x (NDArray): impulse response sampling times. Assumed to be in bins unless `binsize` is specified
            ir_y (NDArray or callable outputting NDArray): impulse response's actual values, must be the same
                                                           length as `ir_x`. Units are opaque and propagate to signal.
            factor (callable outputting float): random factor of IR. If not specified defaults to constant 1.
            binsize (float, optional): If `ir_x` is in some units other than bins, specify this for conversion.
                                       Defaults to 1.0.
        """
        if not isinstance(ir_x, NDArray) or ir_x.ndim != 1:
            raise ValueError("ir_x must be one-dimensional numpy array")
        self.ir_x = ir_x / binsize

        if self.ir_x[1] - self.ir_x[0] >= 1:
            raise ValueError("ir_x seem too spread out, haven't you forgot to set binsize?")

        self.L = math.floor(ir_x[-1])

        self.base_ir_generator: Callable[[], NDArray] = ir_y if callable(ir_y) else lambda: ir_y
        ir_y_realization = self.base_ir_generator()
        if not isinstance(ir_y_realization, NDArray) or ir_y_realization.shape != ir_x.shape:
            raise ValueError("ir_y must be or return numpy array of the same shape as ir_x")
        self.factor_generator: Callable[[], float] = factor or (lambda: 1)

    def _realization(self):
        return self.factor_generator() * self.base_ir_generator()

    def __call__(self, x: NDArray) -> NDArray:
        """Evaluate randomized IR (i.e. its random realization) at given points

        Args:
            x (NDArray): query points for IR

        Returns:
            NDArray: realization of randomized IR
        """
        return interp1d(
            self.ir_x,
            self._realization(),
            kind="linear",
            copy=False,
            fill_value=0,
            bounds_error=False,
        )(x)

    def plot_realizations(self, count: int = 10, ax: plt.Axes = None):
        ax = ax or plt.subplot(111)
        for i in range(count):
            ax.plot(self.ir_x, self._realization())
        plt.show()

    def convolve_with_n_vec(
        self,
        n_vec: NDArray,
        inbin_invcdf: Callable[[float], float] = None,
        debug_inbin_times: bool = False,
    ) -> NDArray:
        """Given a number of delta function in each bin, return their convolution with the RIR. Delta times are assumed
        to be equally distributed in each bin.

        Args:
            n_vec (NDArray): number of delta functions in each bin
            inbin_invcdf (Callable[[float], float], optional): inverse CDF of delta time distribution inside one bin.
                                                               Must have the followind properties: inbin_invcdf(0) = 0,
                                                               inbin_invcdf(1) = 1, monotonous growth.
                                                               Defaults to None, interpreted as uniform distribution.
            debug_inbin_times (bool, optional): if True, print mean and std of inbin time distribution. Useful for
                                                debugging inbin_invcdf. Defaults to False.

        Returns:
            NDArray: convoluted signal
        """
        if not isinstance(n_vec, NDArray) or n_vec.ndim != 1 or n_vec.dtype != int:
            raise ValueError("n_vec must be one dimensional numpy array of integers")

        if debug_inbin_times and inbin_invcdf is not None:
            n_test_sample = 10000
            sample = np.vectorize(inbin_invcdf)(rng.random(size=(n_test_sample,)))
            print(f"Inbin times are distributed with mean = {sample.mean():.3f} and sigma={sample.std():.3f}")

        N = n_vec.size

        convoluted_bins = N + self.L
        out_y = np.zeros((convoluted_bins,))

        ir_x_whole_bins = np.arange(0, self.L, step=1.0)

        for i, n_i in enumerate(n_vec):
            for _ in range(n_i):
                uniform_sample = rng.random()
                inbin_time = inbin_invcdf(uniform_sample) if inbin_invcdf else uniform_sample
                out_y[i : (i + self.L)] += self(ir_x_whole_bins + (1 - inbin_time))  # noqa
        return out_y


class RandomizedIrStats:
    def __init__(
        self,
        rir: RandomizedIr,
        samplesize: int = 100000,
        inbin_invcdf: Callable[[float], float] = None,
    ):
        """Calculate and store statistical representation of randomized IR

        Args:
            rir (RandomizedIr): RandIR for calculation
            samplesize (int, optional): amount of sample delta functions for each IR bin. Defaults to 100000.
            inbin_invcdf (Callable[[float], float], optional): See RanodmizedIr's convolve_with_deltas method.
        """
        self.rir = rir
        self.ir_samples = np.zeros((self.L, samplesize))

        if inbin_invcdf is not None:
            inbin_invcdf = np.vectorize(inbin_invcdf)
        uniform_sample = rng.random(size=(samplesize,))
        inbin_times_sample = inbin_invcdf(uniform_sample) if inbin_invcdf else uniform_sample
        bins = range(self.L)
        for sample_i, inbin_t in enumerate(inbin_times_sample):
            rir_realization = rir(bins + inbin_t)
            self.ir_samples[:, sample_i] = rir_realization

        # mean of C(1, l)
        self.c_vec = np.mean(self.ir_samples, axis=1)

    @property
    def L(self) -> int:
        return self.rir.L

    def estimate_n_vec(self, S_vec: NDArray[(Any,), float]) -> NDArray[(Any,), float]:
        """LLS-based estimation of n vector. See \\subsection{Грубая оценка методом наименьших квадратов}"""
        L = self.L
        N = S_vec.size - L
        C = np.zeros((N + L, N))  # see \label{eq:mean_matrix}
        for i in range(N):
            C[i:i+L, i] = self.c_vec
        return pinv(C) @ S_vec  # LLS solution with Moore-Penrose pseudoinverse matrix

    # MGF calculation methods for MCMC

    def mgf(self, t: float, n: int, lag: int) -> float:
        """Calculate MGF (moment generating function) at argument t for contrbution of deltas after lag bins

        Args:
            t (float): mgf internal argument
            n (int): number of delta functions in bin
            lag (int): contribution lag for bin. Minimum value is 1, because bins are numbered at lower bound.

        Returns:
            NDArray: MGF(t) value
        """
        single_delta_mgf = np.mean(np.exp(t * self.ir_samples[lag - 1, :]))
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

    # diagnostic plots

    def plot_samples(self):
        fig, ax = plt.subplots(figsize=(8, 7))
        for ibin, sample in enumerate(self.ir_samples):
            ax.hist(sample, label=f"values at lag {ibin + 1}", alpha=0.3)
        ax.legend()
        plt.show()

    def plot_moments(self, n: int, lag: int):
        fig, ax = plt.subplots(figsize=(8, 7))

        sample_1 = self.ir_samples[lag - 1, :]
        sample_n = np.zeros_like(sample_1)
        for _ in range(n):
            sample_n += rng.permutation(sample_1)

        ax.hist(sample_n, alpha=0.5, label=f"sample for {n} delta(s) at lag {lag}")

        mgf_mean = self.mgf_moment(1, n, lag)
        mgf_std = np.sqrt(self.mgf_moment(2, n, lag) - mgf_mean ** 2)
        ax.axvline(mgf_mean, color="red", label="MGF mean")
        ax.axvspan(
            mgf_mean - mgf_std,
            mgf_mean + mgf_std,
            color="red",
            alpha=0.3,
            label="MGF sigma",
        )
        ax.legend()
        plt.show()

    def plot_mgf(self, tmax: float, n: int = 1, lag: int = 1):
        """MGF is being differentiated at 0 -- this plot helps assess correctess of the numerical derivative"""
        t = np.linspace(0, tmax, 100)
        mgf = np.zeros_like(t)
        for i, t_i in enumerate(t):
            mgf[i] = self.mgf(t_i, n, lag)

        first_derivative = self.mgf_moment(1, n, lag)
        second_derivative = self.mgf_moment(1, n, lag)
        linear_approx = 1 + first_derivative * t
        quadratic_approx = 1 + first_derivative * t + second_derivative * np.power(t, 2) / 2

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(t, mgf, label='MGF')
        ax.plot(t, linear_approx, '--', label='Linear (mean) approx')
        ax.plot(t, quadratic_approx, '--', label='Quadratic (mean and std) approx')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    from random import random
    from utils import generate_poissonian_ns

    L_true = 3.5
    ir_x = np.linspace(0, L_true, int(L_true * 100))
    ir_y = np.exp(- ir_x)
    rir = RandomizedIr(ir_x, ir_y, factor=lambda: 0.5 + random()*0.5)

    N = 5
    n_vec_mean = 15
    n_vec = generate_poissonian_ns(n_vec_mean, N)

    S_vec = rir.convolve_with_n_vec(n_vec)

    stats = RandomizedIrStats(rir, samplesize=10**6)

    n_vec_estimate = stats.estimate_n_vec(S_vec)
