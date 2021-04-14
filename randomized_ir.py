import math
import numpy as np
from matplotlib import pyplot as plt
import numdifftools as nd

from scipy.interpolate import interp1d
from numpy.linalg import pinv
from math import pi

from functools import partial, lru_cache

from numba import njit

from typing import Union, Callable, Optional, Any
from nptyping import NDArray

import utils


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
            raise ValueError("ir_x must be a one-dimensional numpy array")
        if not ir_x[0] == 0:
            raise ValueError("ir_x must start at zero")
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

        convoluted_pts_count = N + self.L
        out_y = np.zeros((convoluted_pts_count,))

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
        sample_t = range(self.L + 1)
        self.ir_samples = np.zeros((self.L + 1, samplesize))

        if inbin_invcdf is not None:
            inbin_invcdf = np.vectorize(inbin_invcdf)
        uniform_sample = rng.random(size=(samplesize,))
        inbin_times_sample = inbin_invcdf(uniform_sample) if inbin_invcdf else uniform_sample
        for sample_i, inbin_t in enumerate(inbin_times_sample):
            rir_realization = rir(sample_t + inbin_t)
            self.ir_samples[:, sample_i] = rir_realization

        # means and dispersions of C(1, l)
        self.ir_sample_mean = np.mean(self.ir_samples, axis=1)
        self.ir_sample_D = np.power(np.std(self.ir_samples, axis=1), 2)

    @property
    def L(self) -> int:
        return self.rir.L

    def estimate_n_vec(self, s_vec: NDArray[(Any,), float]) -> NDArray[(Any,), float]:
        """LLS-based estimation of n vector using Moore-Penrose pseudoinverse matrix.

        See \\subsection{Грубая оценка методом наименьших квадратов}
        """
        L = self.L
        N = s_vec.size - L

        C = np.zeros((N + L + 1, N))  # see \label{eq:mean_matrix}
        c_vec = self.ir_sample_mean
        for i in range(N):
            C[i : i + L + 1, i] = c_vec  # noqa

        C = utils.slice_edge_effects(C, L, N)
        s_vec = utils.slice_edge_effects(s_vec, L, N)
        return pinv(C) @ s_vec

    def get_loglikelihood_normdist(self, s_vec: NDArray[(Any,), float]) -> Callable[[NDArray[(Any,), float]], float]:
        """Loglikelihood assuming normal distribution of S_j"""

        L = self.L
        N = s_vec.size - L
        # for nested function to be numbifiable
        ir_sample_mean = self.ir_sample_mean
        ir_sample_D = self.ir_sample_D

        def propto_logpdf(x: float, mu: float, sigma: float) -> float:
            return -(((x - mu) / (1.41421356237 * sigma)) ** 2)

        @njit
        def loglikelihood(n_vec: NDArray[(Any,), float]) -> float:
            if np.any(n_vec < 0):  # guard for impossible values
                return -np.inf
            logL = 0
            for j, s_j in enumerate(s_vec):
                j += 1  # from indexing array (0-based) to indexing time points (1-based)
                if j <= L or j > N:  # cutting off signal edges
                    continue
                Es_j = 0
                Ds_j = 0
                for lag in range(L + 1):
                    i = j - lag
                    i -= 1  # from indexing bins (1-based) to indexing array (0-based)
                    Es_j += n_vec[i] * ir_sample_mean[lag]
                    Ds_j += n_vec[i] * ir_sample_D[lag]
                Sigma_s_j = np.sqrt(Ds_j)
                logL_subtract = ((s_j - Es_j) / (1.41421356237 * Sigma_s_j)) ** 2
                if np.isnan(logL_subtract):
                    return - np.inf
                logL -= logL_subtract
            return logL

        return loglikelihood

    # def get_loglikelihood_monte_carlo(self, s_vec: NDArray[(Any,), float]) -> Callable[[NDArray[(Any,), float]], float]:
        
    #     def loglikelihood(n_vec: NDArray[(Any,), float]) -> float:
    #         self.

    #         return logL

    #     return loglikelihood

    # MGF calculation methods

    def mgf(self, t: float, n: int, lag: int) -> float:
        """Calculate MGF (moment generating function) at argument t for C(n, l) (contrbution of n deltas after l bins)

        Args:
            t (float): mgf internal argument
            n (int): number of delta functions in bin
            lag (int): contribution lag for bin. Minimum value is 1, because bins are numbered at lower bound.

        Returns:
            NDArray: MGF(t) value
        """
        return np.power(self._mgf_n_1(t, lag), n)

    def _mgf_n_1(self, t: float, lag: int) -> float:
        """mgf for n = 1"""
        MGF_EPSILON = 1e-7

        if abs(t) < MGF_EPSILON:
            return 1
        else:
            return np.mean(np.exp(t * self.ir_samples[lag - 1, :]))

    @lru_cache(maxsize=100000)
    def mgf_moment(self, i: int, n: int, lag: int) -> float:
        """Compute ith moment of C(n, lag) using MGF

        Args:
            i (int): [description]
            n (int): [description]
            lag (int): [description]

        Returns:
            float: [description]
        """

        derivative = nd.Derivative(partial(self.mgf, n=n, lag=lag), n=i, full_output=True)
        moment, info = derivative(0)
        return moment

    # diagnostic plots

    def plot_samples(self, max_lag: int = None):
        if not max_lag:
            max_lag = self.ir_samples.shape[0] + 1
        fig, ax = plt.subplots(figsize=(8, 7))
        for lag_0_based, sample in enumerate(self.ir_samples):
            lag = lag_0_based + 1
            if lag > max_lag:
                break
            _, _, histogram = ax.hist(sample, label=f"lag={lag}", alpha=0.3, density=True)
            mu = self.mgf_moment(1, 1, lag)
            sigma = np.sqrt(self.mgf_moment(2, 1, lag) - mu ** 2)
            pdf_t = np.linspace(np.min(sample), np.max(sample), 100)
            pdf = (1 / (sigma * np.sqrt(2 * pi))) * np.exp(-0.5 * np.power((pdf_t - mu) / (sigma), 2))
            ax.plot(pdf_t, pdf, color=histogram[0]._facecolor, alpha=1)

        ax.legend()
        ax.set_yscale('log')
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
        t = np.linspace(-tmax, tmax, 100)
        mgf = np.zeros_like(t)
        for i, t_i in enumerate(t):
            mgf[i] = self.mgf(t_i, n, lag)

        first_derivative = self.mgf_moment(1, n, lag)
        second_derivative = self.mgf_moment(2, n, lag)
        third_derivativa = self.mgf_moment(3, n, lag)
        linear_approx = 1 + first_derivative * t
        quadratic_approx = linear_approx + second_derivative * np.power(t, 2) / 2
        cubic_approx = quadratic_approx + third_derivativa * np.power(t, 3) / 6

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(t, mgf - linear_approx, label='MGF')
        for approx, desription in [
            # (linear_approx, 'linear (mean)'),
            (quadratic_approx, 'quadratic (mean and std)'),
            (cubic_approx, 'cubic (mean, std and asymm)'),
        ]:
            ax.plot(t, approx - linear_approx, '--', label=f'$\\Delta$ for {desription} approx')
        # ax.set_yscale('log')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    from random import random
    from utils import generate_poissonian_ns

    L_true = 3.5
    ir_x = np.linspace(0, L_true, int(L_true * 100))
    ir_y = np.exp(-ir_x)
    rir = RandomizedIr(ir_x, ir_y, factor=lambda: 0.5 + random() * 0.5)

    N = 5
    n_vec_mean = 15
    n_vec = generate_poissonian_ns(n_vec_mean, N)

    s_vec = rir.convolve_with_n_vec(n_vec)

    stats = RandomizedIrStats(rir, samplesize=10 ** 6)

    n_vec_estimate = stats.estimate_n_vec(s_vec)
