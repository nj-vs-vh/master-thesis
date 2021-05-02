"""
randomized_ir: classes for modelling randomized impulse response and using it for statistical deconvolution
"""


import math
import numpy as np
from matplotlib import pyplot as plt
import numdifftools as nd

from numba import njit

from scipy.interpolate import interp1d
from numpy.linalg import pinv
from math import pi

from scipy.stats import multivariate_normal

from functools import partial, lru_cache

from tqdm import tqdm_notebook
from typing import Union, Callable, Optional, Any
from nptyping import NDArray

import modules.utils as utils
from modules.ndepdf import ndepdf


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

        L_true = ir_x[-1]  # \tilde{L} in text
        self.L = math.floor(L_true)
        # when L_true is very close to an integer we can drop L by 1, e.g L_true = 5.0 is the same as L_true = 4.9999
        if L_true - self.L < 1e-6:
            self.L -= 1

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


class RandomizedIrEffect:
    def __init__(
        self,
        rir: RandomizedIr,
        N: int,
        samplesize: int = 100000,
        inbin_invcdf: Callable[[float], float] = None,
    ):
        """Statistical representation of a RandomizedIr's effect in linear system.

        Args:
            rir (RandomizedIr): RandomizedIr for calculation.
            N (int): Number of bins we're operating in.
            samplesize (int, optional): Amount of sample functions for each IR bin. Defaults to 100000.
            inbin_invcdf (Callable[[float], float], optional): See RanodmizedIr's convolve_with_deltas method.
        """
        self.rir = rir
        self.N = N
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
        self.C_mat = self.calculate_C_mat()
        self.Xi_mat = self.calculate_Xi_mat()
        self.mvn_mu_Sigma_as_func_of_n_vec = self.get_mvn_mu_Sigma_from_n_vec()

    def explore(self):
        print(f"L={self.L} and N={self.N}")
        print("RIR effects from photon in the bin #1 (t in [0; 1]):")
        print("t\teffect")
        for i, (mean, dispersion) in enumerate(zip(self.ir_sample_mean, self.ir_sample_D)):
            print(f"{i + 1}\t{mean:.2f} +/- {np.sqrt(dispersion):.2f}")
        print()
        print(f"C matrix used to calculate mean-vector for a given n vector (with cut edge effects):\n{self.C_mat}")
        print()
        print(f"Xi matrix used to calculate covariance matrix for a given n vector:\n{self.Xi_mat}")

    @property
    def L(self) -> int:
        return self.rir.L

    def calculate_C_mat(self) -> NDArray[(Any, Any), float]:
        """Matrix C used to calculate mean-vector for a given \\vec{n}

        See \\ref{eq:mean-vector-calculation}"""
        L = self.L
        N = self.N
        c_vec = self.ir_sample_mean
        C = np.zeros((N + L, N))
        for i in range(N):
            C[i : i + L + 1, i] = c_vec  # noqa
        return utils.slice_edge_effects(C, L, N)

    def calculate_Xi_mat(self) -> NDArray[(Any, Any), float]:
        """Matrix used to calculate covariance matrix for a given \\vec{n}

        See \\ref{eq:Xi-matrix-for-Sigma-calculation}"""

        def xi(lag, Delta):
            # np.cov returns covariation _matrix_, but we need only cov(x, y) which is at [0, 1] and [1, 0] cells
            return np.cov(self.ir_samples[[lag, lag + Delta], :])[0, 1]

        L = self.L
        Xi_mat = np.zeros((L+1, L+1))
        for i in range(L+1):
            for j in range(i, L+1):
                Xi_mat[i, j] = xi(L - j, i)

        return Xi_mat

    def estimate_n_vec(self, s_vec: NDArray[(Any,), float]) -> NDArray[(Any,), float]:
        """LLS-based estimation of n vector using Moore-Penrose pseudoinverse matrix.

        See \\subsection{Грубая оценка методом наименьших квадратов}
        """
        s_vec = utils.slice_edge_effects(s_vec, self.L, self.N)
        return pinv(self.C_mat) @ s_vec

    def get_mvn_mu_Sigma_from_n_vec(self):
        L = self.L
        N = self.N
        C_mat = self.C_mat
        Xi_mat = self.Xi_mat

        # extracting it as njitted function
        @njit
        def mu_Sigma(n_vec):
            # mean vector calculation
            mu = C_mat @ n_vec
            # covariance matrix calculation
            Sigma = np.zeros((N - L, N - L))
            for i_cut in range(N - L):
                i = i_cut + L + 1
                # see \\ref{eq:Xi-matrix-for-Sigma-calculation}
                Sigma_i_vec = Xi_mat @ n_vec[i_cut : i]  # noqa
                # cutting end of Sigma_i vec when adding it at the end of the matrix (no effect on the inside-region)
                Sigma_i_vec = Sigma_i_vec[:N - L - i_cut]
                Sigma[i_cut, i_cut:i] = Sigma_i_vec
                Sigma[i_cut:i, i_cut] = Sigma_i_vec
            return mu, Sigma
        return mu_Sigma

    def get_multivariate_normal_S_vec(self, n_vec: NDArray[(Any,), float]):
        mu, Sigma = self.mvn_mu_Sigma_as_func_of_n_vec(n_vec)
        return multivariate_normal(mean=mu, cov=Sigma)

    def get_loglikelihood_mvn(self, s_vec: NDArray[(Any,), float]) -> Callable[[NDArray[(Any,), float]], float]:
        """Loglikelihood function for a given signal s_vec assuming independent and normal distributions of S_j"""
        s_vec = utils.slice_edge_effects(s_vec, self.L, self.N)

        def loglikelihood_mvn(n_vec: NDArray[(Any,), float]) -> float:
            if np.any(n_vec < 0):  # guard for impossible values
                return -np.inf
            return self.get_multivariate_normal_S_vec(n_vec).logpdf(s_vec)

        return loglikelihood_mvn

    def sample_S_vec(
        self, n_vec: NDArray[(Any,), float], n_samples: int, progress: bool = False
    ) -> NDArray[(Any, Any), float]:
        """Generate sample of sigmal realizations for a given input n_vec"""
        n_vec = n_vec.round().astype(int)
        N = n_vec.size

        S_length = N + self.L

        n_ir_samples = self.ir_samples.shape[1]
        s_sample = np.zeros((S_length, n_samples))

        sample_indices = range(n_samples)
        if progress:
            sample_indices = tqdm_notebook(sample_indices)

        for s_sample_index in sample_indices:
            ir_sample_choice = self.ir_samples[:, np.random.choice(np.arange(n_ir_samples), size=(n_vec.sum(),))]
            s_modelled = np.zeros((S_length,))
            ir_sample_index_offset = 0
            for i, n in enumerate(n_vec):
                s_modelled[i : i + self.L + 1] += np.sum(  # noqa
                    ir_sample_choice[:, ir_sample_index_offset : ir_sample_index_offset + n], axis=1  # noqa
                )
                ir_sample_index_offset += n
            s_sample[:, s_sample_index] = s_modelled
        return s_sample

    def get_loglikelihood_monte_carlo(self, s_vec: NDArray[(Any,), float]) -> Callable[[NDArray[(Any,), float]], float]:
        L = self.L
        N = s_vec.size - L
        center_s_vec = s_vec[L:N]

        def loglikelihood_monte_carlo(n_vec: NDArray[(Any,), float], progress: bool = False) -> float:
            s_sample = self.sample_S_vec(n_vec, 10 ** 6, progress=progress)
            s_sample = utils.slice_edge_effects(s_sample, L, N)
            return np.log(ndepdf(s_sample, center_s_vec, bins=5, check_bin_count=True))

        return loglikelihood_monte_carlo

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
            return np.mean(np.exp(t * self.ir_samples[lag, :]))

    @lru_cache(maxsize=100000)
    def mgf_moment(self, i: int, n: int, lag: int) -> float:
        """Compute ith moment of C(n, lag) using MGF"""

        derivative = nd.Derivative(partial(self.mgf, n=n, lag=lag), n=i, full_output=True)
        moment, info = derivative(0)
        return moment

    # diagnostic plots

    def plot_samples(self, max_lag: int = None):
        if max_lag is None:
            max_lag = self.ir_samples.shape[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        for lag, sample in enumerate(self.ir_samples):
            if lag > max_lag:
                break
            _, _, histogram = ax.hist(sample, label=f"lag={lag}", alpha=0.3, density=True)
            mu = self.mgf_moment(1, 1, lag)
            sigma = np.sqrt(self.mgf_moment(2, 1, lag) - mu ** 2)
            pdf_t = np.linspace(np.min(sample), np.max(sample), 100)
            pdf = (1 / (sigma * np.sqrt(2 * pi))) * np.exp(-0.5 * np.power((pdf_t - mu) / (sigma), 2))
            ax.plot(pdf_t, pdf, color=histogram[0]._facecolor, alpha=1)

        ax.legend()
        # ax.set_yscale('log')
        plt.show()

    def plot_moments(self, n: int, lag: int):
        fig, ax = plt.subplots(figsize=(8, 7))

        sample_1 = self.ir_samples[lag, :]
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
    # testing MC vs normdist loglikelihood

    from random import random
    from modules.utils import generate_poissonian_ns

    L_true = 3.5
    ir_x = np.linspace(0, L_true, int(L_true * 100))
    ir_y = np.exp(-ir_x)
    rir = RandomizedIr(ir_x, ir_y, factor=lambda: 0.5 + random() * 0.5)

    N = 10
    n_vec_mean = 20
    n_vec = generate_poissonian_ns(n_vec_mean, N)

    s_vec = rir.convolve_with_n_vec(n_vec)

    stats = RandomizedIrEffect(rir, samplesize=10 ** 5)

    n_vec_estimate = stats.estimate_n_vec(s_vec)

    loglike = stats.get_loglikelihood_mvn(s_vec)
    loglike_mc = stats.get_loglikelihood_monte_carlo(s_vec)
    print(loglike(n_vec_estimate))
    print(loglike_mc(n_vec_estimate))
