"""
Experimental data reading and processing
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from typing import Any, Tuple, Optional, Literal
from nptyping import NDArray

from modules.experiment.rir import get_rireffs
from modules.randomized_ir import RandomizedIr
import modules.mcmc as mcmc
import modules.signal_reconstruction as sigrec
import modules.eas_reconstruction as eas
from modules.utils import apply_mask
from modules.experiment.fov import PmtFov


CUR_DIR = Path(__file__).parent
EXP_DATA_DIR = CUR_DIR / '../../experimental-data'


N_CHANNELS = 109
SIGNAL_LENGTH = Any

# custom typing
FloatPerChannel = NDArray[(N_CHANNELS,), float]
SignalPerChannel = NDArray[(N_CHANNELS, SIGNAL_LENGTH), float]
Signal = NDArray[(SIGNAL_LENGTH,), float]


class Event:
    """Experimental event with all its aspects (frame, mean current, calibration) stored together"""

    # pre-loaded mean currents for all events, converted to code units
    _mean_currents_by_event_id: FloatPerChannel = None

    def __init__(self, event_id):
        self.id_ = event_id
        if self._mean_currents_by_event_id is None:
            self._read_mean_currents()
        self._read_event_frame()
        self._read_calibration()
        self.estimated_frame_center = self._estimate_frame_center()

    @property
    def mean_currents(self) -> FloatPerChannel:
        try:
            return self._mean_currents_by_event_id[self.id_]
        except KeyError:
            raise ValueError(f"Mean current for event ID {self.id_} not found!")

    @property
    def mean_n_photoelectrons(self) -> FloatPerChannel:
        return self.mean_currents / self.C

    C_ref = 0.326  # code units * bin / photoelectron

    @property
    def C(self) -> FloatPerChannel:
        return self.C_ref / self.calibration

    TRIGGER_BIN = 428  # approximately

    def _estimate_frame_center(self):
        window = 300
        frame_sum = None
        for i_ch in range(N_CHANNELS):
            try:
                t, signal, _ = self.signal_in_channel(i_ch=i_ch, center_bin=self.TRIGGER_BIN, window=window)
                frame_sum = signal if frame_sum is None else frame_sum + signal
            except ValueError:
                continue
        frame_sum -= np.min(frame_sum)
        return np.sum(frame_sum * t) / np.sum(frame_sum)

    def signal_in_channel(
        self, i_ch: int, units: str = 'scaled', center_bin: Optional[int] = None, window: Optional[int] = None
    ) -> Tuple[Signal, Signal, float]:
        """Signal in a given channel and it's rounding error.

        'units' argument:
            'code' -- signal is returned as it would be recorded by an ideal ADC
            'scaled' -- signal ready for deconvolution with unit RIR

        Returns:
            t (Signal): times relative to start of the frame
            s (Signal): values of signal corresponding to t
            delta (float): ADC rounding error
        """
        self.validate_i_ch(i_ch)
        if units == 'scaled':
            C = self.C_ref / self.calibration[i_ch]
        elif units == 'code':
            C = 1
        else:
            raise ValueError(f"Unknown units '{units}'")
        center_bin = int(self.estimated_frame_center) if center_bin is None else center_bin
        window = 100 if window is None else window
        window_half = window // 2
        t = np.arange(center_bin - window_half, center_bin + window_half + 1)
        signal = self.mean_currents[i_ch] + self.frame[i_ch, t]
        adc_step = 1
        return t, signal / C, adc_step / C

    BROKEN_CHANNELS = {49, 78}

    def validate_i_ch(self, i_ch: int):
        if not 0 <= i_ch < N_CHANNELS:
            raise ValueError(f"Invalid channel no: {i_ch}, must be from 0 to {N_CHANNELS - 1}")
        if i_ch in self.BROKEN_CHANNELS:
            raise ValueError(f"Channel {i_ch} is broken")

    # Data reading methods #

    def _read_event_frame(self):
        try:
            with open(EXP_DATA_DIR / f'event-frames/{self.id_}.txt') as f:
                self.height: float = float(f.readline().strip())
                self.inclin = [float(v) for v in f.readline().strip().split()]
                f.readline()
                self.frame: SignalPerChannel = np.loadtxt(f).T[:N_CHANNELS, :]
        except FileNotFoundError:
            raise ValueError(f"Frame for event ID {self.id_} not found!")

    def _read_calibration(self):
        try:
            calibration_data = np.loadtxt(EXP_DATA_DIR / f'event-calibrations/{self.id_}.cal')
            self.calibration: FloatPerChannel = calibration_data[:N_CHANNELS, -1]
        except OSError:
            raise ValueError(f"Calibration for event ID {self.id_} not found!")

    @classmethod
    def _read_mean_currents(cls):
        cls._mean_currents_by_event_id = dict()
        with open(EXP_DATA_DIR / 'mean-currents.txt') as f:
            for line in f:
                eid, *currs = line.split()
                eid = int(eid)
                currs = [cls.muA_to_code_units(float(curr)) for curr in currs]
                cls._mean_currents_by_event_id[eid] = np.array(currs)[:N_CHANNELS]

    @staticmethod
    def muA_to_code_units(curr_muA: float) -> float:
        """See \\ref{eq:current-muA-to-code-units-conversion}"""
        return 0.38 * curr_muA


class EventProcessor:
    """Service class wrapping all signal processing routines"""

    TEMP_DATA_DIR = (CUR_DIR / '../../temp-data').resolve()
    DECONV_RESULTS_DIR = TEMP_DATA_DIR / 'deconvolution-results'
    SIGREC_DIR = TEMP_DATA_DIR / 'signal-reconstruction'
    EAS_PARAMS_DIR = TEMP_DATA_DIR / 'eas-geometry'

    def __init__(
        self,
        N: int = 45,
        verbosity: int = 1,
        preliminary_run_length: int = 10 ** 4,
        load_rir: bool = True,
        min_signal_significance: float = 4.0,
    ):
        self.verbosity = verbosity
        self.N = N
        self.preliminary_run_length = preliminary_run_length
        self.min_signal_significance = min_signal_significance
        if load_rir:
            self.ham_rireff, self.feu84_rireff = get_rireffs(N)

    def log(self, msg: str, min_verbosity: int = 1):
        if self.verbosity >= min_verbosity:
            print(msg)

    def __call__(self, event: Event):
        self.log('\n' + '=' * 25 + '\n' + f" Processing event #{event.id_}" + '\n' + '=' * 25 + '\n')
        for i_ch in range(N_CHANNELS):

            deconvolution_result_path = self._deconvolution_result_path(event.id_, i_ch)
            self.log(f'\nDeconvolving channel #{i_ch}...')
            if deconvolution_result_path.exists():
                deconv_data = np.load(deconvolution_result_path)
                signal_t = deconv_data['signal_t']
                sample = deconv_data['sample']
                self.log(f'Channel #{i_ch} deconvolution results loaded', 2)
            else:
                try:
                    self.log('No saved deconvolution results found, processing...', 2)
                    signal_t, signal, adc_step = event.signal_in_channel(i_ch, window=self.N)
                    rireff = self.ham_rireff if i_ch == 0 else self.feu84_rireff
                    sample = self._process_signal_with_rireff(signal, adc_step, rireff)

                    np.savez(deconvolution_result_path, signal_t=signal_t, sample=sample)
                    self.log(f'Channel #{i_ch} deconvolution results saved', 2)
                except ValueError:
                    continue
                except ValueError:
                    self.log(f"Error while processing channel {i_ch}! Moving on...")
                    continue

            sigrec_path = self._signal_reconstruction_path(event.id_, i_ch)
            if sigrec_path.exists():
                self.log(f'\nSignal reconstruction in channel #{i_ch} already done, loading.')
                # theta_sample = self.read_signal_reconstruction(event.id_, i_ch)
            else:
                try:
                    self.log(f'\nReconstructing signal in channel #{i_ch}...')
                    # cutting first and last L badly deconvoluted bins
                    signal_t = signal_t[self.feu84_rireff.L : -self.feu84_rireff.L]  # noqa
                    sample = sample[:, self.feu84_rireff.L : -self.feu84_rireff.L]  # noqa
                    self.log('No saved signal reconstruction, processing...', 2)

                    theta_sample = self._reconstruct_signal(
                        sample, signal_t, mean_n_phels=event.mean_n_photoelectrons[i_ch]
                    )

                    np.save(sigrec_path, theta_sample)
                    self.log(f'Channel #{i_ch} signal reconstruction saved', 2)
                except Exception as e:
                    self.log(f"Error while reconstructing signal in channel {i_ch}: {e}! Moving on...")
                    continue

        frame_sign_path = self._frame_significance_path(event.id_)
        if frame_sign_path.exists():
            pass
        else:
            self.log('\nCalculating signal significances...')
            significances = []
            channels = range(N_CHANNELS)
            if self.verbosity >= 1:
                channels = tqdm(channels)
            for i_ch in channels:
                try:
                    theta_sample = self.read_signal_reconstruction(event.id_, i_ch)
                    signal_sample, signal_t = processor.read_deconv_result(event.id_, i_ch)
                    significances.append(
                        sigrec.signal_delta_bic(
                            signal_sample, signal_t, event.mean_n_photoelectrons[i_ch], theta_sample
                        )
                    )
                except FileNotFoundError:
                    significances.append(-100)
            np.save(frame_sign_path, np.array(significances))
            self.log('Significances saved')

        # try:
        #     x, y, theta, phi = self.read_eas_geometry(event.id_)
        # except FileNotFoundError:
        #     theta, phi, inplane_mask = self.reconstruct_eas_angle(event)
        #     x, y = self.reconstruct_eas_axis_pos(event, inplane_mask=inplane_mask)
        #     np.save(self._eas_geometry_path(event.id_), np.array([x, y, theta, phi]))

    # DECONVOLUTION #

    def _deconvolution_result_path(self, event_id: int, i_ch: int) -> Path:
        event_dir = self.DECONV_RESULTS_DIR / str(event_id)
        event_dir.mkdir(exist_ok=True)
        return event_dir / f"{i_ch}.deconv.npz"

    def _process_signal_with_rireff(
        self, signal: Signal, adc_step: float, rireff: RandomizedIr
    ) -> NDArray[(Any, Any), float]:
        n_vec_estimation = rireff.estimate_n_vec(signal, delta=adc_step)

        result_preliminary = mcmc.run_mcmc(
            logposterior=rireff.get_loglikelihood_independent_normdist(signal, delta=adc_step, density=False),
            init_point=n_vec_estimation,
            config=mcmc.SamplingConfig(
                n_walkers=256, n_samples=self.preliminary_run_length, progress_bar=(self.verbosity > 1)
            ),
        )
        taus = result_preliminary.sampler.get_autocorr_time(tol=0, quiet=True)
        tau = taus[np.logical_not(np.isnan(taus))].mean()
        self.log(f'Estimated tau={tau}', 1)

        n_walkers_final = 128
        init_pts = mcmc.extract_independent_sample(
            result_preliminary.sampler, tau_override=tau, desired_sample_size=n_walkers_final
        )
        result = mcmc.run_mcmc(
            logposterior=rireff.get_loglikelihood_mvn(signal, delta=adc_step, density=False),
            init_point=init_pts,
            config=mcmc.SamplingConfig(
                n_walkers=n_walkers_final,
                n_samples=4 * tau,
                progress_bar=(self.verbosity > 1),
                starting_points_strategy='given',
            ),
        )
        return mcmc.extract_independent_sample(result.sampler, tau_override=tau, debug=(self.verbosity > 2))

    # SIGNAL RECONSTRUCTION #

    def _signal_reconstruction_path(self, event_id: int, i_ch: int) -> Path:
        event_dir = self.SIGREC_DIR / str(event_id)
        event_dir.mkdir(exist_ok=True)
        return event_dir / f"{i_ch}.signal.npy"

    def _frame_significance_path(
        self, event_id: int
    ) -> Path:  # stored separately from reconstruction for historical reasons
        event_dir = self.SIGREC_DIR / str(event_id)
        event_dir.mkdir(exist_ok=True)
        return event_dir / "significances.npy"

    def _reconstruct_signal(
        self, sample: sigrec.SignalSample, signal_t: sigrec.SignalSample, mean_n_phels: float
    ) -> NDArray[(Any, Any), float]:
        loglike = sigrec.get_signal_reconstruction_loglike(sample, signal_t, mean_n_phels, simulate_packets=False)
        logprior = sigrec.get_logprior(sample, signal_t, mean_n_phels)

        def logposterior(theta):
            logp = logprior(theta)
            return logp if np.isinf(logp) else logp + loglike(theta)

        n_walkers = 512
        theta_init = sigrec.estimate_theta_from_sample(sample, signal_t, n_walkers)

        tau = 200

        theta_init_mean = theta_init.mean(axis=0)
        theta_init_std = theta_init.std(axis=0)
        self.log(
            "Rough parameters estimation:"
            + f"\n\tn_eas = {theta_init_mean[0]} +/- {theta_init_std[0]}"
            + f"\n\tt_mean = {theta_init_mean[1]} +/- {theta_init_std[1]}"
            + f"\n\tsigma_t = {theta_init_mean[2]} +/- {theta_init_std[2]}",
            3,
        )

        result = mcmc.run_mcmc(
            logposterior=logposterior,
            init_point=theta_init,
            config=mcmc.SamplingConfig(
                n_walkers=n_walkers,
                n_samples=5 * tau,
                progress_bar=(self.verbosity > 1),
                starting_points_strategy='given',
            ),
        )

        return mcmc.extract_independent_sample(result.sampler, tau_override=tau)

    # EAS parameters reconstruction #
    # NOTE: these methods return only max-likelihood estimations for now, no errors!

    def _eas_geometry_path(self, event_id: int) -> Path:
        return self.EAS_PARAMS_DIR / f"{event_id}.eas.npy"

    def read_eas_geometry(self, event_id: int):
        path = self._eas_geometry_path(event_id)
        if not path.exists():
            raise FileNotFoundError(f"No saved reconstructed EAS geometry for {event_id} event")
        x, y, theta, phi = np.load(path)
        return x, y, theta, phi

    def reconstruct_eas_angle(self, event: Event) -> Tuple[float, float, NDArray]:
        x_fov, y_fov, t_means, t_stds = self.get_arrival_times(event)
        self.log(f'{len(x_fov)} points with significance>{self.min_signal_significance}', 2)

        popt, perr, inplane_mask = eas.adaptive_excluding_fit(
            x_fov,
            y_fov,
            t_means,
            t_stds,
            acceptable_angle_between=0.1,
            absolute_distance_exclusion=True,
        )

        self.log(f'{np.sum(inplane_mask)} points left in fit after exclusion', 2)

        theta_opt, phi_opt = popt[:2]
        theta_err, phi_err = popt[:2]  # not needed for now
        return theta_opt, phi_opt, inplane_mask

    def reconstruct_eas_axis_pos(self, event: Event, inplane_mask: NDArray):
        x_fov, y_fov, n_means, n_stds = self.get_parameter_values_on_plane(event.id_, parameter='n')

        logprior, loglike = eas.get_axis_position_logprior_and_loglike(
            *apply_mask(x_fov, y_fov, n_means, n_stds, mask=inplane_mask)
        )

        def logposterior(ax):
            logp = logprior(ax)
            return logp if np.isinf(logp) else logp + loglike(ax)

        n_walkers = 512
        sigma_estimation = 3
        i_max_ch = np.argmax(n_means[inplane_mask])
        max_ch_xy = np.array([x_fov[inplane_mask][i_max_ch], y_fov[inplane_mask][i_max_ch]]).reshape(1, 2)
        init_point = np.tile(max_ch_xy, (n_walkers, 1)) + np.random.normal(scale=sigma_estimation, size=(n_walkers, 2))

        tau = 200
        result = mcmc.run_mcmc(
            logposterior=logposterior,
            init_point=init_point,
            config=mcmc.SamplingConfig(
                n_walkers=n_walkers,
                n_samples=10 * tau,
                starting_points_strategy='given',
                progress_bar=True,
                autocorr_estimation_each=500,
                debug_acceptance_fraction_each=1000,
            ),
        )
        axis_xy_sample = mcmc.extract_independent_sample(result.sampler, tau_override=tau, debug=True)
        i_max_in_sample = np.argmax(np.array([loglike(axis_xy) for axis_xy in axis_xy_sample]))

        return axis_xy_sample[i_max_in_sample, :]

    # temp data reading functions #

    def read_deconv_result(self, event_id, i_ch):
        deconv_path = self._deconvolution_result_path(event_id, i_ch)
        if not deconv_path.exists():
            raise FileNotFoundError(f"No saved deconvolution for {event_id} event {i_ch} channel")
        data = np.load(deconv_path)
        return data['sample'], data['signal_t']

    def read_deconv_frame(self, event_id):
        frame = []
        channels = np.arange(109)
        nan_signal = np.zeros((self.N,))
        nan_signal[:] = np.nan
        for i_ch in channels:
            try:
                sample, signal_t = self.read_deconv_result(event_id, i_ch)
                frame.append(sample.mean(axis=0))
            except FileNotFoundError:
                frame.append(nan_signal)
        return np.array(frame).T, signal_t, channels

    def read_signal_reconstruction(self, event_id, i_ch) -> NDArray:
        sigrec_path = self._signal_reconstruction_path(event_id, i_ch)
        if not sigrec_path.exists():
            raise FileNotFoundError(f"No saved deconvolution for {event_id} event {i_ch} channel")
        theta_sample = np.load(sigrec_path)
        return theta_sample

    def read_reconstruction_marginal_sample_per_channel(
        self, event_id, parameter: Literal['n', 't', 'sigma']
    ) -> Tuple[NDArray, NDArray]:
        parameter_values = ['n', 't', 'sigma']
        parameter_i = parameter_values.index(parameter)
        channels_with_signals = []
        param_samples = []
        for i_ch in range(N_CHANNELS):
            try:
                theta_sample = self.read_signal_reconstruction(event_id, i_ch)
            except FileNotFoundError:
                continue
            channels_with_signals.append(i_ch)
            param_samples.append(theta_sample[:, parameter_i])
        return np.array(param_samples).T, np.array(channels_with_signals)

    def get_parameter_values_on_plane(self, event_id: int, parameter: Literal):
        fov = PmtFov.for_event(event_id)
        param_samples, has_signal = self.read_reconstruction_marginal_sample_per_channel(event_id, parameter=parameter)
        significant = self.read_signal_significances(event_id)[has_signal] > self.min_signal_significance

        return (
            fov.x[has_signal][significant],
            fov.y[has_signal][significant],
            param_samples.mean(axis=0)[significant],
            param_samples.std(axis=0)[significant],
        )

    def get_arrival_times(self, event: Event):
        fov = PmtFov.for_event(event.id_)

        t_samples, has_signal = self.read_reconstruction_marginal_sample_per_channel(event.id_, parameter='t')
        t_samples = t_samples * 12.5  # bin -> ns

        time_delays_ns = fov.delays(H=event.height)[has_signal]
        t_samples -= time_delays_ns
        t_samples -= t_samples.mean()

        significant = self.read_signal_significances(event.id_)[has_signal] > self.min_signal_significance

        return (
            fov.x[has_signal][significant],
            fov.y[has_signal][significant],
            t_samples.mean(axis=0)[significant],
            t_samples.std(axis=0)[significant],
        )

    def read_signal_significances(self, event_id: int) -> NDArray:
        path = self._frame_significance_path(event_id)
        if not path.exists():
            raise FileNotFoundError(f"No saved significances for {event_id} event")
        data = np.load(path)
        return data


if __name__ == "__main__":
    processor = EventProcessor(N=45, verbosity=3)

    processor(Event(10675))

    # processor(Event(10685))
    # processor(Event(10687))
