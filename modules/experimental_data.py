"""
Experimental data reading and preprocessing
"""

import numpy as np
from pathlib import Path

from typing import Any, Tuple, Optional
from nptyping import NDArray


CUR_DIR = Path(__file__).parent
EXP_DATA_DIR = CUR_DIR / '../experimental-data'


N_CHANNELS = 109
SIGNAL_LENGTH = Any

# custom typing
FloatPerChannel = NDArray[(N_CHANNELS,), float]
SignalPerChannel = NDArray[(N_CHANNELS, SIGNAL_LENGTH), float]
Signal = NDArray[(SIGNAL_LENGTH,), float]


class BrokenChannelException(Exception):
    pass


class Event:
    """Experimental event with all its aspects (frame, mean current, calibration) stored together"""

    # pre-loaded mean currents for all events, converted to code units
    _mean_currents_by_event_id: FloatPerChannel = None

    def __init__(self, event_id):
        self.event_id = event_id
        if self._mean_currents_by_event_id is None:
            self._read_mean_currents()
        self._read_event_frame()
        self._read_calibration()

    @property
    def mean_currents(self) -> FloatPerChannel:
        try:
            return self._mean_currents_by_event_id[self.event_id]
        except KeyError:
            raise ValueError(f"Mean current for event ID {self.event_id} not found!")

    @property
    def mean_n_photoelectrons(self) -> FloatPerChannel:
        return self.mean_currents / self.C

    C_ref = 0.326  # code units * bin / photoelectron

    @property
    def C(self) -> FloatPerChannel:
        return self.C_ref / self.calibration

    TRIGGER_BIN = 433  # approximately

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
        center_bin = self.TRIGGER_BIN if center_bin is None else center_bin
        window = 100 if window is None else window
        window_half = window // 2
        t = np.arange(center_bin - window_half, center_bin + window_half + 1)
        signal = self.mean_currents[i_ch] + self.frame[i_ch, t]
        adc_step = 1
        return t, signal / C, adc_step / C

    BROKEN_CHANNELS = {49, 78}

    def validate_i_ch(self, i_ch: int):
        if not 0 <= i_ch < N_CHANNELS:
            raise ValueError(f"Invalid chanell no: {i_ch}, must be from 0 to {N_CHANNELS - 1}")
        if i_ch in self.BROKEN_CHANNELS:
            raise BrokenChannelException(f"Channel {i_ch} is broken")

    # Data reading methods #

    def _read_event_frame(self):
        try:
            with open(EXP_DATA_DIR / f'event-frames/{self.event_id}.txt') as f:
                self.height: float = float(f.readline().strip())
                # TODO: find out what it is :)
                *self.inclin, self.SOMETHIN_UNKNOWN = [float(v) for v in f.readline().strip().split()]
                f.readline()
                self.frame: SignalPerChannel = np.loadtxt(f).T[:N_CHANNELS, :]
        except FileNotFoundError:
            raise ValueError(f"Frame for event ID {self.event_id} not found!")

    def _read_calibration(self):
        try:
            calibration_data = np.loadtxt(EXP_DATA_DIR / f'event-calibrations/{self.event_id}.cal')
            self.calibration: FloatPerChannel = calibration_data[:N_CHANNELS, -1]
        except OSError:
            raise ValueError(f"Calibration for event ID {self.event_id} not found!")

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


if __name__ == "__main__":
    e = Event(10675)
    # print(mc.min(), mc.max())
    print(e.mean_currents[1])
    print(e.calibration)
