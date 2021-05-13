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


class Event:
    """Experimental event with all its aspects (frame, mean current, calibration) stored together"""

    # pre-loaded mean currents for all events, converted to code units
    _mean_currents_by_event_id: FloatPerChannel = None

    C_ref = 0.326  # code units * bin / photoelectron

    def __init__(self, event_id):
        self.event_id = event_id
        if self._mean_currents_by_event_id is None:
            self._read_mean_currents()
        self._read_event_frame()
        self._read_calibration()

    @property
    def mean_current(self):
        try:
            return self._mean_currents_by_event_id[self.event_id]
        except KeyError:
            raise ValueError(f"Mean current for event ID {self.event_id} not found!")

    def signal_cu(self, i_ch: int, wnd_start: Optional[int] = None, wnd_end: Optional[int] = None) -> Signal:
        """Mean and high freq parts of the signal summed, is code units"""
        wnd_start = 0 if wnd_start is None else wnd_start
        wnd_end = self.frame.shape[1] if wnd_end is None else wnd_end
        return self.mean_current[i_ch] + self.frame[i_ch, wnd_start:wnd_end]

    def signal_relative(
        self, i_ch: int, wnd_start: Optional[int] = None, wnd_end: Optional[int] = None
    ) -> Tuple[Signal, float]:
        """Full signal ready for deconvolution and its associated ADC delta in relative units"""
        C = self.C_ref / self.calibration[i_ch]
        return self.signal_cu(i_ch, wnd_start, wnd_end) / C, 1 / C

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
    print(e.mean_current[1])
    print(e.calibration)
