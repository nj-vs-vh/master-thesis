"""
PMT fields of view (FOV) reading and preprocessing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.io import loadmat

from typing import Any
from nptyping import NDArray

from modules.experiment._shared import EXP_DATA_DIR

INSTANT_FOV_DIR = EXP_DATA_DIR / 'instant-fov'


N_CHANNELS = 109
C_LIGHT = 299792458 * 10 ** -9  # m / ns


@dataclass
class PmtFov:
    FOVc: NDArray[(N_CHANNELS, 2), float]
    FOV: NDArray[(N_CHANNELS, Any, Any), float]

    def __post_init__(self):
        self.n_channels = 109
        self.FOVc = self.FOVc[:self.n_channels, :]
        self.FOV = self.FOV[:self.n_channels, :, :]
        self.side = self.FOV.shape[1]
        # legacy from matlab code: original calculated fov has a spatial step of 1, processed instant - 5
        # but seems like now it's always 5..?
        self.step = 300 / (self.side - 1)

    @property
    def x(self) -> NDArray[(N_CHANNELS,), float]:
        return self.FOVc[:, 0]

    @property
    def y(self) -> NDArray[(N_CHANNELS,), float]:
        return self.FOVc[:, 1]

    @property
    def r(self) -> NDArray[(N_CHANNELS,), float]:
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def grid(self) -> NDArray:
        return self.step * (0.501 + np.arange(self.side) - self.side / 2)

    @classmethod
    def default(cls) -> PmtFov:
        """Load default FOV for 1000 m elevation and not inclined detector"""
        data = loadmat(str(EXP_DATA_DIR / 'default-FOV.mat'))
        return cls(FOV=data['FOV'], FOVc=data['FOVc'])

    @classmethod
    def for_event(cls, event_id: int) -> PmtFov:
        """Load default FOV for 1000 m elevation and not inclined detector"""
        data = loadmat(str(INSTANT_FOV_DIR / f'{event_id}.mat'))
        return cls(FOV=data['FOV'], FOVc=data['FOVc'])

    def delays(self, H) -> NDArray[(N_CHANNELS,), float]:
        """Calculate time delays (in ns) between snow and each channel's PMT"""
        return np.sqrt(H ** 2 + self.r ** 2) / C_LIGHT


if __name__ == "__main__":
    fov = PmtFov.for_event(10675)
    print(fov.step)
    fov = PmtFov.default()
    print(fov.step)
