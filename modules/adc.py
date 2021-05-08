"""
adc: Analog-to-Digital converter effects modeling
"""

import numpy as np

from nptyping import NDArray


def adc_accepted_s_vec(s_vec: NDArray, delta: float) -> NDArray:
    return delta * np.floor(s_vec / delta)
