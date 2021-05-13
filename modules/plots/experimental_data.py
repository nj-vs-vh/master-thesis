import numpy as np
import matplotlib.pyplot as plt

from nptyping import NDArray

from ._shared import TIME_LABEL, Figsize, Color, _save_or_show


def plot_signal_in_channel(signal: NDArray, adc_step: float, filename=None):
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    t_signal = np.arange(1, signal.size + 1)
    ax.errorbar(t_signal, signal + adc_step / 2, yerr=adc_step / 2, fmt='.-', elinewidth=2, color=Color.S.value)

    ax.set_xlabel(TIME_LABEL)

    _save_or_show(filename)
    return fig, ax
