import numpy as np
import matplotlib.pyplot as plt

from ._shared import TIME_LABEL, Figsize, Color, _save_or_show

from ..experimental_data import Event, BrokenChannelException


CHANNEL_NO_LABEL = 'Номер канала'


def plot_signal_in_channel(event: Event, i_ch: int, filename=None, **signal_in_channel_kwargs):
    """Kind may be 'relative' or 'code_units'"""
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    t, signal, adc_step = event.signal_in_channel(i_ch=i_ch, **signal_in_channel_kwargs)

    ax.errorbar(t, signal + adc_step / 2, yerr=adc_step / 2, fmt='.-', elinewidth=2, color=Color.S.value)

    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(f'$s_{{{i_ch}}}$')

    _save_or_show(filename)
    return fig, ax


def plot_signals_frame(event: Event, filename=None, **signal_in_channel_kwargs):
    fig, ax = plt.subplots(figsize=Figsize.NORMAL.value)

    common_t = None
    channel_i = np.arange(0, 109)
    signals = []
    for i_ch in channel_i:
        try:
            t, signal, adc_step = event.signal_in_channel(i_ch=i_ch, **signal_in_channel_kwargs)
            common_t = t
            signals.append(signal + adc_step)
        except BrokenChannelException:
            nan_signal = np.zeros_like(signal)
            nan_signal[:] = np.nan
            signals.append(nan_signal)

    channel_i += 1
    signals = np.array(signals).T

    mesh = ax.pcolormesh(channel_i, common_t, signals, shading='nearest')
    cbar = plt.colorbar(mesh)
    cbar.set_label('Приведённые единицы')

    ax.set_xticks([1, 30, 60, 90, 109])
    ax.set_ylabel(TIME_LABEL)
    ax.set_xlabel(CHANNEL_NO_LABEL)

    _save_or_show(filename)
    return fig, ax
