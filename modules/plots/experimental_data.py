import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scint
import matplotlib.lines as lines

from ._shared import TIME_LABEL, Figsize, Color, _save_or_show

from ..experiment.events import Event, BrokenChannelException
from ..experiment.fov import PmtFov


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


def plot_fov(fov: PmtFov, filename=None):
    # fig, axd = plt.subplot_mosaic(
    #     """
    #     ABCD
    #     EEED
    #     """,
    #     figsize=Figsize.TWOPANEL_VERT.value
    # )
    # ax = axd['E']
    # colormap_ax = axd['D']
    # channel_axes = [axd['A'], axd['B'], axd['C']

    fig = plt.figure(figsize=(7, 8.5))
    gs = plt.GridSpec(ncols=4, nrows=2, width_ratios=[1, 1, 1, 0.15], height_ratios=[1, 4], hspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[1, 0:3])
    colormap_ax = fig.add_subplot(gs[:, 3])
    channel_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    channel_axes[0].set_anchor('W')
    channel_axes[-1].set_anchor('E')

    fov_local_grid = np.arange(fov.side) - fov.side / 2

    # ported from print_FOV.m

    def bounds_for_data(d):
        center = np.median(d)
        halfrange = 1.3 * np.max(np.abs(d - center))
        return center - halfrange, center + halfrange

    RESOLUTION = 600
    COLORMAP = 'OrRd'

    x_global_grid = np.linspace(*bounds_for_data(fov.FOVc[:, 0]), RESOLUTION)
    y_global_grid = np.linspace(*bounds_for_data(fov.FOVc[:, 1]), RESOLUTION)
    image = np.zeros((RESOLUTION, RESOLUTION), dtype=float)

    for i_ch, ch_fov in enumerate(fov.FOV):
        ch_fov = np.squeeze(ch_fov)
        image += scint.interp2d(
            fov.step * fov_local_grid + fov.FOVc[i_ch, 0],
            fov.step * fov_local_grid + fov.FOVc[i_ch, 1],
            ch_fov,
            kind='linear',
            copy=False,
            bounds_error=False,
            fill_value=0,
        )(x_global_grid, y_global_grid)

    image_max = np.max(image)
    plotted_img = ax.imshow(
        image,
        vmax=image_max,
        cmap=COLORMAP,
        origin='lower',
        interpolation='none',
        extent=[x_global_grid[0], x_global_grid[-1], y_global_grid[0], y_global_grid[-1]],
    )

    ax.scatter(fov.FOVc[:, 0], fov.FOVc[:, 1], 10, 'b', marker='.', label='Центры полей зрения')
    ax.scatter(0, 0, 200, 'k', marker='+', label='Проекция установки')
    ax.set_anchor('S')

    cbar = plt.colorbar(plotted_img, cax=colormap_ax)
    cbar_max = (1 + int(image_max * 10 ** 5)) / 10 ** 5
    cbar_ticks = np.linspace(0.0, cbar_max, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_label('Коэффициент сбора')
    ax.set_xlabel('$x_{{\\mathrm{{запад-восток}}}}$, м')
    ax.set_ylabel('$y_{{\\mathrm{{юг-север}}}}$, м')
    ax.legend(loc='lower left')

    # plotting single channel fovs #

    for i_ch, chax in zip([0, 13, 100], channel_axes):
        cut_edges = 4
        center_region_coords = np.arange(fov.side / cut_edges, fov.side * (cut_edges-1) / cut_edges, dtype=int)
        ch_fov = np.squeeze(fov.FOV[i_ch, center_region_coords, :])
        ch_fov = ch_fov[:, center_region_coords]
        chax.imshow(ch_fov, vmax=image_max, cmap=COLORMAP)
        center = ch_fov.shape[0] // 2
        chax.scatter(center, center, 10, 'b', marker='.')
        chax.set_xticks([])
        chax.set_yticks([])

        arrow_start = fig.transFigure.inverted().transform(
            ax.transData.transform(fov.FOVc[i_ch, :])
        )
        arrow_end = fig.transFigure.inverted().transform(
            chax.transAxes.transform((0.5, -0.05))
        )
        arrow_start = arrow_end + (arrow_start - arrow_end) * (1 - 0.015)
        fig.add_artist(
            lines.Line2D([arrow_start[0], arrow_end[0]], [arrow_start[1], arrow_end[1]], c=[0.3] * 3)
        )
        chax.set_title('#' + str(i_ch + 1))

    _save_or_show(filename)

    return fig, ax
