
import matplotlib.pyplot as plt

from .common import TIME_LABEL, Figsize, Color, _save_or_show


def plot_real_ir_shape_and_distribution(ir_t, ir_y, Cpmt_vals, Cpmt_pdf, Cpmt_cdf, filename=None):
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, figsize=Figsize.TWOPANEL_VERT.value)

    ax_top.plot(ir_t, ir_y, color=Color.S.value, linewidth=3)
    ax_top.hlines(0, xmin=0, xmax=ir_t.max(), colors=['k'], linewidths=[1])
    ax_top.set_xlabel(TIME_LABEL)
    ax_top.set_ylabel('$h_I(t)$')

    PDF_COLOR = 'r'
    CDF_COLOR = 'b'
    ax_bot.plot(Cpmt_vals, Cpmt_pdf, color=PDF_COLOR)
    ax_bot.set_xlim(left=0, right=7)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_ylabel('Плотность распределения $\\tilde{{C}}_{{PMT}}$', color=PDF_COLOR)
    ax_bot.set_xlabel('Относительный коэффициент усиления ФЭУ')

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(Cpmt_vals, Cpmt_cdf, color=CDF_COLOR)
    ax_bot_r.axvline(1, color='k', label='Среднее значение')
    ax_bot_r.set_ylim(bottom=0)
    ax_bot_r.set_ylabel('Функция распределения $\\tilde{{C}}_{{PMT}}$', color=CDF_COLOR)

    ax_bot_r.legend()

    _save_or_show(filename)
    return fig, (ax_top, ax_bot)
