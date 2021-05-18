import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from enum import Enum

from typing import Optional


matplotlib.rcParams.update({'font.size': 12})


TIME_LABEL = 'Время, бины'


class Figsize(Enum):
    SMALL = (4, 4)
    NORMAL = (7, 5)
    TWOPANEL_VERT = (7, 10)
    TRIPANEL_HORIZ = (7, 2)


class Color(Enum):
    N = '#0477DC'
    S = '#DC6904'
    S_APPROX = '#db040b'
    N_ESTIMATION = '#f2003d'
    N_INFERRED = '#00a80b'
    THETA = '#a31aff'

    def as_rgb(self):
        hex_color = self.value.lstrip('#')
        return [int(hex_color[i : i + 2], 16) / 256 for i in (0, 2, 4)]  # noqa

    def as_rgba(self, alpha: float = 1.0):
        return [*self.as_rgb(), alpha]

    @classmethod
    def plot_palette(cls):
        _, ax = plt.subplots()
        for i, color in enumerate(cls):
            ax.add_patch(
                Rectangle(xy=(0, i), width=1, height=1, facecolor=color.as_rgba())
            )
            ax.text(0.05, i + 0.5, color.name)
        ax.set_ylim(top=len(cls))
        plt.show()


def _save_or_show(filename: Optional[str]):
    if filename:
        plt.savefig(f'../doc/pic/{filename}.pdf')
    else:
        plt.show()
