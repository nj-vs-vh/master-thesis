import matplotlib
import matplotlib.pyplot as plt

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

    def as_rgb(self):
        hex_color = self.value.lstrip('#')
        return [int(hex_color[i : i + 2], 16) / 256 for i in (0, 2, 4)]  # noqa


def _save_or_show(filename: Optional[str]):
    if filename:
        plt.savefig(f'../doc/pic/{filename}.pdf')
    else:
        plt.show()
