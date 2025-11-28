# todo is it a reasonable idea to just use this as a namespace?
# ideally these would just be values in this module but that would lead to bind by value issues when importing
import typing
from collections import namedtuple
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

Unit = namedtuple("Unit", ["ascii", "tex", "factor"])


class Config:
    class SensitivityUnit(Enum):
        PER_M2 = Unit(ascii="%/m^2", tex=r"\%/\mathrm{m}^2", factor=1)
        PERCENT_PER_MM2 = Unit(ascii="%/mm^2", tex=r"\%/\mathrm{mm}^2", factor=1e2 * (1e-3**2))
        PERCENT_PER_CM2 = Unit(ascii="%/cm^2", tex=r"\%/\mathrm{cm}^2", factor=1e2 * (1e-2**2))

    SENSITIVITY_UNIT = SensitivityUnit.PERCENT_PER_CM2
    OVERLAP_LEVELS: typing.ClassVar = [80, 90, 95, 98, 99, 99.5, 99.8, 99.9, 100]
    OVERLAP_COLORMAP = "turbo"

    @classmethod
    def overlap_colors(cls):
        return plt.get_cmap(cls.OVERLAP_COLORMAP)(np.linspace(0, 1, len(cls.OVERLAP_LEVELS) - 1))
