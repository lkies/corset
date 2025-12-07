"""Configuration for analysis and plotting functions."""

# TODO is it a reasonable idea to just use this as a namespace?
# ideally these would just be values in this module but that would lead to bind by value issues when importing
import typing
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


class Unit(typing.NamedTuple):
    """Unit representation with ASCII and LaTeX strings and conversion factor to base unit."""

    ascii: str  #: ASCII representation of the unit
    tex: str  #: LaTeX representation of the unit
    factor: float  #: Conversion factor to the SI base unit


class Config:
    """Configuration namespace for the mode matching solver."""

    class SensitivityUnit(Enum):
        """Units for sensitivity analysis."""

        PER_M2 = Unit(ascii="%/m^2", tex=r"\%/\mathrm{m}^2", factor=1)  #:
        PERCENT_PER_MM2 = Unit(ascii="%/mm^2", tex=r"\%/\mathrm{mm}^2", factor=1e2 * (1e-3**2))  #:
        PERCENT_PER_CM2 = Unit(ascii="%/cm^2", tex=r"\%/\mathrm{cm}^2", factor=1e2 * (1e-2**2))  #:

    sensitivity_unit = SensitivityUnit.PERCENT_PER_CM2  #: Unit for sensitivity analyses
    overlap_levels: typing.ClassVar = [80, 90, 95, 98, 99, 99.5, 99.8, 99.9, 100]
    """Levels for overlap contours in plots"""
    overlap_colormap: str = "turbo"  #: Colormap for overlap contours in plots
    overwrite_dark_theme: bool | None = None  #: Override automatic detection of dark theme in plots
    plot_max_rayleigh_range: float = 200e-3
    """During plotting, the interval of interest is automatically determined
    based on the Rayleigh range of the beam, for large Rayleigh ranges this can
    significantly inflate the plotted region, making the important features hard to see.
    This parameter limits this effect by capping the maximum Rayleigh range considered when
    determining the plotted interval.
    """

    # TODO specify defaults for plotting functions here instead of in the function signatures?

    @classmethod
    def mpl_is_dark(cls) -> bool:
        """Determine whether the current Matplotlib theme is
        dark by analyzing the figure background color."""
        if cls.overwrite_dark_theme is not None:
            return cls.overwrite_dark_theme
        bg_color = to_rgb(plt.rcParams["figure.facecolor"])
        return bool(np.mean(bg_color[:3]) < 0.5)

    @classmethod
    def overlap_colors(cls):
        """Get the colors corresponding to the overlap levels."""
        return plt.get_cmap(cls.overlap_colormap)(np.linspace(0, 1, len(cls.overlap_levels) - 1))
