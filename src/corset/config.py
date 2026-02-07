"""Configuration for analysis and plotting functions."""

# TODO is it a reasonable idea to just use this as a namespace?
# ideally these would just be values in this module but that would lead to bind by value issues when importing
import typing
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

T = typing.TypeVar("T")


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

    overwrite_dark_theme: bool | None = None  #: Override automatic detection of dark theme in plots

    @classmethod
    def mpl_is_dark(cls) -> bool:
        """Determine whether the current Matplotlib theme is
        dark by analyzing the figure background color.

        Returns:
            `True` if the current Matplotlib theme is dark, `False` otherwise.
        """
        if cls.overwrite_dark_theme is not None:
            return cls.overwrite_dark_theme
        bg_color = to_rgb(plt.rcParams["figure.facecolor"])
        return bool(np.mean(bg_color[:3]) < 0.5)

    class Overlap:
        """Configuration of default values for overlap contour plots."""

        levels: typing.ClassVar[list[float]] = [80, 90, 95, 98, 99, 99.5, 99.8, 99.9, 100]
        """Overlap levels in percent."""
        colormap: str = "turbo"
        """Colormap for overlap levels."""
        grid_resolution: int = 50
        """Grid resolution for overlap contour plots."""

        @classmethod
        def colors(cls) -> typing.Sequence[tuple[float, float, float, float]]:
            """Get the colors corresponding to the overlap levels.

            Returns:
                A sequence of RGBA colors for each overlap level.
            """
            return plt.get_cmap(cls.colormap)(np.linspace(0, 1, len(cls.levels) - 1))  # type: ignore  # noqa: PGH003

    class PlotSetup:
        """Configuration of default values for setup plots."""

        rayleigh_range_cap: float = 200e-3
        """During plotting, the interval of interest is automatically determined
        based on the Rayleigh range of the beam, for large Rayleigh ranges this can
        significantly inflate the plotted region, making the important features hard to see.
        This parameter limits this effect by capping the maximum Rayleigh range considered when
        determining the plotted interval.
        """
        beam_points: int = 500
        """Number of different :math:`z` points to use when plotting beam propagation."""
        show_legend: bool = False
        """Whether to show a legend describing the plot aspects."""
        legend_loc: str = "lower left"
        """Location of the legend in the plot."""
        beam_kwargs: typing.ClassVar[dict] = {"color": "C0", "alpha": 0.5}
        """The kwargs to pass to :meth:`~matplotlib.axes.Axes.fill_between`"""
        confidence_interval: float | bool = 0.95
        """Confidence interval for the beam envelope. If `False`, no confidence interval is shown."""

    class PlotSolution:
        """Configuration of default values for mode matching solution plots."""

        setup_kwargs: typing.ClassVar[dict] = {}
        """Keyword arguments to pass to the setup plot."""
        desired_kwargs: typing.ClassVar[dict] = {"color": "C1", "label": "Desired Beam"}
        """Keyword arguments for the desired beam plot."""
        show_legend: bool = False
        """Whether to show a legend describing the plot aspects."""
        legend_loc: str = "lower left"
        """Location of the legend in the plot."""

    class PlotReachability:
        """Configuration of default values for reachability plots."""

        displacement: float | list[float] = 5e-3
        """Displacement(s) to use for all or each dimension (if a list)."""
        num_samples: int | list[int] = 7
        """Number of samples to use for all or each dimension (if a list)."""
        line_step: int | list[int] = 2
        """Line step to use for all or each dimension (if a list),
        i.e. the step size to skip certain lines for increased smoothness while retaining clarity."""
        confidence_interval: float | bool = 0.95
        """Confidence ellipse for focus position and waist radius. If `False`, no confidence ellipse is shown."""

    class PlotSensitivity:
        """Configuration of default values for sensitivity plots."""

        worst_overlap: float = 0.98
        """Approximate worst overlap that should still be fully contained in plot used for
        automatic plotting range determination."""
        num_samples_z: int = 3  # more is too cluttered most times
        """Number of z-slices to plot if 3 dimensions are used."""
        force_contour_lines: bool = False
        """Whether to always use contour lines instead of filled
        contours for plots with two degrees of freedom."""
        confidence_interval: float | bool = 0.95
        """Confidence ellipse probability in terms of the required x and y displacements.
        If `False`, no confidence ellipse is shown."""

    class PlotAll:
        """Configuration of default values for mode matching overview plots containing the setup,
        reachability, and sensitivity plots."""

        figsize: tuple[float, float] = (16, 4)
        """Figure size for the overview plot."""
        width_ratios: tuple[float, float, float] = (2, 1, 1)
        """Width ratios for the three subplots in the overview plot."""
        tight_layout: bool = True
        """Whether to use tight layout for the overview plot."""
        setup_kwargs: typing.ClassVar[dict] = {}
        """Keyword arguments to pass to the setup plot."""
        reachability_kwargs: typing.ClassVar[dict] = {}
        """Keyword arguments to pass to the reachability plot."""
        sensitivity_kwargs: typing.ClassVar[dict] = {}
        """Keyword arguments to pass to the sensitivity plot."""

    @staticmethod
    def get(value: T | None, default: T) -> T:
        """Get a configuration value, using the default if the provided value is `None`.

        Args:
            value: The value to check.
            default: The default value to use if `value` is `None`.

        Returns:
            The provided value or the default.
        """
        return default if value is None else value
