"""Plotting functions for optical setups and mode matching solutions and analyses."""

from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection, LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter

from .config import Config

# prevent circular import for type annotation in function signature
if TYPE_CHECKING:
    from .core import OpticalSetup
    from .solver import ModeMatchingSolution

RELATIVE_MARGIN = 0.1  #: Relative margin size for plotting optical setups

milli_formatter = FuncFormatter(lambda x, _: f"{x*1e3:.0f}")  #: Formatter for millimeter axes
micro_formatter = FuncFormatter(lambda x, _: f"{x*1e6:.0f}")  #: Formatter for micrometer axes


def fig_to_png(fig: Figure) -> bytes:
    """Convert a Matplotlib figure to a PNG as bytes.

    Args:
        fig: The figure to convert.

    Returns:
        The PNG representation of the figure as bytes.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@dataclass(frozen=True)
class OpticalSetupPlot:
    """Plot of an optical setup with references to the plot elements."""

    ax: Axes  #: Axes containing the plot
    zs: np.ndarray  #: Z-coordinates of the beam profile
    rs: np.ndarray  #: Radii of the beam profile
    beam: FillBetweenPolyCollection  #: Fill between collection for the beam
    lenses: list[tuple[LineCollection, Annotation]]  #: List of lens plot elements

    @cached_property
    def r_max(self) -> float:
        """Maximum beam radius in the plot."""
        return np.max(self.rs)


@dataclass(frozen=True)
class ModeMatchingPlot:
    """Plot of a mode matching solution with references to the plot elements."""

    ax: Axes  #: Axes containing the plot
    setup: OpticalSetupPlot  #: Plot of the actual optical setup
    desired_beam: OpticalSetupPlot  #: Plot of the desired beam profile
    ranges: list[Rectangle]  #: List of ranges in the plot
    apertures: list[Line2D]  #: List of aperture lines in the plot
    passages: list[Rectangle]  #: List of passage rectangles in the plot


def plot_optical_setup(
    self: "OpticalSetup",
    *,
    ax: Axes | None = None,
    points: int | np.ndarray | None = None,
    limits: tuple[float, float] | None = None,
    beam_kwargs: dict = {"color": "C0", "alpha": 0.5},  # noqa: B006
    free_lenses: list[int] = [],  # noqa: B006
    # TODO add back other configuration options?
) -> OpticalSetupPlot:
    """Plot the the beam profile and optical elements of the setup.

    Args:
        self: The optical setup instance.
        ax: The axes to plot on. If `None`, the current axes are used.
        points: Number of points or specific z-coordinates to evaluate the beam profile. If `None`, 500 points are used.
        limits: Z-coordinate limits for the plot. If `None`, limits are determined from the beam and elements.
        beam_kwargs: Additional keyword arguments passed to the beam plot.
        free_lenses: Indices of lenses to treat as free elements in the plot.

    Returns:
        An :class:`OpticalSetupPlot` containing references to the plot elements.
    """

    ax = ax or plt.gca()

    lens_positions = [pos for pos, _ in self.elements]

    if isinstance(points, np.ndarray):
        zs = points
    else:
        if not limits:
            cap = Config.plot_max_rayleigh_range
            all_bounds = [
                self.beams[0].focus - min(cap, self.beams[0].rayleigh_range),
                self.beams[-1].focus + min(cap, self.beams[-1].rayleigh_range),
                *(),
                *lens_positions,
            ]
            if self.elements:
                all_bounds.append(self.elements[0][0] - min(cap, self.beams[0].rayleigh_range))
                all_bounds.append(self.elements[-1][0] + min(cap, self.beams[-1].rayleigh_range))

            limits = (min(all_bounds), max(all_bounds))

        num_points = points if isinstance(points, int) else 500
        zs = np.linspace(limits[0], limits[1], num_points)

    rs = self.radius(zs)
    fb_col = ax.fill_between(zs, -rs, rs, **{"zorder": 100, **beam_kwargs})
    # TODO make beam plot function?

    w_max = np.max(rs)

    # TODO factor out into plot lens function?
    lenses = []
    for i, (pos, lens) in enumerate(self.elements):
        color = "C2"
        zorder = 200
        if lens.left_margin or lens.right_margin:
            rect = Rectangle(
                (pos - lens.left_margin, -w_max * (1 + RELATIVE_MARGIN)),
                lens.left_margin + lens.right_margin,
                2 * w_max * (1 + RELATIVE_MARGIN),
                fc="none",
                ec=color,
                ls="--",
                zorder=zorder,
            )
            ax.add_patch(rect)

        lens_col = ax.vlines(
            pos,
            -w_max * (1 + RELATIVE_MARGIN),
            w_max * (1 + RELATIVE_MARGIN),
            color=color,
            zorder=zorder,
        )
        # TODO label and enumerate free lenses?
        label_text = lens.name if lens.name is not None else f"f={round(lens.focal_length*1e3)}mm"
        if i in free_lenses:
            label_text = f"$L_{i}$: {label_text} @{round(pos*1e3)}mm"
        label = ax.text(
            pos,
            -w_max * (1 + RELATIVE_MARGIN),
            label_text,
            va="center",
            ha="left",
            rotation="vertical",
            rotation_mode="anchor",
            bbox={"fc": plt.rcParams["axes.facecolor"], "ec": "none", "alpha": 0.5},
            zorder=zorder,
        )

        lenses.append((lens_col, label))

    ax.set_ylim(
        min(-w_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[0]),
        max(w_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[1]),
    )

    ax.set_xlabel("z in mm")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(r"w(z) in $\mathrm{\mu m}$")
    ax.yaxis.set_major_formatter(micro_formatter)

    return OpticalSetupPlot(ax=ax, zs=zs, rs=rs, beam=fb_col, lenses=lenses)  # pyright: ignore[reportArgumentType]


def plot_mode_match_solution_setup(self: "ModeMatchingSolution", *, ax: Axes | None = None) -> ModeMatchingPlot:
    """Plot the mode matching solution setup including the desired beam and constraints.

    Args:
        self: The mode matching solution instance.
        ax: The axes to plot on. If `None`, the current axes are used.
    Returns:
        An :class:`ModeMatchingPlot` containing references to the plot elements.
    """
    from .core import OpticalSetup
    from .solver import Aperture, Passage

    ax = ax or plt.gca()

    problem = self.candidate.problem

    setup_plot = plot_optical_setup(self.setup, ax=ax, free_lenses=self.candidate.parametrized_setup.free_elements)
    desired_plot = plot_optical_setup(OpticalSetup(problem.desired_beam, []), ax=ax, beam_kwargs={"color": "C1"})

    ranges = []
    for range_ in problem.ranges:
        # TODO color configuratbility?
        rectangle = Rectangle(
            (range_.left, -setup_plot.r_max * (1 + 2 * RELATIVE_MARGIN)),
            range_.right - range_.left,
            2 * setup_plot.r_max * (1 + 2 * RELATIVE_MARGIN),
            fc="none",
            ec="lightgrey",
            ls="--",
            zorder=50,  # TODO
        )
        ranges.append(ax.add_patch(rectangle))

    apertures = []
    passages = []
    for con in problem.constraints:
        if isinstance(con, Aperture):
            apertures.append(
                ax.plot([con.position, con.position], [-con.radius, con.radius], marker="o", color="C4", zorder=80)
            )
        elif isinstance(con, Passage):
            rect = Rectangle(
                (con.left, -con.radius),
                con.right - con.left,
                2 * con.radius,
                fc="C4",
                ec="none",
            )
            passages.append(ax.add_patch(rect))

    ax.set_title(f"Optical Setup ({self.overlap*100:.2f}% mode overlap)")

    return ModeMatchingPlot(
        ax=ax,
        setup=setup_plot,
        desired_beam=desired_plot,
        ranges=ranges,
        apertures=apertures,
        passages=passages,
    )


@dataclass(frozen=True)
class ReachabilityPlot:
    """Plot of a reachability analysis with references to the plot elements."""

    ax: Axes  #: Axes containing the plot
    lines: list[list[Line2D]]  #: Lines representing parameter variations
    contour: Any  #: Contour plot of the mode overlap
    colorbar: Colorbar | None  #: Colorbar for the contour plot


@dataclass(frozen=True)
class SensitivityPlot:
    """Plot of a sensitivity analysis with references to the plot elements."""

    ax: Axes  #: Axes containing the plot
    contours: list[Any]  #: Contour plots of the sensitivity
    colorbar: Colorbar | None  #: Colorbar for the contour plots
    handles: list  #: Handles for the plot elements


def plot_reachability(
    self: "ModeMatchingSolution",
    *,
    displacement: float | list[float] = 5e-3,
    num_samples: int | list[int] = 7,
    grid_step: int | list[int] = 2,
    dimensions: list[int] | None = None,
    focus_range: tuple[float, float] | None = None,
    waist_range: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> ReachabilityPlot:
    """Plot a reachability analysis of the mode matching solution.

    Args:
        self: The mode matching solution instance.
        displacement: Maximum displacement from the optimal position(s) in meters.
        num_samples: Number of samples to take along each dimension.
        grid_step: Step size to skip certain lines for increased smoothness while retaining clarity.
        dimensions: Indices of the dimensions to analyze. If `None`, all dimensions are used.
        focus_range: Range of focus values to display on the x-axis. If `None`, determined automatically.
        waist_range: Range of waist values to display on the y-axis. If `None`, determined automatically.
        ax: The axes to plot on. If `None`, the current axes are used.

    Returns:
        A :class:`ReachabilityPlot` containing references to the plot elements.
    """

    from .analysis import make_focus_and_waist, vector_partial
    from .solver import mode_overlap

    ax = ax or plt.gca()
    if dimensions is None:
        dimensions = list(range(len(self.positions)))

    num_dof = len(dimensions)
    displacement = cast(list, [displacement] * num_dof if np.isscalar(displacement) else displacement)
    num_samples = cast(list, [num_samples] * num_dof if np.isscalar(num_samples) else num_samples)
    grid_step = cast(list, [grid_step] * num_dof if np.isscalar(grid_step) else grid_step)

    linspaces = [
        np.linspace(-d, d, num=n) + offset
        for d, n, offset in zip(
            displacement, num_samples, self.positions, strict=True  # pyright: ignore[reportArgumentType]
        )
    ]

    if focus_range is not None:
        ax.set_xlim(focus_range)

    if waist_range is not None:
        ax.set_ylim(waist_range)

    focus_and_waist = np.vectorize(
        vector_partial(make_focus_and_waist(self), self.positions, dimensions), signature="(n)->(2)"
    )

    grids = np.moveaxis(np.meshgrid(*linspaces, indexing="ij"), 0, -1)  # pyright: ignore[reportArgumentType]
    focuses, waists = np.moveaxis(focus_and_waist(grids), -1, 0)
    delta_focuses = focuses - self.candidate.problem.desired_beam.focus

    ax.scatter(0, self.candidate.problem.desired_beam.waist, color=plt.rcParams["text.color"], marker="o", zorder=100)

    def select_lines(arr: np.ndarray, dim: int, steps: list[int]) -> np.ndarray:
        steps = [step if i != dim else 1 for i, step in enumerate(steps)]
        indices = tuple(slice(None, None, step) for step in steps)
        return np.moveaxis(arr[indices], dim, -1).reshape(-1, arr.shape[dim])

    lines: list[list[Line2D]] = []
    for i, (dim, disp) in enumerate(zip(dimensions, displacement, strict=True)):
        focuses_flat = select_lines(delta_focuses, i, grid_step)  # pyright: ignore[reportArgumentType]
        waists_flat = select_lines(waists, i, grid_step)  # pyright: ignore[reportArgumentType]
        lines.append([])
        for focus, waist in zip(focuses_flat, waists_flat, strict=True):
            line = ax.plot(focus, waist, color=f"C{dim}", label=rf"$\Delta x_{i}$ ($\pm{disp*1e3:.1f}$ mm)")[0]
            lines[-1].append(line)
        ax.annotate(
            "",
            xy=(focuses_flat[0][1], waists_flat[0][1]),
            xytext=(focuses_flat[0][0], waists_flat[0][0]),
            arrowprops={"color": f"C{i}", "headwidth": 10, "width": 2},
        )

    ax.legend(handles=[line[0] for line in lines])

    grid_n = 50
    focuses_grid, waists_grid = np.meshgrid(np.linspace(*ax.get_xlim(), grid_n), np.linspace(*ax.get_ylim(), grid_n))
    overlap = mode_overlap(
        focuses_grid,  # pyright: ignore[reportArgumentType]
        self.candidate.problem.desired_beam.waist,
        waists_grid,  # pyright: ignore[reportArgumentType]
        self.candidate.problem.setup.initial_beam.wavelength,
    )

    levels = Config.overlap_levels
    colors = Config.overlap_colors()
    res = ax.contourf(focuses_grid, waists_grid, overlap * 100, levels=levels, colors=colors, alpha=0.5)
    cb = ax.figure.colorbar(res, label="Mode overlap (%)")

    jacobian = self.analysis.focus_and_waist_jacobian

    ax.set_xlabel(rf"$\Delta z_0$ in mm ($\nabla z_0$=[{' '.join(f'{x:.3f}' for x in jacobian[0])}])")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(rf"$w_0$ in $\mathrm{{\mu m}}$ ($\nabla w_0$=[{' '.join(f'{x:.3f}' for x in jacobian[1]*1e3)}]e-3")
    ax.yaxis.set_major_formatter(micro_formatter)

    ax.set_title("Reachability Analysis")

    return ReachabilityPlot(ax=ax, lines=lines, contour=res, colorbar=cb)


def plot_sensitivity(
    self: "ModeMatchingSolution",
    *,
    dimensions: tuple[int, int] | tuple[int, int, int] | None = None,
    worst_overlap: float = 0.98,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    z_n: int = 3,  # more is too cluttered most times
    force_contours: bool = False,
    ax: Axes | None = None,
) -> SensitivityPlot:
    """Plot a sensitivity analysis of the mode matching solution.

    Args:
        self: The mode matching solution instance.
        dimensions: Indices of the dimensions to analyze. If `None`, the two least
            coupled dimensions are used as the x and y dimensions, and the remaining most sensitive
            dimension is used as the auxiliary z dimension if applicable.
        worst_overlap: Worst-case mode overlap contour line that should still be fully visible in the plot.
        x_range: Range of x-axis values to display. If `None`, determined automatically from worst_overlap.
        y_range: Range of y-axis values to display. If `None`, determined automatically from worst_overlap.
        z_range: Range of z-axis values to display. If `None`, determined automatically from worst_overlap.
        z_n: Number of z-slices to plot if 3 dimensions are used.
        force_contours: If `True`, always use contour lines instead of filled contours for plots with two degrees of freedom.
        ax: The axes to plot on. If `None`, the current axes are used.

    Returns:
        A :class:`SensitivityPlot` containing references to the plot elements.
    """

    from .analysis import make_mode_overlap, vector_partial

    ax = ax or plt.gca()

    if dimensions is None:
        dimensions = self.analysis.min_coupling_pair
        if len(self.positions) > 2:
            all_dims = set(range(len(self.positions)))
            remaining_dims = list(all_dims - set(dimensions))
            most_sensitive = max(remaining_dims, key=lambda dim: self.analysis.sensitivities[dim, dim])
            dimensions = (*dimensions, most_sensitive)

    mode_overlap = np.vectorize(
        vector_partial(make_mode_overlap(self), self.positions, dimensions),  # pyright: ignore[reportArgumentType]
        signature="(n)->()",
    )

    # the hessian will always be badly conditioned for ndim > 2, so only use the subspace we care about
    # which should be well conditioned
    A_xy_inv = np.linalg.inv(-self.analysis.hessian[np.ix_(dimensions[:2], dimensions[:2])] / 2)  # noqa: N806
    if x_range is None:
        abs_max_range_x = np.sqrt(A_xy_inv[0, 0] * (1 - worst_overlap))
        x_range = (-abs_max_range_x * 1.1, abs_max_range_x * 1.1)
    if y_range is None:
        abs_max_range_y = np.sqrt(A_xy_inv[1, 1] * (1 - worst_overlap))
        y_range = (-abs_max_range_y * 1.1, abs_max_range_y * 1.1)

    if len(dimensions) > 2:
        A_xyz = -self.analysis.hessian[np.ix_(dimensions, dimensions)]  # noqa: N806
        eigs = np.linalg.eigh(A_xyz)
        v_min = eigs.eigenvectors[:, np.argmin(eigs.eigenvalues)]
        abs_max_range_z = min(abs(v_min[2] / v_min[0] * abs_max_range_x), abs(v_min[2] / v_min[1] * abs_max_range_y))
        z_range = (-abs_max_range_z, abs_max_range_z)

    grid_n = 50
    base = np.array([self.positions[dim] for dim in dimensions])
    xs = np.linspace(*x_range, grid_n)
    ys = np.linspace(*y_range, grid_n)
    xsg, ysg = np.meshgrid(xs, ys)
    cmap = plt.get_cmap("coolwarm") if Config.mpl_is_dark() else plt.get_cmap("berlin")

    contours = []
    colorbar = None
    handles: list = []

    if len(dimensions) == 2:
        overlaps = mode_overlap(np.stack([xsg, ysg], axis=-1) + base)
        if force_contours:
            cont = ax.contour(xsg, ysg, overlaps * 100, levels=Config.overlap_levels, colors=cmap(0.5))
            ax.clabel(cont, fmt="%1.1f%%")
            contours.append(cont)
        else:
            cont = ax.contourf(xsg, ysg, overlaps * 100, levels=Config.overlap_levels, colors=Config.overlap_colors())
            colorbar = ax.figure.colorbar(cont, label="Mode overlap (%)")
            contours.append(cont)
    else:
        z_colors = cmap(np.linspace(0, 1, z_n))
        zs = np.linspace(*z_range, z_n)  # pyright: ignore[reportCallIssue]
        for i, (color, z) in enumerate(zip(z_colors, zs, strict=True)):
            positions = np.stack([xsg, ysg, z * np.ones_like(xsg)], axis=-1) + base
            overlaps = mode_overlap(positions)
            zorder = 100 if i == z_n // 2 else 10
            cont = ax.contour(
                xsg, ysg, overlaps * 100, levels=Config.overlap_levels, colors=[color], alpha=0.7, zorder=zorder
            )
            contours.append(cont)
            if i == z_n // 2:
                ax.clabel(cont, fmt="%1.1f%%", colors=[color])
        handles = [
            Line2D([], [], color=color, label=rf"$\Delta x_{{{dimensions[2]}}} = {value*1e3:.1f}$ mm")
            for color, value in zip(z_colors, zs, strict=True)
        ]
        ax.legend(handles=handles).set_zorder(1000)

    unit = Config.sensitivity_unit
    dims = dimensions
    sens_x = self.analysis.sensitivities[dims[0], dims[0]] * unit.value.factor
    sens_y = self.analysis.sensitivities[dims[1], dims[1]] * unit.value.factor
    ax.set_xlabel(rf"$\Delta x_{{{dims[0]}}}$ in mm ($s_{{{str(dims[0])*2}}}={sens_x:.2f}{unit.value.tex}$)")
    ax.set_ylabel(rf"$\Delta x_{{{dims[1]}}}$ in mm ($s_{{{str(dims[1])*2}}}={sens_y:.2f}{unit.value.tex}$)")
    ax.xaxis.set_major_formatter(milli_formatter)
    ax.yaxis.set_major_formatter(milli_formatter)

    ax.set_title(rf"Sensitivity Analysis ($r_{{{dims[0]}{dims[1]}}}={self.analysis.min_coupling*100:.2f}\%$)")

    return SensitivityPlot(ax=ax, contours=contours, colorbar=colorbar, handles=handles)


def plot_mode_match_solution_all(
    self: "ModeMatchingSolution",
    *,
    plot_kwargs: dict = {},  # noqa: B006
    reachability_kwargs: dict = {},  # noqa: B006
    sensitivity_kwargs: dict = {},  # noqa: B006
) -> tuple[Figure, tuple[ModeMatchingPlot, ReachabilityPlot, SensitivityPlot]]:
    """Plot the mode matching solution setup, reachability analysis, and sensitivity analysis in a single figure.

    Args:
        self: The mode matching solution instance.
        plot_kwargs: Additional keyword arguments for the mode matching plot.
        reachability_kwargs: Additional keyword arguments for the reachability plot.
        sensitivity_kwargs: Additional keyword arguments for the sensitivity plot.

    Returns:
        A tuple containing the figure and an inner tuple with the three plot objects.
    """

    fig, (axl, axr, axc) = plt.subplots(
        1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [2, 1, 1]}, tight_layout=True
    )
    sol_plot = self.plot_setup(ax=axl, **plot_kwargs)
    reach_plot = self.plot_reachability(ax=axr, **reachability_kwargs)
    sens_plot = self.plot_sensitivity(ax=axc, **sensitivity_kwargs)
    return fig, (sol_plot, reach_plot, sens_plot)
