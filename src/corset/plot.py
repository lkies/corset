"""Plotting functions for optical setups and mode matching solutions and analyses."""

from dataclasses import dataclass
from io import BytesIO
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import (
    FillBetweenPolyCollection,
    LineCollection,
    PathCollection,
)
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter
from scipy import stats

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
    beam_ci: list[Line2D]  #: List of lines representing beam confidence intervals
    lenses: list[tuple[LineCollection, Annotation, Rectangle | None]]  #: List of lens plot elements
    handles: list[tuple[str, Any]]  #: List of plot handles by their legend labels
    r_max: float  #: Maximum beam radius in the plot


@dataclass(frozen=True)
class ModeMatchingPlot:
    """Plot of a mode matching solution with references to the plot elements."""

    ax: Axes  #: Axes containing the plot
    setup: OpticalSetupPlot  #: Plot of the actual optical setup
    desired_beam: OpticalSetupPlot  #: Plot of the desired beam profile
    ranges: list[Rectangle]  #: List of ranges in the plot
    apertures: list[Line2D]  #: List of aperture lines in the plot
    passages: list[Rectangle]  #: List of passage rectangles in the plot


def get_handles(ax: Axes) -> list[tuple[str, Any]]:
    """Get a list of plot handles by their legend labels.

    Args:
        ax: The axes to get the handles from.

    Returns:
        A list of tuples containing legend labels and their corresponding plot handles, empty if no legend exists.
    """

    legend = ax.get_legend()
    if legend is None:
        return []
    return [(text.get_text(), handle) for handle, text in zip(legend.legend_handles, legend.texts, strict=True)]


# TODO refactor this function to reduce complexity
def plot_setup(  # noqa: C901
    self: "OpticalSetup",
    *,
    points: int | np.ndarray | None = None,
    limits: tuple[float, float] | None = None,
    beam_kwargs: dict | None = None,
    confidence_interval: float | bool | None = None,
    rayleigh_range_cap: float | None = None,
    free_lenses: list[int] = [],  # noqa: B006
    ax: Axes | None = None,
    show_legend: bool | None = None,
    legend_loc: str | None = None,
    # TODO add back other configuration options?
) -> OpticalSetupPlot:
    """Plot the the beam profile and optical elements of the setup.

    Args:
        self: The optical setup instance.
        points: Number of points or specific z-coordinates to evaluate the beam profile.
            If `None`, this defaults to :attr:`Config.PlotSetup.beam_points <corset.config.Config.PlotSetup.beam_points>`.
        limits: Z-coordinate limits for the plot.
            If `None`, limits are determined from the beam and elements.
        beam_kwargs: Additional keyword arguments passed to the beam plot.
            If `None` this defaults to :attr:`Config.PlotSetup.beam_kwargs <corset.config.Config.PlotSetup.beam_kwargs>`.
        confidence_interval: Confidence interval probability for beam uncertainty visualization.
            If `False`, no uncertainty is plotted.
            If `None` this defaults to :attr:`Config.PlotSetup.confidence_interval <corset.config.Config.PlotSetup.confidence_interval>`.
        rayleigh_range_cap: Maximum Rayleigh range to consider when determining plot limits.
            If `None` this defaults to :attr:`Config.PlotSetup.rayleigh_range_cap <corset.config.Config.PlotSetup.rayleigh_range_cap>`.
        free_lenses: Indices of lenses to treat as free elements in the plot.
        ax: The axes to plot on. If `None`, the current axes are used.
        show_legend: Whether to show a legend for the plot.
            If `None`, this defaults to :attr:`Config.PlotSetup.show_legend <corset.config.Config.PlotSetup.show_legend>`.
        legend_loc: Location of the legend in the plot.
            If `None`, this defaults to :attr:`Config.PlotSetup.legend_loc <corset.config.Config.PlotSetup.legend_loc>`.

    Returns:
        An :class:`OpticalSetupPlot` containing references to the plot elements.
    """

    ax = ax or plt.gca()
    show_legend = Config.get(show_legend, Config.PlotSetup.show_legend)
    legend_loc = Config.get(legend_loc, Config.PlotSetup.legend_loc)
    beam_kwargs = Config.get(beam_kwargs, Config.PlotSetup.beam_kwargs)
    if confidence_interval is True:
        raise ValueError("confidence_interval cannot be True, must be a float between 0 and 1 or False")
    confidence_interval = Config.get(confidence_interval, Config.PlotSetup.confidence_interval)
    rayleigh_range_cap = Config.get(rayleigh_range_cap, Config.PlotSetup.rayleigh_range_cap)

    lens_positions = [pos for pos, _ in self.elements]

    if isinstance(points, np.ndarray):
        zs = points
    else:
        if not limits:
            cap = rayleigh_range_cap
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

        num_points = points if isinstance(points, int) else Config.PlotSetup.beam_points
        zs = np.linspace(limits[0], limits[1], num_points)

    handles = get_handles(ax)

    rs = cast(np.ndarray, self.radius(zs))
    fill_between = ax.fill_between(zs, -rs, rs, **{"zorder": 100, **beam_kwargs})
    beam_label = beam_kwargs.get("label", "Beam")
    handles.append((beam_label, fill_between))
    r_max = np.max(rs)

    beam_deviation = []
    if confidence_interval is not False and self.beams[0].gauss_cov is not None:
        rs_ci = self.radius_dev(zs) * stats.norm.interval(confidence_interval)[1]
        ci_color = beam_kwargs.get("color")
        if ci_color is None or ci_color == "none":
            ci_color = beam_kwargs.get("edgecolor")
        for r0, ci in product([-rs, rs], [-rs_ci, rs_ci]):
            line = ax.plot(zs, r0 + ci, ls="--", color=ci_color, alpha=beam_kwargs.get("alpha"), zorder=105)[0]
            beam_deviation.append(line)
        r_max = np.max(rs + rs_ci)
        handles.append((f"{round(confidence_interval*100)}% CI ({beam_label})", beam_deviation[0]))
    # TODO make beam plot function?

    # TODO factor out into plot lens function?
    lenses = []
    for i, (pos, lens) in enumerate(self.elements):
        color = "C2"
        zorder = 200

        line = ax.vlines(
            pos,
            -r_max * (1 + RELATIVE_MARGIN),
            r_max * (1 + RELATIVE_MARGIN),
            color=color,
            zorder=zorder,
        )

        label_text = lens.name if lens.name is not None else f"f={round(lens.focal_length*1e3)}mm"
        if i in free_lenses:
            label_text = f"$L_{i}$: {label_text} @{round(pos*1e3)}mm"
        label = ax.text(
            pos,
            -r_max * (1 + RELATIVE_MARGIN),
            label_text,
            va="center",
            ha="left",
            rotation="vertical",
            rotation_mode="anchor",
            bbox={"fc": plt.rcParams["axes.facecolor"], "ec": "none", "alpha": 0.5},
            zorder=zorder,
        )

        rect = None
        if lens.left_margin or lens.right_margin:
            rect = Rectangle(
                (pos - lens.left_margin, -r_max * (1 + RELATIVE_MARGIN)),
                lens.left_margin + lens.right_margin,
                2 * r_max * (1 + RELATIVE_MARGIN),
                fc="none",
                ec=color,
                ls="--",
                zorder=zorder,
            )
            ax.add_patch(rect)

        lenses.append((line, label, rect))

    if lenses and not any(label == "Lens" for label, _ in handles):
        # fake handle with vertical line marker
        handles.append(("Lens", Line2D([], [], color=color, label="Lens", marker="|", linestyle="None", markersize=10)))
    rectangles = [r for _, _, r in lenses if r is not None]
    if rectangles and not any(label == "Lens Margin" for label, _ in handles):
        handles.append(("Lens Margin", rectangles[0]))

    ax.set_ylim(
        min(-r_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[0]),
        max(r_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[1]),
    )

    ax.set_xlabel("z in mm")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(r"w(z) in $\mathrm{\mu m}$")
    ax.yaxis.set_major_formatter(micro_formatter)

    if show_legend:
        ax.legend([lbl for _, lbl in handles], [han for han, _ in handles], loc=legend_loc).set_zorder(1000)

    return OpticalSetupPlot(
        ax=ax,
        zs=zs,
        rs=rs,
        beam=fill_between,
        beam_ci=beam_deviation,
        lenses=lenses,
        handles=handles,
        r_max=r_max,
    )


def plot_mode_match_solution_setup(
    self: "ModeMatchingSolution",
    *,
    ax: Axes | None = None,
    show_legend: bool | None = None,
    legend_loc: str | None = None,
) -> ModeMatchingPlot:
    """Plot the mode matching solution setup including the desired beam and constraints.

    Args:
        self: The mode matching solution instance.
        ax: The axes to plot on. If `None`, the current axes are used.
        show_legend: Whether to show a legend for the plot.
            If `None`, this defaults to :attr:`Config.PlotSetup.show_legend <corset.config.Config.PlotSetup.show_legend>`.
        legend_loc: Location of the legend in the plot.
            If `None`, this defaults to :attr:`Config.PlotSetup.legend_loc <corset.config.Config.PlotSetup.legend_loc>`.
    Returns:
        An :class:`ModeMatchingPlot` containing references to the plot elements.
    """
    from .core import OpticalSetup
    from .solver import Aperture, Passage

    ax = ax or plt.gca()
    show_legend = Config.get(show_legend, Config.PlotSetup.show_legend)
    legend_loc = Config.get(legend_loc, Config.PlotSetup.legend_loc)

    problem = self.candidate.problem

    setup_plot = plot_setup(
        self.setup, ax=ax, free_lenses=self.candidate.parametrized_setup.free_elements, show_legend=show_legend
    )
    desired_plot = plot_setup(
        OpticalSetup(problem.desired_beam, []),
        ax=ax,
        beam_kwargs={"color": "C1", "label": "Desired Beam"},
        show_legend=show_legend,
    )

    handles = get_handles(ax)
    if show_legend:
        desired = handles.pop(-1)
        assert desired[0] == "Desired Beam"  # noqa: S101
        handles.insert(1, desired)  # hack to show desired beam 2nd in legend

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
    if ranges:
        handles.append(("Shifting Range", ranges[0]))

    apertures = []
    passages = []
    for con in problem.constraints:
        if isinstance(con, Aperture):
            apertures.extend(
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

    if apertures:
        # TODO add dots to vertical line?
        handles.append(  # fake handle with line marker
            ("Aperture", Line2D([], [], color="C4", label="Aperture", marker="|", linestyle="None", markersize=10))
        )
    if passages:
        handles.append(("Passage", passages[0]))

    if show_legend:
        ax.legend([lbl for _, lbl in handles], [han for han, _ in handles], loc=legend_loc).set_zorder(1000)

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
    center: PathCollection  #: Point representing the optimal focus and waist
    center_ci: Ellipse | None  #: Uncertainty of the arriving beam around the optimal point
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
    displacement: float | list[float] | None = None,
    num_samples: int | list[int] | None = None,
    line_step: int | list[int] | None = None,
    dimensions: list[int] | None = None,
    focus_range: tuple[float, float] | None = None,
    waist_range: tuple[float, float] | None = None,
    confidence_interval: float | None = None,
    grid_resolution: int | None = None,
    ax: Axes | None = None,
) -> ReachabilityPlot:
    """Plot a reachability analysis of the mode matching solution.

    Args:
        self: The mode matching solution instance.
        displacement: Maximum displacement from the optimal position(s) in meters.
            If `None`, this defaults to :attr:`Config.PlotReachability.displacement <corset.config.Config.PlotReachability.displacement>`.
        num_samples: Number of samples to take along each dimension.
            If `None`, this defaults to :attr:`Config.PlotReachability.num_samples <corset.config.Config.PlotReachability.num_samples>`.
        line_step: Step size to skip certain lines for increased smoothness while retaining clarity.
            If `None`, this defaults to :attr:`Config.PlotReachability.line_step <corset.config.Config.PlotReachability.line_step>`.
        dimensions: Indices of the dimensions to analyze. If `None`, all dimensions are used.
        focus_range: Range of focus values to display on the x-axis. If `None`, determined automatically.
        waist_range: Range of waist values to display on the y-axis. If `None`, determined automatically.
        confidence_interval: Confidence ellipse probability for focus position and waist radius.
            If `None`, this defaults to :attr:`Config.PlotReachability.confidence_interval <corset.config.Config.PlotReachability.confidence_interval>`.
        grid_resolution: Resolution of the grid for the background mode overlap contour plot.
            If `None`, this defaults to :attr:`Config.Overlap.grid_resolution <corset.config.Config.Overlap.grid_resolution>`.
        ax: The axes to plot on. If `None`, the current axes are used.

    Returns:
        A :class:`ReachabilityPlot` containing references to the plot elements.
    """

    from .analysis import vector_partial
    from .solver import mode_overlap

    ax = ax or plt.gca()
    displacement = Config.get(displacement, Config.PlotReachability.displacement)
    num_samples = Config.get(num_samples, Config.PlotReachability.num_samples)
    line_step = Config.get(line_step, Config.PlotReachability.line_step)
    confidence_interval = Config.get(confidence_interval, Config.PlotReachability.confidence_interval)
    grid_resolution = Config.get(grid_resolution, Config.Overlap.grid_resolution)

    if dimensions is None:
        dimensions = list(range(len(self.positions)))

    num_dof = len(dimensions)
    displacement = cast(list, [displacement] * num_dof if np.isscalar(displacement) else displacement)
    num_samples = cast(list, [num_samples] * num_dof if np.isscalar(num_samples) else num_samples)
    line_step = cast(list, [line_step] * num_dof if np.isscalar(line_step) else line_step)

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
        vector_partial(self.candidate.parametrized_focus_and_waist, self.positions, dimensions), signature="(n)->(2)"
    )

    grids = np.moveaxis(np.meshgrid(*linspaces, indexing="ij"), 0, -1)  # pyright: ignore[reportArgumentType]
    focuses, waists = np.moveaxis(focus_and_waist(grids), -1, 0)
    delta_focuses = focuses - self.candidate.problem.desired_beam.focus

    contrast_color = plt.rcParams["text.color"]
    center = ax.scatter(0, self.candidate.problem.desired_beam.waist, color=contrast_color, marker="o", zorder=100)
    center_ci = None
    if confidence_interval is not False and self.setup.beams[-1].gauss_cov is not None:
        # plot the confidence ellipse around the optimal point
        eigenvalues, eigenvectors = np.linalg.eigh(self.setup.beams[-1].gauss_cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        diameters = 2 * np.sqrt(eigenvalues) * stats.chi2.ppf(confidence_interval, df=2) ** 0.5
        desired_waist = self.candidate.problem.desired_beam.waist
        center_ci = Ellipse(
            (0, desired_waist), *diameters, angle=angle, fill=False, ls="--", color=contrast_color, zorder=100
        )
        ax.add_patch(center_ci)

    def select_lines(arr: np.ndarray, dim: int, steps: list[int]) -> np.ndarray:
        steps = [step if i != dim else 1 for i, step in enumerate(steps)]
        indices = tuple(slice(None, None, step) for step in steps)
        return np.moveaxis(arr[indices], dim, -1).reshape(-1, arr.shape[dim])

    lines: list[list[Line2D]] = []
    for i, (dim, disp) in enumerate(zip(dimensions, displacement, strict=True)):
        focuses_flat = select_lines(delta_focuses, i, line_step)  # pyright: ignore[reportArgumentType]
        waists_flat = select_lines(waists, i, line_step)  # pyright: ignore[reportArgumentType]
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

    ax.legend(handles=[line[0] for line in lines], loc="lower left").set_zorder(1000)

    focuses_grid, waists_grid = np.meshgrid(
        np.linspace(*ax.get_xlim(), grid_resolution), np.linspace(*ax.get_ylim(), grid_resolution)
    )
    overlap = mode_overlap(
        focuses_grid,  # pyright: ignore[reportArgumentType]
        self.candidate.problem.desired_beam.waist,
        waists_grid,  # pyright: ignore[reportArgumentType]
        self.candidate.problem.setup.initial_beam.wavelength,
    )

    levels = Config.Overlap.levels
    colors = Config.Overlap.colors()
    res = ax.contourf(focuses_grid, waists_grid, overlap * 100, levels=levels, colors=colors, alpha=0.5)
    cb = ax.figure.colorbar(res, label="Mode overlap (%)")

    jacobian = self.analysis.focus_and_waist_jacobian

    ax.set_xlabel(rf"$\Delta z_0$ in mm ($\nabla z_0$=[{' '.join(f'{x:.3f}' for x in jacobian[0])}])")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(rf"$w_0$ in $\mathrm{{\mu m}}$ ($\nabla w_0$=[{' '.join(f'{x:.3f}' for x in jacobian[1]*1e3)}]e-3)")
    ax.yaxis.set_major_formatter(micro_formatter)

    ax.set_title("Reachability Analysis")

    return ReachabilityPlot(ax=ax, center=center, center_ci=center_ci, lines=lines, contour=res, colorbar=cb)


def plot_sensitivity(
    self: "ModeMatchingSolution",
    *,
    dimensions: tuple[int, int] | tuple[int, int, int] | None = None,
    worst_overlap: float | None = None,
    x_displacement: float | None = None,
    y_displacement: float | None = None,
    z_displacement: float | None = None,
    num_samples_z: int | None = None,
    force_contour_lines: bool | None = None,
    grid_resolution: int | None = None,
    ax: Axes | None = None,
) -> SensitivityPlot:
    """Plot a sensitivity analysis of the mode matching solution.

    Args:
        self: The mode matching solution instance.
        dimensions: Indices of the dimensions to analyze. If `None`, the two least
            coupled dimensions are used as the x and y dimensions, and the remaining most sensitive
            dimension is used as the auxiliary z dimension if applicable.
        worst_overlap: Worst-case mode overlap contour line that should still be fully visible in the
            plot used for automatic range determination.
            If `None`, this defaults to :attr:`Config.PlotSensitivity.worst_overlap <corset.config.Config.PlotSensitivity.worst_overlap>`.
        x_displacement: Displacement magnitude for x-axis values. If `None`, determined automatically from worst_overlap.
        y_displacement: Displacement magnitude for y-axis values. If `None`, determined automatically from worst_overlap.
        z_displacement: Displacement magnitude for z-axis values. If `None`, determined automatically from worst_overlap.
        num_samples_z: Number of z-slices to plot if 3 dimensions are used.
            If `None`, this defaults to :attr:`Config.PlotSensitivity.num_samples_z <corset.config.Config.PlotSensitivity.num_samples_z>`.
        force_contour_lines: If `True`, always use contour lines instead of filled contours for plots with two
            degrees of freedom. If `None`, this defaults to :attr:`Config.PlotSensitivity.force_contour_lines <corset.config.Config.PlotSensitivity.force_contour_lines>`.
        grid_resolution: Resolution of the grid for the contour plots.
            If `None`, this defaults to :attr:`Config.Overlap.grid_resolution <corset.config.Config.Overlap.grid_resolution>`.
        ax: The axes to plot on. If `None`, the current axes are used.

    Returns:
        A :class:`SensitivityPlot` containing references to the plot elements.
    """

    from .analysis import vector_partial

    worst_overlap = Config.get(worst_overlap, Config.PlotSensitivity.worst_overlap)
    num_samples_z = Config.get(num_samples_z, Config.PlotSensitivity.num_samples_z)
    force_contour_lines = Config.get(force_contour_lines, Config.PlotSensitivity.force_contour_lines)
    grid_resolution = Config.get(grid_resolution, Config.Overlap.grid_resolution)
    ax = ax or plt.gca()

    if dimensions is None:
        dimensions = self.analysis.min_coupling_pair
        if len(self.positions) > 2:
            all_dims = set(range(len(self.positions)))
            remaining_dims = list(all_dims - set(dimensions))
            most_sensitive = max(remaining_dims, key=lambda dim: self.analysis.sensitivities[dim, dim])
            dimensions = (*dimensions, most_sensitive)

    mode_overlap = np.vectorize(
        vector_partial(
            self.candidate.parametrized_overlap, self.positions, dimensions  # pyright: ignore[reportArgumentType]
        ),
        signature="(n)->()",
    )

    # the hessian will always be badly conditioned for ndim > 2, so only use the subspace we care about
    # which should be well conditioned
    A_xy_inv = np.linalg.inv(-self.analysis.hessian[np.ix_(dimensions[:2], dimensions[:2])] / 2)  # noqa: N806
    # use Config.get so the type checker knows the displacements are not None after this point
    x_displacement = Config.get(x_displacement, np.sqrt(A_xy_inv[0, 0] * (1 - worst_overlap)) * 1.1)
    y_displacement = Config.get(y_displacement, np.sqrt(A_xy_inv[1, 1] * (1 - worst_overlap)) * 1.1)

    if z_displacement is None and len(dimensions) > 2:
        A_xyz = -self.analysis.hessian[np.ix_(dimensions, dimensions)]  # noqa: N806
        eigs = np.linalg.eigh(A_xyz)
        v_min = eigs.eigenvectors[:, np.argmin(eigs.eigenvalues)]
        z_displacement = min(abs(v_min[2] / v_min[0] * x_displacement), abs(v_min[2] / v_min[1] * y_displacement)) / 1.1

    base = np.array([self.positions[dim] for dim in dimensions])
    xs = np.linspace(-x_displacement, x_displacement, grid_resolution)
    ys = np.linspace(-y_displacement, y_displacement, grid_resolution)
    xsg, ysg = np.meshgrid(xs, ys)
    cmap = plt.get_cmap("coolwarm") if Config.mpl_is_dark() else plt.get_cmap("berlin")

    contours = []
    colorbar = None
    handles: list = []

    if len(dimensions) == 2:
        overlaps = mode_overlap(np.stack([xsg, ysg], axis=-1) + base)
        if force_contour_lines:
            cont = ax.contour(xsg, ysg, overlaps * 100, levels=Config.Overlap.levels, colors=cmap(0.5))
            ax.clabel(cont, fmt="%1.1f%%")
            contours.append(cont)
        else:
            cont = ax.contourf(xsg, ysg, overlaps * 100, levels=Config.Overlap.levels, colors=Config.Overlap.colors())
            colorbar = ax.figure.colorbar(cont, label="Mode overlap (%)")
            contours.append(cont)
    else:
        z_displacement = cast(float, z_displacement)  # make type checker happy
        z_colors = cmap(np.linspace(0, 1, num_samples_z))
        zs = np.linspace(-z_displacement, z_displacement, num_samples_z)
        for i, (color, z) in enumerate(zip(z_colors, zs, strict=True)):
            positions = np.stack([xsg, ysg, z * np.ones_like(xsg)], axis=-1) + base
            overlaps = mode_overlap(positions)
            zorder = 100 if i == num_samples_z // 2 else 10
            cont = ax.contour(
                xsg, ysg, overlaps * 100, levels=Config.Overlap.levels, colors=[color], alpha=0.7, zorder=zorder
            )
            contours.append(cont)
            if i == num_samples_z // 2:
                ax.clabel(cont, fmt="%1.1f%%", colors=[color])
        handles = [
            Line2D([], [], color=color, label=rf"$\Delta x_{{{dimensions[2]}}} = {value*1e3:.1f}$ mm")
            for color, value in zip(z_colors, zs, strict=True)
        ]
        ax.legend(handles=handles, loc="lower left").set_zorder(1000)

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
    figsize: tuple[float, float] | None = None,
    width_ratios: tuple[float, float, float] | None = None,
    tight_layout: bool | None = None,
    setup_kwargs: dict | None = None,
    reachability_kwargs: dict | None = None,
    sensitivity_kwargs: dict | None = None,
) -> tuple[Figure, tuple[ModeMatchingPlot, ReachabilityPlot, SensitivityPlot]]:
    """Plot the mode matching solution setup, reachability analysis, and sensitivity analysis in a single figure.

    Args:
        self: The mode matching solution instance.
        figsize: Figure size for the combined plot.
            If `None`, this defaults to :attr:`Config.PlotAll.figsize <corset.config.Config.PlotAll.figsize>`.
        width_ratios: Width ratios for the three subplots.
            If `None`, this defaults to :attr:`Config.PlotAll.width_ratios <corset.config.Config.PlotAll.width_ratios>`.
        tight_layout: Whether to use tight layout for the figure.
            If `None`, this defaults to :attr:`Config.PlotAll.tight_layout <corset.config.Config.PlotAll.tight_layout>`.
        setup_kwargs: Additional keyword arguments for the setup plot.
            If `None`, this defaults to :attr:`Config.PlotAll.setup_kwargs <corset.config.Config.PlotAll.setup_kwargs>`.
        reachability_kwargs: Additional keyword arguments for the reachability plot.
            If `None`, this defaults to :attr:`Config.PlotAll.reachability_kwargs <corset.config.Config.PlotAll.reachability_kwargs>`.
        sensitivity_kwargs: Additional keyword arguments for the sensitivity plot.
            If `None`, this defaults to :attr:`Config.PlotAll.sensitivity_kwargs <corset.config.Config.PlotAll.sensitivity_kwargs>`.

    Returns:
        A tuple containing the figure and an inner tuple with the three plot objects.
    """

    figsize = Config.get(figsize, Config.PlotAll.figsize)
    width_ratios = Config.get(width_ratios, Config.PlotAll.width_ratios)
    tight_layout = Config.get(tight_layout, Config.PlotAll.tight_layout)
    setup_kwargs = Config.get(setup_kwargs, Config.PlotAll.setup_kwargs)
    reachability_kwargs = Config.get(reachability_kwargs, Config.PlotAll.reachability_kwargs)
    sensitivity_kwargs = Config.get(sensitivity_kwargs, Config.PlotAll.sensitivity_kwargs)

    fig, (axl, axr, axc) = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={"width_ratios": width_ratios}, tight_layout=tight_layout
    )
    sol_plot = self.plot_setup(ax=axl, **setup_kwargs)
    reach_plot = self.plot_reachability(ax=axr, **reachability_kwargs)
    sens_plot = self.plot_sensitivity(ax=axc, **sensitivity_kwargs)
    return fig, (sol_plot, reach_plot, sens_plot)
