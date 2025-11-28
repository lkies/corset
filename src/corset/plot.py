from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection, LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib.ticker import FuncFormatter

# prevent circular import for type annotation in function signature
if TYPE_CHECKING:
    from .core import OpticalSetup
    from .solver import ModeMatchSolution

RELATIVE_MARGIN = 0.1

milli_formatter = FuncFormatter(lambda x, _: f"{x*1e3:.0f}")
micro_formatter = FuncFormatter(lambda x, _: f"{x*1e6:.0f}")


def dilute_color(color: str, alpha: float) -> str:
    mixed = np.array(colors.to_rgb(color)) * alpha + np.array([1, 1, 1]) * (1 - alpha)
    return colors.to_hex(mixed)  # pyright: ignore[reportArgumentType]


@dataclass(frozen=True)
class OpticalSetupPlot:
    ax: Axes
    z: np.ndarray
    w: np.ndarray
    beam: FillBetweenPolyCollection
    lenses: list[tuple[LineCollection, Annotation]]

    @cached_property
    def w_max(self) -> float:
        return np.max(self.w)

    # TODO put back into plot optical setup function?


@dataclass(frozen=True)
class ModeMatchingPlot:
    ax: Axes
    setup: OpticalSetupPlot
    desired_beam: OpticalSetupPlot
    regions: list[Rectangle]
    apertures: list[Line2D]
    passages: list[Rectangle]

    # apertures


def plot_optical_setup(
    self: "OpticalSetup",
    *,
    ax: Axes | None = None,
    points: None | int | np.ndarray = None,
    limits: tuple[float, float] | None = None,
    beam_kwargs: dict = {"color": "C0", "alpha": 0.5},  # noqa: B006
    free_lenses: list[int] = [],  # noqa: B006
    # TODO add back other configuration options?
) -> OpticalSetupPlot:
    ax = ax or plt.gca()

    lens_positions = [pos for pos, _ in self.elements]

    if isinstance(points, np.ndarray):
        zs = points
    else:
        if not limits:
            all_bounds = [
                self.beams[0].focus - self.beams[0].rayleigh_range,
                *lens_positions,
                self.beams[-1].focus + self.beams[0].rayleigh_range,
            ]
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
            label_text = f"$L_{i}$: " + label_text
        label = ax.annotate(label_text, xy=(pos, w_max * (1 + 2 * RELATIVE_MARGIN)), va="center", ha="center")
        lenses.append((lens_col, label))

    ax.set_ylim(
        min(-w_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[0]),
        max(w_max * (1 + 3 * RELATIVE_MARGIN), ax.get_ylim()[1]),
    )

    ax.set_xlabel("z in mm")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(r"w(z) in $\mathrm{\mu m}$")
    ax.yaxis.set_major_formatter(micro_formatter)

    return OpticalSetupPlot(ax=ax, z=zs, w=rs, beam=fb_col, lenses=lenses)  # pyright: ignore[reportArgumentType]


def plot_mode_match_solution(self: "ModeMatchSolution", ax: Axes | None = None) -> ModeMatchingPlot:
    from .core import OpticalSetup
    from .solver import Aperture, Passage

    ax = ax or plt.gca()

    problem = self.candidate.problem

    setup_plot = plot_optical_setup(self.setup, ax=ax, free_lenses=self.candidate.parametrized_setup.free_elements)
    desired_plot = plot_optical_setup(OpticalSetup(problem.desired_beam, []), ax=ax, beam_kwargs={"color": "C1"})

    regions = []
    for region in problem.regions:
        # TODO color configuratbility?
        rectangle = Rectangle(
            (region.left, -setup_plot.w_max * (1 + 2 * RELATIVE_MARGIN)),
            region.right - region.left,
            2 * setup_plot.w_max * (1 + 2 * RELATIVE_MARGIN),
            fc="none",
            ec="lightgrey",
            ls="--",
            zorder=50,  # TODO
        )
        regions.append(ax.add_patch(rectangle))

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
        regions=regions,
        apertures=apertures,
        passages=passages,
    )
