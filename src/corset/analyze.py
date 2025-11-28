import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines
from scipy import linalg as sp_linalg
from scipy.differentiate import hessian

from .config import Config
from .plot import micro_formatter, milli_formatter

if TYPE_CHECKING:
    from .solver import ModeMatchSolution


def wrap_for_hessian(
    func: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], np.ndarray]:
    @wraps(func)
    def wrapped(x: np.ndarray) -> np.ndarray:
        inputs = np.moveaxis(x, 0, -1)
        raw_res = [func(inp) for inp in inputs.reshape(-1, x.shape[0])]
        return np.reshape(raw_res, x.shape[1:])

    return wrapped


def vector_partial(
    func: Callable[[np.ndarray], Any], default: np.ndarray, dims: Iterable[int]
) -> Callable[[np.ndarray], np.ndarray]:
    default = default.copy()

    @wraps(func)
    def wrapped(var: np.ndarray) -> np.ndarray:
        full_input = default.copy()
        for i, dim in enumerate(dims):
            full_input[dim] = var[i]
        return func(full_input)

    return wrapped


def make_mode_overlap(solution: "ModeMatchSolution") -> Callable[[np.ndarray], float]:
    from . import solver

    def overlap(positions: np.ndarray) -> float:
        setup = solution.candidate.parametrized_setup.substitute(positions)
        final_beam = setup.beams[-1]
        problem = solution.candidate.problem
        return solver.mode_overlap(
            final_beam.waist,
            problem.desired_beam.waist,
            final_beam.focus - problem.desired_beam.focus,
            problem.setup.initial_beam.wavelength,
        )

    return overlap


def make_focus_and_waist(solution: "ModeMatchSolution") -> Callable[[np.ndarray], np.ndarray]:
    def focus_and_waist(positions: np.ndarray) -> np.ndarray:
        setup = solution.candidate.parametrized_setup.substitute(positions)
        final_beam = setup.beams[-1]
        return np.array([final_beam.focus, final_beam.waist])

    return focus_and_waist


@dataclass(frozen=True)
class SensitivityAnalysis:
    solution: "ModeMatchSolution"

    @cached_property
    def hessian(self) -> np.ndarray:
        mode_overlap = wrap_for_hessian(make_mode_overlap(self.solution))

        # the default initial step is 0.5 which would lead to invalid lens position
        # 1e-2 ensures, that only physical configurations are evaluated
        hess_res = hessian(mode_overlap, self.solution.positions, initial_step=1e-2)
        if np.any(hess_res.status != 0):
            warnings.warn(f"Hessian calculation did not converge: {hess_res.status}", stacklevel=2)
        print(hess_res.ddf)

        return hess_res.ddf

    @cached_property
    def couplings(self) -> np.ndarray:
        normalizer = 1 / np.sqrt(-np.diag(self.hessian))
        return -self.hessian * np.outer(normalizer, normalizer)

    # TODO are these reasonable units?
    @cached_property
    def sensitivities(self) -> np.ndarray:
        return -self.hessian / 2

    # TODO should this be in terms of couplings or sensitivities?
    @cached_property
    def min_coup_pair(self) -> tuple[int, int]:
        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_couplings = np.abs(self.couplings[indices])
        best = np.argmin(abs_couplings)
        return (int(indices[0][best]), int(indices[1][best]))

    @cached_property
    def min_sens_pair(self) -> tuple[int, int]:
        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_sensitivities = np.abs(self.sensitivities[indices])
        best = np.argmin(abs_sensitivities)
        return (int(indices[0][best]), int(indices[1][best]))

    # the vectors spanning the sub space in which the mode overlap stays constant
    # equivalent to the null space of the hessian
    @cached_property
    def const_space(self) -> np.ndarray:
        null_space = sp_linalg.null_space(self.hessian, rcond=1e-5)
        if null_space.shape[1] != len(self.solution.positions) - 2:
            warnings.warn("Constancy space does not have expected dimension. Results may be inaccurate.", stacklevel=2)
        return null_space


def plot_reachability(
    solution: ModeMatchSolution,
    displacement: float | list[float] = 5e-3,
    num_samples: int | list[int] = 7,
    grid_step: int | list[int] = 2,
    dimensions: list[int] | None = None,
    focus_range: tuple[float, float] | None = None,
    waist_range: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,  # pyright: ignore[reportPrivateImportUsage]
) -> None:
    # TODO add reachability analysis plot class
    from .solver import mode_overlap

    ax = ax or plt.gca()
    if dimensions is None:
        dimensions = list(range(len(solution.positions)))

    num_dof = len(dimensions)
    displacement = cast(list, [displacement] * num_dof if np.isscalar(displacement) else displacement)
    num_samples = cast(list, [num_samples] * num_dof if np.isscalar(num_samples) else num_samples)
    grid_step = cast(list, [grid_step] * num_dof if np.isscalar(grid_step) else grid_step)

    linspaces = [
        np.linspace(-d, d, num=n) + offset
        for d, n, offset in zip(
            displacement, num_samples, solution.positions, strict=True  # pyright: ignore[reportArgumentType]
        )
    ]

    if focus_range is not None:
        ax.set_xlim(focus_range)

    if waist_range is not None:
        ax.set_ylim(waist_range)

    focus_and_waist = np.vectorize(
        vector_partial(make_focus_and_waist(solution), solution.positions, dimensions), signature="(n)->(2)"
    )

    grids = np.moveaxis(np.meshgrid(*linspaces, indexing="ij"), 0, -1)  # pyright: ignore[reportArgumentType]
    focuses, waists = np.moveaxis(focus_and_waist(grids), -1, 0)
    delta_focuses = focuses - solution.candidate.problem.desired_beam.focus

    artists: list = [
        ax.scatter(0, solution.candidate.problem.desired_beam.waist, color="k", marker="o", label="Desired", zorder=100)
    ]

    def select_lines(arr: np.ndarray, dim: int, steps: list[int]) -> np.ndarray:
        steps = [step if i != dim else 1 for i, step in enumerate(steps)]
        indices = tuple(slice(None, None, step) for step in steps)
        return np.moveaxis(arr[indices], dim, -1).reshape(-1, arr.shape[dim])

    for i, (dim, disp) in enumerate(zip(dimensions, displacement, strict=True)):
        focuses_flat = select_lines(delta_focuses, i, grid_step)  # pyright: ignore[reportArgumentType]
        waists_flat = select_lines(waists, i, grid_step)  # pyright: ignore[reportArgumentType]
        for focus, waist in zip(focuses_flat, waists_flat, strict=True):
            line = ax.plot(focus, waist, color=f"C{dim}", label=rf"$\Delta x_{i}$ ($\pm{disp*1e3:.1f}$ mm)")[0]
        artists.append(line)
        ax.annotate(
            "",
            xy=(focuses_flat[0][1], waists_flat[0][1]),
            xytext=(focuses_flat[0][0], waists_flat[0][0]),
            arrowprops={"color": f"C{i}", "headwidth": 10, "width": 2},
        )

    ax.legend(handles=artists)

    grid_n = 50
    focuses_grid, waists_grid = np.meshgrid(np.linspace(*ax.get_xlim(), grid_n), np.linspace(*ax.get_ylim(), grid_n))
    overlap = mode_overlap(
        solution.candidate.problem.desired_beam.waist,
        waists_grid,  # pyright: ignore[reportArgumentType]
        focuses_grid,  # pyright: ignore[reportArgumentType]
        solution.candidate.problem.setup.initial_beam.wavelength,
    )

    levels = Config.OVERLAP_LEVELS
    colors = Config.overlap_colors()
    res = ax.contourf(focuses_grid, waists_grid, overlap * 100, levels=levels, colors=colors, alpha=0.5)
    ax.figure.colorbar(res, label="Mode overlap (%)")

    ax.set_xlabel(r"Focus Displacement in mm")
    ax.xaxis.set_major_formatter(milli_formatter)

    ax.set_ylabel(r"Waist Size in $\mathrm{\mu m}$")
    ax.yaxis.set_major_formatter(micro_formatter)

    ax.set_title("Reachability Analysis")


def plot_sensitivity(
    solution: "ModeMatchSolution",
    dimensions: tuple[int, int] | tuple[int, int, int] | None = None,
    worst_overlap: float = 0.95,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    z_n: int = 3,  # more is too cluttered most times
    force_contours: bool = False,
    ax: plt.Axes | None = None,  # pyright: ignore[reportPrivateImportUsage]
) -> None:
    ax = ax or plt.gca()

    if dimensions is None:
        dimensions = solution.analysis.min_coup_pair
        if len(solution.positions) > 2:
            all_dims = set(range(len(solution.positions)))
            remaining_dims = list(all_dims - set(dimensions))
            most_sensitive = max(remaining_dims, key=lambda dim: solution.analysis.sensitivities[dim, dim])
            dimensions = (*dimensions, most_sensitive)

    mode_overlap = np.vectorize(
        vector_partial(make_mode_overlap(solution), solution.positions, dimensions), signature="(n)->()"
    )

    # the hessian will always be badly conditioned for ndim > 2, so only use the subspace we care about
    # which should be well conditioned
    A_xy_inv = np.linalg.inv(-solution.analysis.hessian[np.ix_(dimensions[:2], dimensions[:2])] / 2)  # noqa: N806
    if x_range is None:
        abs_max_range_x = np.sqrt(A_xy_inv[0, 0] * (1 - worst_overlap))
        x_range = (-abs_max_range_x * 1.1, abs_max_range_x * 1.1)
    if y_range is None:
        abs_max_range_y = np.sqrt(A_xy_inv[1, 1] * (1 - worst_overlap))
        y_range = (-abs_max_range_y * 1.1, abs_max_range_y * 1.1)

    if len(dimensions) > 2:
        A_xyz = -solution.analysis.hessian[np.ix_(dimensions, dimensions)]  # noqa: N806
        eigs = np.linalg.eigh(A_xyz)
        v_min = eigs.eigenvectors[:, np.argmin(eigs.eigenvalues)]
        abs_max_range_z = min(abs(v_min[2] / v_min[0] * abs_max_range_x), abs(v_min[2] / v_min[1] * abs_max_range_y))
        z_range = (-abs_max_range_z, abs_max_range_z)

    grid_n = 50
    base = np.array([solution.positions[dim] for dim in dimensions])
    xs = np.linspace(*x_range, grid_n)
    ys = np.linspace(*y_range, grid_n)
    xsg, ysg = np.meshgrid(xs, ys)
    cmap = plt.get_cmap("berlin")
    if len(dimensions) == 2:
        overlaps = mode_overlap(np.stack([xsg, ysg], axis=-1) + base)
        if force_contours:
            cont = ax.contour(xsg, ysg, overlaps * 100, levels=Config.OVERLAP_LEVELS, colors=cmap(0.5))
            ax.clabel(cont, fmt="%1.1f%%")
        else:
            cont = ax.contourf(xsg, ysg, overlaps * 100, levels=Config.OVERLAP_LEVELS, colors=Config.overlap_colors())
            ax.figure.colorbar(cont, label="Mode overlap (%)")
    else:
        z_colors = cmap(np.linspace(0, 1, z_n))
        zs = np.linspace(*z_range, z_n)  # pyright: ignore[reportCallIssue]
        for i, (color, z) in enumerate(zip(z_colors, zs, strict=True)):
            positions = np.stack([xsg, ysg, z * np.ones_like(xsg)], axis=-1) + base
            overlaps = mode_overlap(positions)
            zorder = 100 if i == z_n // 2 else 10
            res = ax.contour(
                xsg, ysg, overlaps * 100, levels=Config.OVERLAP_LEVELS, colors=[color], alpha=0.7, zorder=zorder
            )
            if i == z_n // 2:
                ax.clabel(res, fmt="%1.1f%%", colors=[color])
        handles = [
            lines.Line2D([], [], color=color, label=rf"$\Delta x_{{{dimensions[2]}}} = {value*1e3:.1f}$ mm")
            for color, value in zip(z_colors, zs, strict=True)
        ]
        ax.legend(handles=handles).set_zorder(1000)

    ax.set_xlabel(rf"$\Delta x_{{{dimensions[0]}}}$ in mm")
    ax.xaxis.set_major_formatter(milli_formatter)
    ax.set_ylabel(rf"$\Delta x_{{{dimensions[1]}}}$ in mm")
    ax.yaxis.set_major_formatter(milli_formatter)

    ax.set_title("Sensitivity Analysis")
