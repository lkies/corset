"""Mode matching solver and related data structures.

The main entry point is the function :func:`mode_match`, it will setup
the :class:`ModeMatchingProblem`.
The problem defined with the help of :class:`ShiftingRange`, :class:`Aperture`, and :class:`Passage` instances.
The constructed :class:`ModeMatchingProblem` yields :class:`ModeMatchingCandidate` instances
that are then optimized to produce :class:`ModeMatchingSolution` instances.
Each step containing a reference to the data structure it is based on.
The actual constrained optimization is carried out using :func:`scipy.optimize.minimize` using the SLSQP method.
All solutions will then be collected in a :class:`SolutionList` for convenient analysis and filtering.
"""

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations_with_replacement, pairwise
from typing import Any, cast, overload

import numpy as np
import pandas as pd
from scipy import optimize

from .analysis import ModeMatchingAnalysis
from .core import Beam, Lens, OpticalSetup, ThickLens, ThinLens
from .plot import (
    fig_to_png,
    plot_mode_match_solution_all,
    plot_mode_match_solution_setup,
    plot_reachability,
    plot_sensitivity,
)


@dataclass(frozen=True)
class ShiftingRange:
    """Range where lenses can be placed."""

    left: float  #: Range left boundary
    right: float  #: Range right boundary
    min_elements: int = 0  #: Minimum number of lenses to be placed in this range
    max_elements: int = float("inf")  # pyright: ignore[reportAssignmentType]
    """Maximum number of lenses that can be placed in this range."""
    selection: list[Lens] = field(default_factory=list)  # TODO name
    """Optional set of lenses to use for this range, if empty the global selection is used."""

    def __post_init__(self):
        if self.right <= self.left:
            raise ValueError("Range right boundary must be greater than left boundary")
        if self.min_elements < 0:
            raise ValueError("min_elements cannot be negative")
        if self.max_elements < self.min_elements:
            raise ValueError("max_elements cannot be less than min_elements")


@dataclass(frozen=True)
class ParametrizedSetup:
    """Parametrized optical setup where some or all element positions are free parameters."""

    initial_beam: Beam  #: Initial beam before left most element
    elements: list[tuple[float | None, Lens]]
    """Optical elements as (position | None, element) tuples. None represents a free element / parameter."""

    def substitute(self, positions: list[float] | np.ndarray, validate: bool = True) -> OpticalSetup:
        """Substitute free parameters with given positions.

        Args:
            positions: Positions to substitute for free parameters in order, must match number of free parameters.
            validate: Whether to validate the positions in the resulting setup.

        Returns:
            The resulting optical setup with substituted positions.

        """
        substituted_elements = []
        pos_index = 0
        try:
            for pos, element in self.elements:
                if pos is None:
                    pos = positions[pos_index]
                    pos_index += 1
                substituted_elements.append((pos, element))
        except IndexError as e:
            raise ValueError("Not enough positions provided to substitute all parametrized elements") from e
        if pos_index < len(positions):
            raise ValueError("Too many positions provided for the number of parametrized elements")
        return OpticalSetup(self.initial_beam, substituted_elements, validate)

    @cached_property
    def free_elements(self) -> list[int]:
        """List of indices of free elements (positions that are None)."""
        return [i for i, (pos, _) in enumerate(self.elements) if pos is None]


@dataclass(frozen=True)
class Aperture:
    """Aperture constraint for the optical setup."""

    position: float  #: Axial position of the aperture
    radius: float  #: Aperture radius, i.e. maximum beam radius at this position.

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Aperture radius must be positive")

    @cached_property
    def apertures(self) -> tuple["Aperture"]:
        """A tuple containing this aperture. Used for uniform interface with :class:`Passage`."""
        return (self,)


@dataclass(frozen=True)
class Passage:
    left: float  #: Left boundary of the passage
    right: float  #: Right boundary of the passage
    radius: float  #: Passage radius, i.e. maximum beam radius within the passage

    def __post_init__(self):
        if self.right <= self.left:
            raise ValueError("Left boundary must be less than right boundary")
        if self.radius <= 0:
            raise ValueError("Passage radius must be positive")

    @classmethod
    def centered(cls, center: float, width: float, radius: float) -> "Passage":
        """Convenience constructor for a passage centered at a given position.

        Args:
            center: Center position of the passage.
            width: Width of the passage.
            radius: Passage radius.

        Returns:
            Passage instance centered at the given position.
        """
        half_width = width / 2
        return cls(left=center - half_width, right=center + half_width, radius=radius)

    @cached_property
    def apertures(self) -> tuple[Aperture, Aperture]:
        """Representation of the constraint as two :class:`Aperture` constraints at the passage boundaries."""
        return (Aperture(self.left, self.radius), Aperture(self.right, self.radius))


# TODO should this be a member of Beam?
def mode_overlap(delta_z: float, waist_a: float, waist_b: float, wavelength: float) -> float:
    """Compute the mode overlap between two Gaussian beams.

    Args:
        delta_z: Axial distance between the beam waists.
        waist_a: Waist radius of the first beam.
        waist_b: Waist radius of the second beam.
        wavelength: Wavelength of the beams.

    Returns:
        Mode overlap between the two beams.
    """
    return (
        2 * np.pi * waist_a * waist_b / np.sqrt(wavelength**2 * delta_z**2 + np.pi**2 * (waist_a**2 + waist_b**2) ** 2)
    )


@dataclass(frozen=True)
class ModeMatchingProblem:
    """Mode matching problem definition."""

    setup: OpticalSetup  #: Initial optical setup
    desired_beam: Beam  #: Desired output beam after the optical setup
    ranges: list[ShiftingRange]  #: Ranges where lenses can be placed
    selection: Sequence[Lens]  #: Selection of lenses to choose from for mode matching
    min_elements: int  #: Minimum number of elements to use
    max_elements: int  #: Maximum number of elements to use
    constraints: Sequence[Aperture | Passage]  #: Beam constraints on the optical setup
    # TODO make it so that the order of evaluating the candidates does not change their random values
    rng: np.random.Generator  #: Seeded random number generator for reproducibility

    # TODO maybe also verify that there is a combination of lenses that can fit in the ranges?
    def __post_init__(self):
        self._verify_selection()
        self._verify_no_overlaps()
        if self.setup.initial_beam.wavelength != self.desired_beam.wavelength:
            # testing for equality of floats should be fine here since they should come from the same source
            raise ValueError("Setup initial beam and desired beam must have the same wavelength")

    def _verify_selection(self):
        if self.min_elements < 1:
            raise ValueError("Global min_elements must be at least 1")

        if self.max_elements is not None and self.max_elements < self.min_elements:
            raise ValueError("Global max_elements cannot be less than min_elements")

        total_min = sum(r.min_elements for r in self.ranges)
        total_max = sum(r.max_elements for r in self.ranges)

        if total_max == float("inf") and self.max_elements == float("inf"):
            raise ValueError("Cannot have unbounded maximum elements when global maximum is not set")

        if total_min > self.max_elements:
            raise ValueError("Sum of range minimum elements exceeds global maximum elements")
        if total_max < self.min_elements:
            raise ValueError("Sum of range maximum elements is less than global minimum elements")

    def _verify_no_overlaps(self):
        regions: list[tuple[float, float, Any]] = []  # left right
        regions.extend((pos, pos, element) for pos, element in self.setup.elements)
        regions.extend((range_.left, range_.right, range_) for range_ in self.ranges)
        regions.extend((ap.position, ap.position, ap) for ap in self.aperture_constraints if isinstance(ap, Aperture))
        regions.extend((pas.left, pas.right, pas) for pas in self.constraints if isinstance(pas, Passage))
        regions.sort(key=lambda x: x[0])
        for r1, r2 in pairwise(regions):
            if r1[1] > r2[0]:
                raise ValueError(f"Overlapping regions/elements detected: {r1[2]} and {r2[2]}")

    @cached_property
    def aperture_constraints(self) -> list[Aperture]:
        """List of all constraints as :class:`Aperture` constraints.
        :class:`Passage` constraints are converted to two :class:`Aperture` constraints."""
        return [aperture for constraint in self.constraints for aperture in constraint.apertures]

    @cached_property
    def interleaved_elements(self) -> list[tuple[float, Lens] | int]:
        """List of all (potential) elements, ranges where elements can be placed are represented
        by their index instead of the (position, element) tuple. Used for easy construction of parametrized setups."""
        merged: list[tuple[float, Lens] | int] = []
        next_boundary = self.ranges[0].left if self.ranges else float("inf")
        range_index = 0

        for pos, element in self.setup.elements:
            while pos > next_boundary:
                merged.append(range_index)
                range_index += 1
                next_boundary = self.ranges[range_index].left if range_index < len(self.ranges) else float("inf")
            merged.append((pos, element))

        while range_index < len(self.ranges):
            merged.append(range_index)
            range_index += 1

        return merged

    @classmethod
    def lens_combinations(
        cls,
        ranges: list[ShiftingRange],
        base_selection: Sequence[Lens],
        min_elements: int,
        max_elements: int,
        current_populations: list[tuple[Lens, ...]] = [],  # noqa: B006
    ) -> Generator[list[tuple[Lens, ...]], None, None]:
        """Recursive helper function to generate all possible lens populations for the given ranges.

        Args:
            ranges: Remaining ranges to process.
            base_selection: Global selection of lenses to choose from.
            min_elements: Minimum number of elements remaining to place.
            max_elements: Maximum number of elements remaining to place.
            current_populations: Current lens populations for the processed ranges.

        Yields:
            list[tuple[Lens, ...]]: The next lens populations for all ranges.
        """

        if not ranges:
            if min_elements <= 0:
                yield current_populations
            return

        first, *rest = ranges

        if first.min_elements > max_elements:
            return

        for num_elements in range(first.min_elements, min(first.max_elements, max_elements) + 1):
            selection = first.selection if first.selection else base_selection
            for comb in combinations_with_replacement(selection, num_elements):
                new_setup = [*current_populations, comb]
                yield from cls.lens_combinations(
                    rest, base_selection, min_elements - num_elements, max_elements - num_elements, new_setup
                )

    def candidates(self) -> Generator["ModeMatchingCandidate", None, None]:
        """Generate all possible range population candidates for the mode matching problem.

        Returns:
            Generator of :class:`ModeMatchingCandidate` instances.
        """
        for population in self.lens_combinations(self.ranges, self.selection, self.min_elements, self.max_elements):
            yield ModeMatchingCandidate(problem=self, populations=population)


@dataclass(frozen=True)
class ModeMatchingCandidate:
    """A candidate lens population for a mode matching problem."""

    problem: ModeMatchingProblem  #: The parent mode matching problem
    populations: list[tuple[Lens, ...]]  #: Lens populations for each range

    # TODO seeding?
    def generate_initial_positions(self, random: bool) -> np.ndarray:
        """Generate a set of initial positions for free lenses of the candidate.

        Args:
            random: Whether to generate random initial positions or evenly spaced positions.

        Returns:
            Array of initial positions for the lenses.
        """
        positions = []
        for range_, population in zip(self.problem.ranges, self.populations, strict=True):
            if not population:
                continue
            total_margin = sum(lens.left_margin + lens.right_margin for lens in population)
            available_space = range_.right - range_.left - total_margin
            if available_space < 0:
                raise ValueError("Not enough space in range for the lenses with their margins")

            if random:
                distances = np.diff(np.sort(self.problem.rng.uniform(0, available_space, len(population))), prepend=0)
            else:
                distances = np.repeat(available_space / (len(population) + 1), len(population))

            current_pos = range_.left
            for lens, distance in zip(population, distances, strict=True):
                current_pos += lens.left_margin + distance
                positions.append(current_pos)
                current_pos += lens.right_margin

        return np.array(positions)

    @cached_property
    def parametrized_setup(self) -> ParametrizedSetup:
        """Parametrized optical setup for the candidate."""
        elements = []
        for elem in self.problem.interleaved_elements:
            if isinstance(elem, int):
                for lens in self.populations[elem]:
                    elements.append((None, lens))
            else:
                elements.append(elem)
        return ParametrizedSetup(self.problem.setup.initial_beam, elements)

    @cached_property
    def position_constraint(self) -> optimize.LinearConstraint:
        """Linear constraint ensuring lenses do not overlap, stay within ranges and do not swap order."""
        constraints: list[tuple[np.ndarray, float, float]] = []
        pop_sizes = [len(pop) for pop in self.populations]
        index_offsets = np.cumsum([0, *pop_sizes[:-1]])
        mask = np.identity(sum(pop_sizes))
        for range_, population, base_idx in zip(self.problem.ranges, self.populations, index_offsets, strict=True):
            if not population:
                continue
            if len(population) == 1:
                constraints.append(
                    (mask[base_idx], range_.left + population[0].left_margin, range_.right - population[0].right_margin)
                )
            else:
                right_index = base_idx + len(population) - 1
                constraints.append((mask[base_idx], range_.left + population[0].left_margin, np.inf))
                constraints.append((mask[right_index], -np.inf, range_.right - population[-1].right_margin))

            for i in range(len(population) - 1):
                left_lens = population[i]
                right_lens = population[i + 1]
                constraints.append(
                    (
                        mask[base_idx + i + 1] - mask[base_idx + i],
                        left_lens.right_margin + right_lens.left_margin,
                        np.inf,
                    )
                )
        cols, lb, ub = zip(*constraints, strict=True)
        return optimize.LinearConstraint(
            np.vstack(cols), np.array(lb), np.array(ub)  # pyright: ignore[reportArgumentType]
        )

    @cached_property
    def beam_constraint(self) -> optimize.NonlinearConstraint:
        """Nonlinear constraint to ensure beam radius is within aperture constraints."""
        zs, rs = np.transpose([(c.position, c.radius) for c in self.problem.aperture_constraints])
        return optimize.NonlinearConstraint(
            lambda x, zs=zs, rs=rs, s=self.parametrized_setup: s.substitute(x, validate=False).radius(zs) / rs, 0, 1
        )

    @cached_property
    def constraints(self) -> list[optimize.NonlinearConstraint | optimize.LinearConstraint]:
        """List of position constraints and beam constraints if there are any."""
        return [self.position_constraint] + ([self.beam_constraint] if self.problem.constraints else [])

    def optimize(
        self,
        filter_pred: Callable[["ModeMatchingSolution"], bool],
        random_initial_positions: int,
        equal_setup_tol: float,
        solution_per_population: int,
        # optimize_coupling: bool,
    ) -> list["ModeMatchingSolution"]:
        """Optimize the candidate to find mode matching solutions.

        Args:
            filter_pred: Predicate function that must return True for a solution to be accepted.
            random_initial_positions: Number of random initial positions to generate in addition to the equally spaced one.
            equal_setup_tol: Tolerance for considering two solutions as equal for eliminating duplicates.
            solution_per_population: Maximum number of solutions to optimize and return.

        Returns:
            List of optimized mode matching solutions for the candidate.
        """
        initial_setups = [self.generate_initial_positions(random=False)]
        for _ in range(random_initial_positions):
            initial_setups.append(self.generate_initial_positions(random=True))

        # TODO unify this with the make_mode_overlap function in analyze.py?
        def objective(positions: np.ndarray) -> float:
            setup = self.parametrized_setup.substitute(positions, validate=False)  # pyright: ignore[reportArgumentType]
            initial_beam = self.problem.setup.initial_beam
            final_beam = Beam(
                setup.beam_parameters[-1], z_offset=setup.elements[-1][0], wavelength=initial_beam.wavelength
            )
            desired_beam = self.problem.desired_beam
            return -mode_overlap(
                final_beam.focus - desired_beam.focus,
                final_beam.waist,
                desired_beam.waist,
                final_beam.wavelength,  # pyright: ignore[reportArgumentType]
            )

        solutions = []
        solution_positions = []  # for this lens population
        constrained_initial_setups = []
        for x0 in initial_setups:
            if len(solutions) >= solution_per_population:
                break

            if self.problem.constraints:
                constrain_res = optimize.minimize(
                    lambda x: np.max(self.beam_constraint.fun(x)),
                    x0,
                    constraints=[self.position_constraint],
                    method="SLSQP",
                    options={"ftol": 2e-1},
                )
                if not np.all(self.beam_constraint.fun(constrain_res.x) <= self.beam_constraint.ub) or any(
                    not np.allclose(constrain_res.x, x0) for x0 in constrained_initial_setups
                ):
                    # failed to converge to a feasible setup or already tried this setup
                    continue
                constrained_initial_setups.append(constrain_res.x)
                x0 = constrain_res.x

            res = optimize.minimize(objective, x0, constraints=self.constraints)
            if not res.success:
                continue

            sol = ModeMatchingSolution(
                candidate=self,
                overlap=-res.fun,
                positions=res.x,
            )
            if any(np.allclose(sol.positions, pos, atol=equal_setup_tol, rtol=0) for pos in solution_positions):
                continue
            if filter_pred(sol):  # pyright: ignore[reportCallIssue]
                solutions.append(sol)
                solution_positions.append(sol.positions)

            # if optimize_coupling and sol.overlap >= 1 - 1e-3 and len(solutions) < solution_per_population:
            #     improved_sol = sol.optimize_coupling()
            #     if improved_sol is not None and filter_pred(improved_sol):  # pyright: ignore[reportCallIssue]
            #         if any(
            #             np.allclose(improved_sol.positions, pos, atol=equal_setup_tol, rtol=0)
            #             for pos in solution_positions
            #         ):
            #             continue
            #         solutions.append(improved_sol)
            #         solution_positions.append(improved_sol.positions)

        return solutions


@dataclass(frozen=True)
class ModeMatchingSolution:
    """A solution to a mode matching problem.

    Implements :meth:`_repr_png_` to show a plot of the solution setup in IPython environments.
    """

    candidate: ModeMatchingCandidate  #: The candidate that produced this solution
    overlap: float  #: Mode overlap of the solution
    positions: np.ndarray  #: Positions of the lenses in the solution

    @property
    def setup(self) -> OpticalSetup:
        """Optical setup corresponding to this solution."""
        return self.candidate.parametrized_setup.substitute(self.positions)  # pyright: ignore[reportArgumentType]

    plot_setup = plot_mode_match_solution_setup  #: Plot the solution setup, see :func:`corset.plot.plot_mode_match_solution_setup`
    plot_reachability = plot_reachability  #: Plot the reachability analysis, see :func:`corset.plot.plot_reachability`
    plot_sensitivity = plot_sensitivity  #: Plot the sensitivity analysis, see :func:`corset.plot.plot_sensitivity`
    plot_all = plot_mode_match_solution_all  #: Plot all analyses, see :func:`corset.plot.plot_mode_match_solution_all`

    def _repr_png_(self) -> bytes:
        fig, _ = self.plot_all()
        return fig_to_png(fig)

    @cached_property
    def analysis(self) -> "ModeMatchingAnalysis":
        """Analysis data for this solution."""
        from . import analysis

        return analysis.ModeMatchingAnalysis(solution=self)

    def optimize_coupling(
        self, min_abs_improvement: float = 0.1, min_rel_improvement: float = 0.5
    ) -> "ModeMatchingSolution | None":
        """Optimize the solution to reduce coupling between least coupled pair while maintaining mode matching.

        Note that this requires at least 3 elements so that there is at least one degree of freedom left after mode matching.

        Args:
            min_abs_improvement: Minimum absolute improvement in coupling to return the new solution.
            min_rel_improvement: Minimum relative improvement in coupling to return the new solution.

        Returns:
            A new :class:`ModeMatchingSolution` with improved coupling if successful, otherwise None.
        """

        from . import analysis

        if not len(self.positions) > 2:
            return None
            # raise ValueError("Need at least 3 free parameters to optimize coupling")
        if self.overlap < 1 - 1e-3:
            raise ValueError("Can only optimize coupling for solutions ~100% mode overlap")

        focus_and_waist = analysis.make_focus_and_waist(self)
        desired_beam = self.candidate.problem.desired_beam
        desired_focus_and_waist = np.array([desired_beam.focus, desired_beam.waist])
        mode_matching_constraint = optimize.NonlinearConstraint(
            lambda x: focus_and_waist(x) - desired_focus_and_waist, [0, 0], [0, 0]
        )

        res = optimize.minimize(
            lambda x: ModeMatchingSolution(
                candidate=self.candidate,
                overlap=1,
                positions=x,
            ).analysis.min_coupling,
            self.positions,
            constraints=[*self.candidate.constraints, mode_matching_constraint],
        )

        if not res.success:
            return None

        mode_matching = analysis.make_mode_overlap(self)(res.x)
        new_solution = ModeMatchingSolution(
            candidate=self.candidate,
            overlap=mode_matching,
            positions=res.x,
        )
        old_coupling = self.analysis.min_coupling
        new_coupling = new_solution.analysis.min_coupling
        if (old_coupling - new_coupling) > min_abs_improvement or new_coupling / old_coupling < min_rel_improvement:
            return new_solution
        return None


@dataclass(frozen=True)
class SolutionList:
    """List of mode matching solutions with convenient methods for filtering and sorting.

    Supports (array) indexing, iteration, and other list-like operations.
    Implements :meth:`_repr_html_` to show a :class:`pandas.DataFrame` representation in IPython environments.
    """

    solutions: list[ModeMatchingSolution]  #: List of mode matching solutions

    @overload
    def __getitem__(self, index: int) -> "ModeMatchingSolution": ...

    @overload
    def __getitem__(self, index: slice | list[int]) -> "SolutionList": ...

    def __getitem__(self, index: int | slice | list[int]) -> "ModeMatchingSolution | SolutionList":
        if isinstance(index, int):
            return self.solutions[index]
        else:
            return SolutionList(solutions=np.array(self.solutions)[index].tolist())

    def __len__(self) -> int:
        return len(self.solutions)

    def __iter__(self):
        return iter(self.solutions)

    @cached_property
    def df(self) -> pd.DataFrame:
        """DataFrame representation of the solutions for easy analysis."""
        return pd.DataFrame([sol.analysis.summary() for sol in self.solutions])

    def _repr_html_(self) -> str:
        return self.df.to_html(notebook=True)

    def query(self, expr: str) -> "SolutionList":
        """Filter solutions based on a :meth:`pandas.DataFrame.query` expression applied to the DataFrame representation.

        Args:
            expr: Query string, see :meth:`pandas.DataFrame.query` for details.

        Returns:
            A new :class:`SolutionList` containing only the solutions that satisfy the query.
        """
        return self[cast(list[int], self.df.query(expr).index)]

    def sort_values(self, by: str | list[str], ascending: bool = True) -> "SolutionList":
        """Sort solutions based on a column or list of columns in the DataFrame representation.

        Args:
            by: Column name or list of column names to sort by.
            ascending: Whether to sort in ascending order.

        Returns:
            A new :class:`SolutionList` with solutions sorted by the specified columns.
        """
        return self[cast(list[int], self.df.sort_values(by=by, ascending=ascending).index)]


# TODO should this be a method of ModeMatchingProblem?
def mode_match(
    setup: Beam | OpticalSetup,
    desired_beam: Beam,
    ranges: list[ShiftingRange],
    selection: Sequence[Lens] = [],
    min_elements: int = 1,
    max_elements: int = float("inf"),  # pyright: ignore[reportArgumentType]
    constraints: Sequence[Aperture | Passage] = [],
    filter_pred: Callable[[ModeMatchingSolution], bool] | float | None = 0.999,  # allow for some numerical error
    random_initial_positions: int = 0,
    solution_per_population: int = 1,
    equal_setup_tol: float = 1e-3,
    random_seed: int = 0,
    # optimize_coupling: bool = False,
    # pure_constraints: bool = False,  # TODO also give constraints a slight weight when there are excess degrees of freedom
    # TODO other solver options
):
    """Solve a mode matching problem by optimizing lens positions while respecting constraints.

    Args:
        setup: Initial optical setup or initial beam.
        desired_beam: Desired output beam after the optical setup.
        ranges: Shifting ranges where lenses can be placed.
        selection: Selection of lenses to choose from for mode matching.
        min_elements: Minimum number of elements to use.
        max_elements: Maximum number of elements to use.
        constraints: Beam constraints on the resulting optical setup.
        filter_pred: Predicate function or minimum overlap float to filter solutions.
        random_initial_positions: Number of random initial positions to generate per candidate.
        solution_per_population: Maximum number of solutions to optimize and return per lens population.
        equal_setup_tol: Tolerance for considering two solutions as equal for eliminating duplicates.
        random_seed: Random seed for reproducibility.

    Returns:
        A :class:`SolutionList` containing the optimized mode matching solutions.
    """

    # verify and prepare inputs
    if isinstance(setup, Beam):
        setup = OpticalSetup(setup, [])

    if isinstance(filter_pred, float):
        min_overlap = filter_pred
        filter_pred = lambda s: s.overlap >= min_overlap
    elif filter_pred is None:
        filter_pred = lambda s: True

    problem = ModeMatchingProblem(
        setup=setup,
        desired_beam=desired_beam,
        ranges=ranges,
        selection=selection,
        min_elements=min_elements,
        max_elements=max_elements,
        constraints=constraints,
        rng=np.random.default_rng(random_seed),
    )

    solutions = []
    # TODO parallelize this loop?
    for candidate in problem.candidates():
        solutions.extend(
            candidate.optimize(
                filter_pred=filter_pred,  # pyright: ignore[reportArgumentType]
                random_initial_positions=random_initial_positions,
                equal_setup_tol=equal_setup_tol,
                solution_per_population=solution_per_population,
                # optimize_coupling=optimize_coupling,
            )
        )

    return SolutionList(solutions)

    # TODO some sanity checks to ensure the desired beam is after the last setup part
    # TODO other checks like non overlapping regions
    # TODO return special solution list type that allows sorting and filtering and other convenient stuff
