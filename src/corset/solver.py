from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations_with_replacement

import numpy as np
from scipy import optimize

from .core import Beam, Lens, OpticalSetup
from .plot import plot_mode_match_solution


@dataclass(frozen=True)
class Region:
    left: float  # TODO left/right, begin/end start/stop?
    right: float
    min_elements: int = 0
    max_elements: int = float("inf")  # pyright: ignore[reportAssignmentType]
    selection: list[Lens] = field(default_factory=list)  # TODO name

    def __post_init__(self):
        if self.right <= self.left:
            raise ValueError("Region right boundary must be greater than left boundary")
        if self.min_elements < 0:
            raise ValueError("min_elements cannot be negative")
        if self.max_elements is not None and self.max_elements < self.min_elements:
            raise ValueError("max_elements cannot be less than min_elements")


@dataclass(frozen=True)
class ParametrizedSetup:
    initial_beam: Beam
    elements: list[tuple[float | None, Lens]]  # position, element

    def substitute(self, positions: list[float] | np.ndarray) -> OpticalSetup:
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
        return OpticalSetup(self.initial_beam, substituted_elements)


@dataclass(frozen=True)
class Aperture:  # ApertureConstraint ?
    position: float
    radius: float

    def apertures(self) -> tuple["Aperture"]:
        return (self,)


@dataclass(frozen=True)
class Passage:  # PassageConstraint ? or some other better name
    left: float
    right: float
    radius: float

    @classmethod
    def centered(cls, center: float, width: float, radius: float) -> "Passage":
        half_width = width / 2
        return cls(left=center - half_width, right=center + half_width, radius=radius)

    def apertures(self) -> tuple[Aperture, Aperture]:
        return (Aperture(self.left, self.radius), Aperture(self.right, self.radius))


def mode_overlap(waist_a: float, waist_b: float, delta_z: float, wavelength: float) -> float:
    return (
        2 * np.pi * waist_a * waist_b / np.sqrt(wavelength**2 * delta_z**2 + np.pi**2 * (waist_a**2 + waist_b**2) ** 2)
    )


# TODO should this be part of some class?
# TODO verify no overlap between regions, maybe also verify regions sizes can fit lenses?
def verify_regions(regions: list[Region], min_elements: int, max_elements: int):
    if max_elements is not None and max_elements < min_elements:
        raise ValueError("Global max_elements cannot be less than min_elements")

    total_min = sum(region.min_elements for region in regions)
    total_max = sum(region.max_elements for region in regions)

    if total_max == float("inf") and max_elements == float("inf"):
        raise ValueError("Cannot have unbounded maximum elements when global maximum is not set")

    if total_min > max_elements:
        raise ValueError("Sum of region minimum elements exceeds global maximum elements")
    if total_max < min_elements:
        raise ValueError("Sum of region maximum elements is less than global minimum elements")


@dataclass(frozen=True)
class ModeMatchingProblem:
    setup: OpticalSetup
    desired_beam: Beam
    regions: list[Region]
    selection: list[Lens]
    min_elements: int
    max_elements: int  # pyright: ignore[reportAssignmentType]
    constraints: list[Aperture | Passage]
    # TODO make it so that the order of evaluating the candidates does not change their random values
    rng: np.random.Generator

    @cached_property
    def aperture_constraints(self) -> list[Aperture]:
        return [aperture for constraint in self.constraints for aperture in constraint.apertures()]

    @cached_property
    def interleaved_elements(self) -> list[tuple[float, Lens] | int]:
        merged: list[tuple[float, Lens] | int] = []
        next_boundary = self.regions[0].left if self.regions else float("inf")
        region_index = 0

        for pos, element in self.setup.elements:
            while pos > next_boundary:
                merged.append(region_index)
                region_index += 1
                next_boundary = self.regions[region_index].left if region_index < len(self.regions) else float("inf")
            merged.append((pos, element))

        while region_index < len(self.regions):
            merged.append(region_index)
            region_index += 1

        return merged

    @classmethod
    def lens_combinations(
        cls,
        regions: list[Region],
        base_selection: list[Lens],
        min_elements: int,
        max_elements: int,
        current_setup: list[tuple[Lens, ...]] = [],  # noqa: B006
    ) -> Generator[list[tuple[Lens, ...]], None, None]:
        if not regions:
            if min_elements <= 0:
                yield current_setup
            return

        first, *rest = regions

        if first.min_elements > max_elements:
            return

        for num_elements in range(first.min_elements, min(first.max_elements, max_elements) + 1):
            selection = first.selection if first.selection else base_selection
            for comb in combinations_with_replacement(selection, num_elements):
                new_setup = [*current_setup, comb]
                yield from cls.lens_combinations(
                    rest, base_selection, min_elements - num_elements, max_elements - num_elements, new_setup
                )

    def candidates(self) -> Generator["ModeMatchingCandidate", None, None]:
        for population in self.lens_combinations(self.regions, self.selection, self.min_elements, self.max_elements):
            yield ModeMatchingCandidate(problem=self, populations=population)


@dataclass(frozen=True)
class ModeMatchingCandidate:
    problem: ModeMatchingProblem
    populations: list[tuple[Lens, ...]]

    # TODO seeding?
    def generate_initial_positions(self, randomize: bool = True) -> np.ndarray:
        positions = []
        for region, population in zip(self.problem.regions, self.populations, strict=True):
            if not population:
                continue
            total_margin = sum(lens.left_margin + lens.right_margin for lens in population)
            available_space = region.right - region.left - total_margin
            if available_space < 0:
                raise ValueError("Not enough space in region for the lenses with their margins")

            if randomize:
                distances = np.diff(np.sort(self.problem.rng.uniform(0, available_space, len(population))), prepend=0)
            else:
                distances = np.repeat(available_space / (len(population) + 1), len(population))

            current_pos = region.left
            for lens, distance in zip(population, distances, strict=True):
                current_pos += lens.left_margin + distance
                positions.append(current_pos)
                current_pos += lens.right_margin

        return np.array(positions)

    @cached_property
    def parametrized_setup(self) -> ParametrizedSetup:
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
        constraints: list[tuple[np.ndarray, float, float]] = []
        pop_sizes = [len(pop) for pop in self.populations]
        index_offsets = np.cumsum([0, *pop_sizes[:-1]])
        mask = np.identity(sum(pop_sizes))
        for region, population, base_idx in zip(self.problem.regions, self.populations, index_offsets, strict=True):
            if not population:
                continue
            if len(population) == 1:
                constraints.append(
                    (mask[base_idx], region.left + population[0].left_margin, region.right - population[0].right_margin)
                )
            else:
                right_index = base_idx + len(population) - 1
                constraints.append((mask[base_idx], region.left + population[0].left_margin, np.inf))
                constraints.append((mask[right_index], -np.inf, region.right - population[-1].right_margin))

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
        positions, radii = np.transpose([(c.position, c.radius) for c in self.problem.aperture_constraints])
        return optimize.NonlinearConstraint(
            lambda x, pos=positions, setup=self.parametrized_setup: setup.substitute(x).radius(pos) / radii, 0, 1
        )

    def optimize(
        self,
        filter_pred: Callable[["ModeMatchSolution"], bool],
        random_initial_positions: int,
        equal_setup_tol: float,
    ) -> list["ModeMatchSolution"]:
        initial_setups = [self.generate_initial_positions(randomize=False)]
        for _ in range(random_initial_positions):
            initial_setups.append(self.generate_initial_positions(randomize=True))

        solver_constraints: list = [self.position_constraint]

        if self.problem.constraints:
            solver_constraints.append(self.beam_constraint)

            # make initial setups satisfy beam constraints and filter out non feasible ones
            filtered_setups = []
            for x0 in initial_setups:
                res = optimize.minimize(
                    lambda x: np.max(self.beam_constraint.fun(x)),
                    x0,
                    constraints=solver_constraints,
                    method="SLSQP",
                    options={"ftol": 2e-1},
                )
                if np.all(self.beam_constraint.fun(res.x) <= self.beam_constraint.ub) and all(
                    not np.allclose(res.x, x0) for x0 in filtered_setups
                ):
                    filtered_setups.append(res.x)
            initial_setups = filtered_setups

        def objective(positions: np.ndarray) -> float:
            setup = self.parametrized_setup.substitute(positions)  # pyright: ignore[reportArgumentType]
            final_beam = setup.beams[-1]
            desired_beam = self.problem.desired_beam
            return -mode_overlap(
                final_beam.waist,
                desired_beam.waist,
                final_beam.focus - desired_beam.focus,
                final_beam.wavelength,  # pyright: ignore[reportArgumentType]
            )

        solutions = []
        solution_positions = []  # for this lens population
        for x0 in initial_setups:
            res = optimize.minimize(objective, x0, constraints=solver_constraints)
            if not res.success:
                continue

            sol = ModeMatchSolution(
                candidate=self,
                overlap=-res.fun,
                positions=res.x,
            )
            if any(np.allclose(sol.positions, pos, atol=equal_setup_tol, rtol=0) for pos in solution_positions):
                continue
            if filter_pred(sol):  # pyright: ignore[reportCallIssue]
                solutions.append(sol)
                solution_positions.append(sol.positions)

        return solutions


@dataclass(frozen=True)
class ModeMatchSolution:
    candidate: ModeMatchingCandidate
    overlap: float
    positions: np.ndarray

    @property
    def setup(self) -> OpticalSetup:
        return self.candidate.parametrized_setup.substitute(self.positions)  # pyright: ignore[reportArgumentType]

    plot = plot_mode_match_solution


# TODO should this be a method of ModeMatchingProblem?
def mode_match(
    setup: Beam | OpticalSetup,
    desired_beam: Beam,
    regions: list[Region],
    selection: list[Lens] = [],  # noqa: B006
    min_elements: int = 0,
    max_elements: int = float("inf"),  # pyright: ignore[reportArgumentType]
    constraints: list[Aperture | Passage] = [],  # noqa: B006
    filter_pred: Callable[[ModeMatchSolution], bool] | float | None = None,
    random_initial_positions: int = 0,
    equal_setup_tol: float = 1e-3,
    random_seed: int = 0,
    # pure_constraints: bool = False,  # TODO also give constraints a slight weight when there are excess degrees of freedom
    # TODO other solver options
):
    # verify and prepare inputs
    if isinstance(setup, Beam):
        setup = OpticalSetup(setup, [])
    verify_regions(regions, min_elements, max_elements)

    if isinstance(filter_pred, float):
        min_overlap = filter_pred
        filter_pred = lambda s: s.overlap >= min_overlap
    elif filter_pred is None:
        filter_pred = lambda s: True

    problem = ModeMatchingProblem(
        setup=setup,
        desired_beam=desired_beam,
        regions=regions,
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
            )
        )

    return sorted(solutions, key=lambda s: s.overlap, reverse=True)

    # TODO some sanity checks to ensure the desired beam is after the last setup part
    # TODO other checks like non overlapping regions
    # TODO return special solution list type that allows sorting and filtering and other convenient stuff
