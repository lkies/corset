from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
from scipy import optimize

from .core import Beam, OpticalSetup, thin_lens


# TODO Element?
@dataclass(frozen=True)
class Lens:
    focal_length: float
    left_margin: float  # TODO default values?
    right_margin: float

    @property
    def matrix(self) -> np.ndarray:
        return thin_lens(self.focal_length)


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
    elements: list[tuple[float | None, np.ndarray]]  # position, element

    def substitute(self, positions: list[float] | np.ndarray) -> OpticalSetup:
        substituted_elements = []
        pos_index = 0
        try:
            for pos, matrix in self.elements:
                if pos is None:
                    pos = positions[pos_index]
                    pos_index += 1
                substituted_elements.append((pos, matrix))
        except IndexError as e:
            raise ValueError("Not enough positions provided to substitute all parametrized elements") from e
        if pos_index < len(positions):
            raise ValueError("Too many positions provided for the number of parametrized elements")
        return OpticalSetup(self.initial_beam, substituted_elements)


def mode_overlap(waist_a: float, waist_b: float, delta_z: float, wavelength: float) -> float:
    return (
        2 * np.pi * waist_a * waist_b / np.sqrt(wavelength**2 * delta_z**2 + np.pi**2 * (waist_a**2 + waist_b**2) ** 2)
    )


# TODO tests
def lens_combinations(
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
            yield from lens_combinations(
                rest, base_selection, min_elements - num_elements, max_elements - num_elements, new_setup
            )


# TODO verify no overlap between regions, maybe also verify regions sizes can fit lenses?
def verify_regions(regions: list[Region], selection: list[Lens], min_elements: int, max_elements: int):
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

    pass
    # for region in regions:
    #     if region.left >= region.right:
    #         raise ValueError("Region left boundary must be less than right boundary")


def make_position_constraint(
    regions: list[Region], region_populations: list[tuple[Lens, ...]]
) -> optimize.LinearConstraint:
    constraints: list[tuple[np.ndarray, float, float]] = []
    pop_sizes = [len(pop) for pop in region_populations]
    index_offsets = np.cumsum([0, *pop_sizes[:-1]])
    mask = np.identity(sum(pop_sizes))
    for region, population, base_idx in zip(regions, region_populations, index_offsets, strict=True):
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
                (mask[base_idx + i + 1] - mask[base_idx + i], left_lens.right_margin + right_lens.left_margin, np.inf)
            )
    cols, lb, ub = zip(*constraints, strict=True)
    return optimize.LinearConstraint(np.vstack(cols), np.array(lb), np.array(ub))  # pyright: ignore[reportArgumentType]


# TODO seeding?
def generate_initial_positions(
    regions: list[Region], region_populations: list[tuple[Lens, ...]], randomize: bool = True
) -> np.ndarray:
    positions = []
    for region, population in zip(regions, region_populations, strict=True):
        if not population:
            continue
        total_margin = sum(lens.left_margin + lens.right_margin for lens in population)
        available_space = region.right - region.left - total_margin
        if available_space < 0:
            raise ValueError("Not enough space in region for the lenses with their margins")

        if randomize:
            distances = np.diff(np.sort(np.random.uniform(0, available_space, len(population))), prepend=0)
        else:
            distances = np.repeat(available_space / (len(population) + 1), len(population))

        current_pos = region.left
        for lens, distance in zip(population, distances, strict=True):
            current_pos += lens.left_margin + distance
            positions.append(current_pos)
            current_pos += lens.right_margin

    return np.array(positions)


@dataclass(frozen=True)
class ModeMatchSolution:
    base_setup: OpticalSetup
    desired_beam: Beam
    overlap: float
    parametrized_setup: ParametrizedSetup
    positions: np.ndarray
    regions: list[Region]
    region_populations: list[tuple[Lens, ...]]

    @property
    def setup(self) -> OpticalSetup:
        return self.parametrized_setup.substitute(self.positions)


def merge_elements(setup: OpticalSetup, regions: list[Region]) -> list[tuple[float, np.ndarray] | int]:
    merged: list[tuple[float, np.ndarray] | int] = []
    next_boundary = regions[0].left if regions else float("inf")
    region_index = 0

    for pos, matrix in setup.elements:
        while pos > next_boundary:
            merged.append(region_index)
            region_index += 1
            next_boundary = regions[region_index].left if region_index < len(regions) else float("inf")
        merged.append((pos, matrix))

    while region_index < len(regions):
        merged.append(region_index)
        region_index += 1

    return merged


def mode_match(  # noqa: C901
    setup: Beam | OpticalSetup,
    desired_beam: Beam,
    regions: list[Region],
    selection: list[Lens] = [],  # noqa: B006
    min_elements: int = 0,
    max_elements: int = float("inf"),  # pyright: ignore[reportArgumentType]
    random_initial_positions: int = 0,
    filter_pred: Callable[[ModeMatchSolution], bool] | float | None = None,
    equal_setup_tol: float = 1e-3,
    # TODO other beam constraints
    # TODO constraints
    # TODO other solver options
):
    # verify and prepare inputs
    if isinstance(setup, Beam):
        setup = OpticalSetup(setup, [])
    verify_regions(regions, selection, min_elements, max_elements)
    merged_elements = merge_elements(setup, regions)

    if isinstance(filter_pred, float):
        min_overlap = filter_pred
        filter_pred = lambda s: s.overlap >= min_overlap
    elif filter_pred is None:
        filter_pred = lambda s: True

    solutions = []
    # TODO parallelize this loop?
    for lens_population in lens_combinations(regions, selection, min_elements, max_elements):
        initial_setups = [generate_initial_positions(regions, lens_population, randomize=False)]
        for _ in range(random_initial_positions):
            initial_setups.append(generate_initial_positions(regions, lens_population, randomize=True))

        # TODO refactor this into merge elements, although this would require more work in every iteration
        elements = []
        for elem in merged_elements:
            if isinstance(elem, int):
                for lens in lens_population[elem]:
                    elements.append((None, lens.matrix))
            else:
                elements.append(elem)
        parametrized_setup = ParametrizedSetup(setup.initial_beam, elements)
        constraint = make_position_constraint(regions, lens_population)

        def objective(positions: np.ndarray) -> float:
            setup = parametrized_setup.substitute(positions)  # noqa: B023
            final_beam = setup.beams[-1]
            return -mode_overlap(
                final_beam.waist,
                desired_beam.waist,
                final_beam.focus - desired_beam.focus,
                final_beam.wavelength,  # pyright: ignore[reportArgumentType]
            )

        solution_positions = []  # for this lens population
        for x0 in initial_setups:
            res = optimize.minimize(objective, x0, constraints=[constraint])
            if not res.success:
                continue
            sol = ModeMatchSolution(
                base_setup=setup,
                desired_beam=desired_beam,
                overlap=-res.fun,
                parametrized_setup=parametrized_setup,
                positions=res.x,
                regions=regions,
                region_populations=lens_population,
            )
            if any(np.allclose(sol.positions, pos, atol=equal_setup_tol, rtol=0) for pos in solution_positions):
                continue
            if filter_pred(sol):  # pyright: ignore[reportCallIssue]
                solutions.append(sol)
                solution_positions.append(sol.positions)

    return sorted(solutions, key=lambda s: s.overlap, reverse=True)

    # TODO some sanity checks to ensure the desired beam is after the last setup part
    # TODO other checks like non overlapping regions
