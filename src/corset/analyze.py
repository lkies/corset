import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg as sp_linalg
from scipy.differentiate import hessian

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
