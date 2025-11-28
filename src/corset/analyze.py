import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.differentiate import hessian

from .config import Config

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
    def min_coupling_pair(self) -> tuple[int, int]:
        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_couplings = np.abs(self.couplings[indices])
        best = np.argmin(abs_couplings)
        return (int(indices[0][best]), int(indices[1][best]))

    @cached_property
    def min_coupling(self) -> float:
        return abs(float(self.couplings[self.min_coupling_pair]))

    @cached_property
    def min_cross_sens_pair(self) -> tuple[int, int]:
        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_sensitivities = np.abs(self.sensitivities[indices])
        best = np.argmin(abs_sensitivities)
        return (int(indices[0][best]), int(indices[1][best]))

    @cached_property
    def min_cross_sens(self) -> float:
        return abs(float(self.sensitivities[self.min_cross_sens_pair]))

    @cached_property
    def min_sensitivity_axis(self) -> int:
        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmin(diag_sensitivities))

    @cached_property
    def min_sensitivity(self) -> float:
        return float(self.sensitivities[self.min_sensitivity_axis, self.min_sensitivity_axis])

    @cached_property
    def max_sensitivity_axis(self) -> int:
        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmax(diag_sensitivities))

    @cached_property
    def max_sensitivity(self) -> float:
        return float(self.sensitivities[self.max_sensitivity_axis, self.max_sensitivity_axis])

    # the vectors spanning the sub space in which the mode overlap stays approximately constant
    # equivalent to the null space of the hessian assuming the minor eigenvalues are zero
    @cached_property
    def const_space(self) -> list[np.ndarray]:
        eigs = np.linalg.eigh(self.hessian)
        return list(eigs.eigenvectors.T[np.argsort(eigs.eigenvalues)[2:]])

    def report(self, sensitivity_unit: Config.SensitivityUnit | None | bool = None) -> dict:
        unit_suffix = ""
        factor = 1.0
        if sensitivity_unit is None:
            sensitivity_unit = Config.SENSITIVITY_UNIT
            unit_suffix = "_" + sensitivity_unit.value.ascii
            factor = sensitivity_unit.value.factor
        return {
            "overlap": self.solution.overlap,
            "elements": len(self.solution.positions),
            "min_sensitivity_axis": self.min_sensitivity_axis,
            "min_sensitivity" + unit_suffix: self.min_sensitivity * factor,
            "max_sensitivity_axis": self.max_sensitivity_axis,
            "max_sensitivity" + unit_suffix: self.max_sensitivity * factor,
            "min_cross_sens_pair": self.min_cross_sens_pair,
            "min_cross_sens" + unit_suffix: self.min_cross_sens * factor,
            "min_coupling_pair": self.min_coupling_pair,
            "min_coupling": self.min_coupling,
            "sensitivities" + unit_suffix: self.sensitivities * factor,
            "couplings": self.couplings,
            "const_space": self.const_space,
            "solution": self.solution,
        }

    def report_df(self, sensitivity_unit: Config.SensitivityUnit | None | bool = None) -> pd.DataFrame:
        report_data = self.report(sensitivity_unit=sensitivity_unit)
        return pd.DataFrame([report_data]).T
