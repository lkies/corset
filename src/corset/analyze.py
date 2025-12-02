import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.differentiate import hessian, jacobian

from .config import Config

if TYPE_CHECKING:
    from .solver import ModeMatchingSolution


def wrap_for_differentiate(
    func: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    @wraps(func)
    def wrapped(x: np.ndarray) -> np.ndarray:
        inputs = np.moveaxis(x, 0, -1)
        raw_res = [func(inp) for inp in inputs.reshape(-1, x.shape[0])]
        out_shape = np.shape(raw_res[0])
        res = np.reshape(raw_res, x.shape[1:] + out_shape)  # pyright: ignore[reportCallIssue]
        if len(out_shape) != 0:
            return np.moveaxis(res, -1, 0)
        return res

    return wrapped


def vector_partial(
    func: Callable[[np.ndarray], np.ndarray], default: np.ndarray, dims: Iterable[int]
) -> Callable[[np.ndarray], np.ndarray]:
    default = default.copy()

    @wraps(func)
    def wrapped(var: np.ndarray) -> np.ndarray:
        full_input = default.copy()
        for i, dim in enumerate(dims):
            full_input[dim] = var[i]
        return func(full_input)

    return wrapped


def make_mode_overlap(solution: "ModeMatchingSolution") -> Callable[[np.ndarray], float]:
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


def make_focus_and_waist(solution: "ModeMatchingSolution") -> Callable[[np.ndarray], np.ndarray]:
    def focus_and_waist(positions: np.ndarray) -> np.ndarray:
        setup = solution.candidate.parametrized_setup.substitute(positions)
        final_beam = setup.beams[-1]
        return np.array([final_beam.focus, final_beam.waist])

    return focus_and_waist


@dataclass(frozen=True)
class SensitivityAnalysis:
    solution: "ModeMatchingSolution"

    @cached_property
    def hessian(self) -> np.ndarray:
        mode_overlap = wrap_for_differentiate(make_mode_overlap(self.solution))  # pyright: ignore[reportArgumentType]

        # the default initial step is 0.5 which would lead to invalid lens position
        # 1e-2 ensures, that only physical configurations are evaluated
        tolerances = {"atol": 1e-6, "rtol": 1e-6}
        hess_res = hessian(mode_overlap, self.solution.positions, initial_step=1e-2, tolerances=tolerances)
        if np.any(hess_res.status != 0):
            warnings.warn(f"Hessian calculation did not converge: {hess_res.status}", stacklevel=2)

        return hess_res.ddf

    @cached_property
    def focus_and_waist_jacobian(self) -> np.ndarray:
        waist_and_focus = wrap_for_differentiate(make_focus_and_waist(self.solution))
        jac_res = jacobian(waist_and_focus, self.solution.positions, initial_step=1e-2)
        if np.any(jac_res.status != 0):
            warnings.warn(f"Jacobian calculation did not converge: {jac_res.status}", stacklevel=2)
        return jac_res.df

    @cached_property
    def couplings(self) -> np.ndarray:
        normalizer = 1 / np.sqrt(-np.diag(self.hessian))
        return -self.hessian * np.outer(normalizer, normalizer)

    @cached_property
    def sensitivities(self) -> np.ndarray:
        return -self.hessian / 2

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

    @cached_property
    def grad_focus(self) -> np.ndarray:
        return self.focus_and_waist_jacobian[0]

    @cached_property
    def grad_waist(self) -> np.ndarray:
        return self.focus_and_waist_jacobian[1]

    def summary(self, sensitivity_unit: Config.SensitivityUnit | None | bool = None) -> dict:
        sensitivity_unit = cast(Config.SensitivityUnit, sensitivity_unit or Config.sensitivity_unit)
        factor = sensitivity_unit.value.factor
        return {
            "overlap": self.solution.overlap,
            "num_elements": len(self.solution.positions),
            "elements": [str(elem) for _, elem in self.solution.setup.elements],
            "min_sensitivity_axis": self.min_sensitivity_axis,
            "min_sensitivity": self.min_sensitivity * factor,
            "max_sensitivity_axis": self.max_sensitivity_axis,
            "max_sensitivity": self.max_sensitivity * factor,
            "min_cross_sens_pair": self.min_cross_sens_pair,
            "min_cross_sens": self.min_cross_sens * factor,
            "min_coupling_pair": self.min_coupling_pair,
            "min_coupling": self.min_coupling,
            "sensitivities": self.sensitivities * factor,
            "couplings": self.couplings,
            "const_space": self.const_space,
            "sensitivity_unit": sensitivity_unit,
            "solution": self.solution,
        }

    def summary_df(self, sensitivity_unit: Config.SensitivityUnit | None | bool = None) -> pd.DataFrame:
        summary_data = self.summary(sensitivity_unit=sensitivity_unit)
        return pd.DataFrame([summary_data]).T
