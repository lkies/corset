"""Analysis tools for detailed analysis of mode matching solutions."""

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
    """Wrap a function to implement the vectorized input/output behavior
    expected by :func:`scipy.differentiate.hessian` and :func:`scipy.differentiate.jacobian`."""

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
    """Partial function application for functions with a single vector valued argument.

    Args:
        func: Function to partially apply.
        default: Base vector that is partially applied. The values at the unbound dimensions are ignored.
        dims: The indices of the elements that are the inputs to the resulting function, the remaining
            values are taken from the default vector.

    Returns:
        A function that takes only the specified dimensions as input and fills in the rest from the default vector.
    """

    default = default.copy()

    @wraps(func)
    def wrapped(var: np.ndarray) -> np.ndarray:
        full_input = default.copy()
        for i, dim in enumerate(dims):
            full_input[dim] = var[i]
        return func(full_input)

    return wrapped


def make_mode_overlap(solution: "ModeMatchingSolution") -> Callable[[np.ndarray], float]:
    """Create a function that computes the mode overlap as the function of the free element positions.

    Args:
        solution: The mode matching solution to base the overlap function on.

    Returns:
        A function that takes an array of element positions and returns the mode overlap.
    """
    from . import solver

    def overlap(positions: np.ndarray) -> float:
        setup = solution.candidate.parametrized_setup.substitute(positions, validate=False)
        final_beam = setup.beams[-1]
        problem = solution.candidate.problem
        return solver.mode_overlap(
            final_beam.focus - problem.desired_beam.focus,
            final_beam.waist,
            problem.desired_beam.waist,
            problem.setup.initial_beam.wavelength,
        )

    return overlap


def make_focus_and_waist(solution: "ModeMatchingSolution") -> Callable[[np.ndarray], np.ndarray]:
    """Create a function that computes the focus and waist of the final beam as a function of the free element positions.

    Args:
        solution: The mode matching solution to base the focus and waist function on.

    Returns:
        A function that takes an array of element positions and returns the focus and waist.
    """

    def focus_and_waist(positions: np.ndarray) -> np.ndarray:
        setup = solution.candidate.parametrized_setup.substitute(positions, validate=False)
        final_beam = setup.beams[-1]
        return np.array([final_beam.focus, final_beam.waist])

    return focus_and_waist


@dataclass(frozen=True)
class ModeMatchingAnalysis:
    """Analysis of a mode matching solution providing various sensitivity metrics."""

    solution: "ModeMatchingSolution"  #: The mode matching solution to analyze.

    @cached_property
    def hessian(self) -> np.ndarray:
        r"""The Hessian matrix :math:`\mathbf{H}` of the mode overlap function
        :math:`o(\mathbf{x})` around the optimum :math:`\mathbf{x}^*`. The individual elements
        :math:`h_{ij}` of the Hessian are given by:

        .. math::
            h_{ij} = \left. \frac{\partial^2 o(\mathbf{x})}{\partial x_i \partial x_j} \right|_{\mathbf{x} = \mathbf{x}^*}

        For problems with two degrees of freedom the Hessian is always negative definite, if there are
        more than two degrees of freedom it generally negative semi-definite with reduced rank or at least
        very bad conditioning.
        """

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
        r"""The Jacobian :math:`\mathbf{J}` of the final beam focus and waist :math:`\mathbf{f}_{fw}(\mathbf{x})` with respect to
        the element positions :math:`\mathbf{x}` around the optimum :math:`\mathbf{x}^*`. The individual elements
        :math:`j_{ij}` of the Jacobian are given by:

        .. math::
            j_{ij} = \left.\frac{\partial f_{fw,i}(\mathbf{x})}{\partial x_j} \right|_{\mathbf{x} = \mathbf{x}^*}

        """
        waist_and_focus = wrap_for_differentiate(make_focus_and_waist(self.solution))
        jac_res = jacobian(waist_and_focus, self.solution.positions, initial_step=1e-2)
        if np.any(jac_res.status != 0):
            warnings.warn(f"Jacobian calculation did not converge: {jac_res.status}", stacklevel=2)
        return jac_res.df

    @cached_property
    def couplings(self) -> np.ndarray:
        r"""The coupling matrix :math:`\mathbf{R}` between the different degrees of freedom.

        The coupling :math:`r_{ij}` between degrees of freedom indexed :math:`i` and :math:`j` is
        the normalized cross-sensitivity :math:`s_{ij}` between the two degrees of freedom:

        .. math::
            r_{ij} = \frac{s_{ij}}{\sqrt{s_{ii} s_{jj}}}

        """
        normalizer = 1 / np.sqrt(np.diag(self.sensitivities))
        return self.sensitivities * np.outer(normalizer, normalizer)

    @cached_property
    def sensitivities(self) -> np.ndarray:
        r"""The sensitivity matrix :math:`\mathbf{S}` of the mode overlap around the optimum.

        It is proportional to the Hessian :math:`\mathbf{H}` with :math:`\mathbf{S} = -\mathbf{H}/2`.
        That way the positive loss in mode overlap :math:`\Delta o` for small perturbations
        :math:`\Delta \mathbf{x}` around the optimum can be expressed as:

        .. math::
            \Delta o \approx \mathbf{\Delta x}^T \mathbf{S} \mathbf{\Delta x}

        """

        return -self.hessian / 2

    @cached_property
    def min_coupling_pair(self) -> tuple[int, int]:
        """The indices :math:`(i, j)` of the pair of degrees of freedom with minimal absolute coupling :math:`r_{ij}`.
        The second index is always larger than the first.
        """

        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_couplings = np.abs(self.couplings[indices])
        best = np.argmin(abs_couplings)
        return (int(indices[0][best]), int(indices[1][best]))

    @cached_property
    def min_coupling(self) -> float:
        """The minimal absolute coupling between any pair of degrees of freedom."""
        return abs(float(self.couplings[self.min_coupling_pair]))

    @cached_property
    def min_cross_sens_pair(self) -> tuple[int, int]:
        """The indices :math:`(i, j)` of the pair of degrees of freedom with minimal absolute cross-sensitivity :math:`s_{ij}`.
        The second index is always larger than the first.
        """
        indices = np.triu_indices(len(self.sensitivities), k=1)
        abs_sensitivities = np.abs(self.sensitivities[indices])
        best = np.argmin(abs_sensitivities)
        return (int(indices[0][best]), int(indices[1][best]))

    @cached_property
    def min_cross_sens(self) -> float:
        """The minimal absolute cross-sensitivity between any pair of degrees of freedom."""
        return abs(float(self.sensitivities[self.min_cross_sens_pair]))

    @cached_property
    def min_cross_sens_direction(self) -> np.ndarray:
        """The direction of the least cross-sensitive pair of degrees of freedom.
        This is the smallest eigenvector of the 2x2 cross-sensitivity sub-matrix of the least cross-sensitive pair.
        """
        pair = self.min_cross_sens_pair
        eigs = np.linalg.eigh(self.sensitivities[np.ix_(pair, pair)])
        vec = eigs.eigenvectors[:, np.argmin(eigs.eigenvalues)]
        return vec if vec[0] >= 0 else -vec

    @cached_property
    def min_sensitivity_axis(self) -> int:
        """The index :math:`i` of the degree of freedom with minimal absolute sensitivity :math:`s_{ii}`."""

        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmin(diag_sensitivities))

    @cached_property
    def min_sensitivity(self) -> float:
        """The minimal absolute sensitivity among all degrees of freedom."""

        return float(self.sensitivities[self.min_sensitivity_axis, self.min_sensitivity_axis])

    @cached_property
    def max_sensitivity_axis(self) -> int:
        """The index :math:`i` of the degree of freedom with maximal absolute sensitivity :math:`s_{ii}`."""
        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmax(diag_sensitivities))

    @cached_property
    def max_sensitivity(self) -> float:
        """The maximal absolute sensitivity among all degrees of freedom."""

        return float(self.sensitivities[self.max_sensitivity_axis, self.max_sensitivity_axis])

    # the vectors spanning the sub space in which the mode overlap stays approximately constant
    # equivalent to the null space of the hessian assuming the minor eigenvalues are zero
    @cached_property
    def const_space(self) -> list[np.ndarray]:
        r"""The basis vectors spanning the constant overlap sub-space around the optimum.

        Note:
            This is simply determined as the corresponding eigenvectors two all but the two largest eigenvalues
            of the Hessian :math:`\mathbf{H}`. These eigenvalues are usually many orders of magnitude smaller than the
            two largest ones, but generally not zero.
        """

        eigs = np.linalg.eigh(self.hessian)
        return list(eigs.eigenvectors.T[np.argsort(eigs.eigenvalues)[2:]])

    @cached_property
    def grad_focus(self) -> np.ndarray:
        r"""The gradient of the final beam focus with respect to the element positions,
        equal to the first row of the Jacobian."""

        return self.focus_and_waist_jacobian[0]

    @cached_property
    def grad_waist(self) -> np.ndarray:
        r"""The gradient of the final beam waist with respect to the element positions,
        equal to the second row of the Jacobian."""
        return self.focus_and_waist_jacobian[1]

    def summary(self, sensitivity_unit: Config.SensitivityUnit | bool | None = None) -> dict:
        """Create a summary dictionary of the analysis results

        Args:
            sensitivity_unit: The unit to use for sensitivities in the summary. If `None` the default from :class:`Config` is used.
                If `False`, the raw sensitivities without unit conversion are used.

        Returns:
            A dictionary containing the summary data. The keys are:
            - "overlap": The mode overlap of the solution.
            - "num_elements": The number of free elements (i.e. elements used for mode matching) in the setup.
            - "elements": A list of the free elements (i.e. elements used for mode matching) in the setup.
            - "positions": The positions of the free elements in the setup.
            - "min_sensitivity_axis": The index of the degree of freedom with minimal sensitivity.
            - "min_sensitivity": The minimal sensitivity in the specified unit.
            - "max_sensitivity_axis": The index of the degree of freedom with maximal sensitivity.
            - "max_sensitivity": The maximal sensitivity in the specified unit.
            - "min_cross_sens_pair": The indices of the pair of degrees of freedom with minimal cross-sensitivity.
            - "min_cross_sens": The minimal cross-sensitivity in the specified unit.
            - "min_cross_sens_direction": The direction of the least cross-sensitive pair of degrees of freedom.
            - "min_coupling_pair": The indices of the pair of degrees of freedom with minimal coupling.
            - "min_coupling": The minimal coupling.
            - "sensitivities": The sensitivity matrix in the specified unit.
            - "couplings": The coupling matrix.
            - "const_space": The basis vectors spanning the constant overlap sub-space.
            - "grad_focus": The gradient of the final beam focus with respect to the element positions.
            - "grad_waist": The gradient of the final beam waist with respect to the element positions.
            - "sensitivity_unit": The sensitivity unit used.
            - "solution": The analyzed mode matching solution.

        """

        sensitivity_unit = cast(Config.SensitivityUnit, sensitivity_unit or Config.sensitivity_unit)
        factor = sensitivity_unit.value.factor
        sol = self.solution
        return {
            "overlap": sol.overlap,
            "num_elements": len(sol.positions),
            "elements": [sol.setup.elements[i][1] for i in sol.candidate.parametrized_setup.free_elements],
            "positions": sol.positions,
            "min_sensitivity_axis": self.min_sensitivity_axis,
            "min_sensitivity": self.min_sensitivity * factor,
            "max_sensitivity_axis": self.max_sensitivity_axis,
            "max_sensitivity": self.max_sensitivity * factor,
            "min_cross_sens_pair": self.min_cross_sens_pair,
            "min_cross_sens": self.min_cross_sens * factor,
            "min_cross_sens_direction": self.min_cross_sens_direction,
            "min_coupling_pair": self.min_coupling_pair,
            "min_coupling": self.min_coupling,
            "sensitivities": self.sensitivities * factor,
            "couplings": self.couplings,
            "const_space": self.const_space,
            "grad_focus": self.grad_focus,
            "grad_waist": self.grad_waist,
            "sensitivity_unit": sensitivity_unit,
            "solution": sol,
        }

    def summary_df(self, sensitivity_unit: Config.SensitivityUnit | None | bool = None) -> pd.DataFrame:
        """Create a summary DataFrame of the analysis results

        Args:
            sensitivity_unit: The unit to use for sensitivities in the summary. If `None` the default from :class:`Config` is used.
                If `False`, the raw sensitivities without unit conversion are used.

        Returns:
            A DataFrame containing the summary data with one row per value, see :meth:`summary` for details.
        """

        summary_data = self.summary(sensitivity_unit=sensitivity_unit)
        return pd.DataFrame([summary_data]).T
