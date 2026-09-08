"""Analysis tools for detailed analysis of mode matching solutions."""

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from .config import Config
from .core import Lens
from .display import FormattedDataFrame, FractionUnit, LengthUnit, SensitivityUnit, Units

if TYPE_CHECKING:
    from .solver import ModeMatchingSolution, ShiftingRange


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

        Note:
            The Hessian computation assumes, that the gradient of the overlap function is zero at the point
            where it is computed. This mean the Hessian is only valid if the overlap is approximately 100%.
            However this should not matter, since Hessian based metrics are only meaningful if computed
            around a stationary point.
        """
        # derivation
        """
        >>> import sympy as sp
        >>> focus, waist, desired_focus, desired_waist, wavelength = sp.symbols('focus waist desired_focus desired_waist wavelength', real=True)
        >>> delta_z = focus - desired_focus
        >>> overlap = 2 * sp.pi * waist * desired_waist / sp.sqrt(wavelength**2 * delta_z**2 + sp.pi**2 * (waist**2 + desired_waist**2) ** 2)
        >>> hess = sp.simplify(sp.hessian(overlap, sp.Matrix([focus, waist])).subs({desired_focus: focus, desired_waist: waist}))
        >>> print(hess)
        Matrix([[-wavelength**2/(4*pi**2*waist**4), 0], [0, -1/waist**2]])
        """

        from .solver import PERFECT_OVERLAP

        if self.solution.overlap < PERFECT_OVERLAP:
            # the hessian based metrics are only meaningful if the setup is actually around a minimum
            # additionally, the analytical hessian calculation will be incorrect since it assumes that
            # the gradient is zero at the point to evaluate which is only true around stationary points
            warnings.warn(
                f"Imperfect mode overlap ({self.solution.overlap} < {PERFECT_OVERLAP} ≈ 1), Hessian based metrics may be meaningless.",
                stacklevel=2,
            )

        db = self.solution.candidate.problem.desired_beam
        base_hessian = np.diag([-(db.wavelength**2) / (4 * np.pi**2 * db.waist**4), -1 / db.waist**2])
        return self.focus_and_waist_jacobian.T @ base_hessian @ self.focus_and_waist_jacobian

    @cached_property
    def focus_and_waist_jacobian(self) -> np.ndarray:
        r"""The Jacobian :math:`\mathbf{J}` of the final beam focus and waist :math:`\mathbf{f}_{fw}(\mathbf{x})` with respect to
        the element positions :math:`\mathbf{x}` around the optimum :math:`\mathbf{x}^*`. The individual elements
        :math:`j_{ij}` of the Jacobian are given by:

        .. math::
            j_{ij} = \left.\frac{\partial f_{fw,i}(\mathbf{x})}{\partial x_j} \right|_{\mathbf{x} = \mathbf{x}^*}

        """
        setup = self.solution.setup

        jac_beam_parameter = setup._jac_ri_to_fw(setup.beam_parameters[-1], setup.initial_beam.wavelength)
        jac_res = np.zeros((len(setup.elements), 2))

        for i, (beam_before, (pos, element)) in reversed(
            list(enumerate(zip(setup.beams[:-1], setup.elements, strict=True)))
        ):
            jac_res[i] += jac_beam_parameter @ np.array([-1, 0])
            beam_param_before = beam_before.beam_parameter + (pos - beam_before.z_offset)
            jac_beam_parameter = jac_beam_parameter @ setup._jac_ri_to_ri(beam_param_before, element.matrix)
            jac_res[i] += jac_beam_parameter @ np.array([1, 0])

        return jac_res[self.solution.candidate.parametrized_setup.free_elements].T

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
        """The index :math:`i` of the degree of freedom with minimal sensitivity :math:`s_{ii}`."""

        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmin(diag_sensitivities))

    @cached_property
    def min_sensitivity(self) -> float:
        """The minimal sensitivity among all degrees of freedom."""

        return float(self.sensitivities[self.min_sensitivity_axis, self.min_sensitivity_axis])

    @cached_property
    def max_sensitivity_axis(self) -> int:
        """The index :math:`i` of the degree of freedom with maximal sensitivity :math:`s_{ii}`."""
        diag_sensitivities = np.abs(np.diag(self.sensitivities))
        return int(np.argmax(diag_sensitivities))

    @cached_property
    def max_sensitivity(self) -> float:
        """The maximal sensitivity among all degrees of freedom."""

        return float(self.sensitivities[self.max_sensitivity_axis, self.max_sensitivity_axis])

    # the vectors spanning the sub space in which the mode overlap stays approximately constant
    # equivalent to the null space of the hessian assuming the minor eigenvalues are zero
    @cached_property
    def const_space(self) -> list[np.ndarray]:
        r"""The basis vectors spanning the constant overlap sub-space around the optimum.

        Note:
            This is simply determined as the corresponding eigenvectors to all but the two largest eigenvalues
            of the Hessian :math:`\mathbf{H}`. These eigenvalues are generally not zero but they should by
            orders of magnitude smaller than the two largest ones.
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

    class SolutionSummary(TypedDict):
        """Summary dictionary for mode matching analysis."""

        overlap: float  #: The mode overlap of the solution.
        num_elements: int  #: The number of free elements (i.e. elements used for mode matching) in the setup.
        elements: list[Lens]  #: A list of the free elements (i.e. elements used for mode matching) in the setup.
        positions: np.ndarray  #: The positions of the free elements in the setup.
        min_sensitivity_axis: int  #: The index of the degree of freedom with minimal sensitivity.
        min_sensitivity: float  #: The minimal sensitivity.
        max_sensitivity_axis: int  #: The index of the degree of freedom with maximal sensitivity.
        max_sensitivity: float  #: The maximal sensitivity.
        min_cross_sens_pair: tuple[int, int]
        """The indices of the pair of degrees of freedom with minimal cross-sensitivity."""
        min_cross_sens: float  #: The minimal cross-sensitivity.
        min_cross_sens_direction: np.ndarray  #: The direction of the least cross-sensitive pair of degrees of freedom.
        min_coupling_pair: tuple[int, int]  #: The indices of the pair of degrees of freedom with minimal coupling.
        min_coupling: float  #: The minimal coupling.
        sensitivities: np.ndarray  #: The sensitivity matrix.
        couplings: np.ndarray  #: The coupling matrix.
        const_space: list[np.ndarray]  #: The basis vectors spanning the constant overlap sub-space.
        grad_focus: np.ndarray  #: The gradient of the final beam focus with respect to the element positions.
        grad_waist: np.ndarray  #: The gradient of the final beam waist with respect to the element positions.
        solution: "ModeMatchingSolution"  #: The analyzed mode matching solution.

    @cached_property
    def summary(self) -> SolutionSummary:
        """Summary dictionary of the analysis results."""

        sol = self.solution
        return {
            "overlap": sol.overlap,
            "num_elements": len(sol.positions),
            "elements": [sol.setup.elements[i][1] for i in sol.candidate.parametrized_setup.free_elements],
            "positions": sol.positions,
            "min_sensitivity_axis": self.min_sensitivity_axis,
            "min_sensitivity": self.min_sensitivity,
            "max_sensitivity_axis": self.max_sensitivity_axis,
            "max_sensitivity": self.max_sensitivity,
            "min_cross_sens_pair": self.min_cross_sens_pair,
            "min_cross_sens": self.min_cross_sens,
            "min_cross_sens_direction": self.min_cross_sens_direction,
            "min_coupling_pair": self.min_coupling_pair,
            "min_coupling": self.min_coupling,
            "sensitivities": self.sensitivities,
            "couplings": self.couplings,
            "const_space": self.const_space,
            "grad_focus": self.grad_focus,
            "grad_waist": self.grad_waist,
            "solution": sol,
        }

    # for use in solver.SolutionList
    @staticmethod
    def _summary_formatters(
        fraction_unit: FractionUnit | None = None,
        sensitivity_unit: SensitivityUnit | None = None,
        radial_unit: LengthUnit | None = None,
        axial_unit: LengthUnit | None = None,
    ):
        from .solver import ModeMatchingSolution

        fraction_unit = Config.get(fraction_unit, Config.Units.fraction)
        sensitivity_unit = Config.get(sensitivity_unit, Config.Units.sensitivity)
        radial_unit = Config.get(radial_unit, Config.Units.radial)
        axial_unit = Config.get(axial_unit, Config.Units.axial)

        axial_per_axial_unit = axial_unit / axial_unit
        radial_per_axial_unit = radial_unit / axial_unit
        unitless = Units.Fraction.UNITY

        return {
            "overlap": fraction_unit.format,
            "elements": lambda x: f"[{', '.join(str(e) for e in x)}]",  # pyright: ignore[reportGeneralTypeIssues]
            "positions": axial_unit.format,
            "min_sensitivity": sensitivity_unit.format,
            "max_sensitivity": sensitivity_unit.format,
            "min_cross_sens": sensitivity_unit.format,
            "min_cross_sens_direction": unitless.format,
            "min_coupling": fraction_unit.format,
            "sensitivities": sensitivity_unit.format,
            "couplings": fraction_unit.format,
            "const_space": unitless.format,
            "grad_focus": axial_per_axial_unit.format,
            "grad_waist": radial_per_axial_unit.format,
            "solution": lambda _: f"{ModeMatchingSolution.__name__}(...)",
        }

    class ElementInfo(TypedDict):
        """Information about an element in a mode matching solution."""

        element: Lens  #: The element object itself
        shape: str  #: ASCII representation of the element shape
        focal_length: float  #: Focal length of the element
        position: float  #: Position of the element in the setup
        clearance_left: float | None
        """Clearance to the previous element or shifting range boundary if the element is a free element"""
        clearance_right: float | None
        """Clearance to the next element or shifting range boundary if the element is a free element"""
        dof: int | None
        """Degree of freedom index of the element in the parametrized setup if it is a free element"""
        sensitivity: float | None
        """Sensitivity of the element, if it is a free element"""
        grad_focus: float | None
        """Gradient of the final beam focus with respect to the element position if it is a free element"""
        grad_waist: float | None
        """Gradient of the final beam waist with respect to the element position if it is a free element"""
        sensitivities: np.ndarray | None
        """Sensitivity vector of the element with respect to all degrees of freedom if it is a free element"""
        couplings: np.ndarray | None
        """Coupling vector of the element with respect to all degrees of freedom if it is a free element"""
        shifting_range: "ShiftingRange | None"
        """The shifting range the element belongs to if it is a free element"""

    @cached_property
    def element_summary(self) -> list[ElementInfo]:
        """A summary of the elements in the mode matching solution.

        Returns:
            A list of dictionaries containing the summary data for each element, see :class:`ElementInfo` for details.
        """

        candidate = self.solution.candidate

        index_to_range: list[ShiftingRange | None] = []  # mapping from element index to potential shifting range
        for elem_or_index in candidate.problem.interleaved_elements:
            match elem_or_index:
                case int(index):
                    index_to_range.extend([candidate.problem.ranges[index]] * len(candidate.populations[index]))
                case (_, _):
                    index_to_range.append(None)

        infos = []
        elements = self.solution.setup.elements
        for i, (pos, element) in enumerate(elements):
            entry = {
                "element": element,
                "shape": element.shape,
                "focal_length": element.focal_length,
                "position": pos,
                "clearance_left": None,
                "clearance_right": None,
                "dof": None,
                "sensitivity": None,
                "grad_focus": None,
                "grad_waist": None,
                "sensitivities": None,
                "couplings": None,
                "shifting_range": None,
                # TODO add focus and waist sensitivities?
            }
            if (shifting_range := index_to_range[i]) is not None:
                entry["shifting_range"] = shifting_range
                entry["dof"] = candidate.parametrized_setup.free_elements.index(i)
                entry["sensitivity"] = self.sensitivities[entry["dof"], entry["dof"]]
                entry["clearance_left"] = pos - shifting_range.left - element.left_margin
                entry["clearance_right"] = shifting_range.right - pos - element.right_margin
                if i - 1 >= 0 and index_to_range[i - 1] == shifting_range:
                    prev_pos, prev_element = elements[i - 1]
                    entry["clearance_left"] = pos - prev_pos - prev_element.right_margin - element.left_margin
                if i + 1 < len(elements) and index_to_range[i + 1] == shifting_range:
                    next_pos, next_element = elements[i + 1]
                    entry["clearance_right"] = next_pos - pos - element.right_margin - next_element.left_margin
                entry["grad_focus"] = self.grad_focus[entry["dof"]]
                entry["grad_waist"] = self.grad_waist[entry["dof"]]
                entry["sensitivities"] = self.sensitivities[entry["dof"], :]
                entry["couplings"] = self.couplings[entry["dof"], :]

            infos.append(entry)

        return infos

    def element_summary_df(
        self,
        axial_unit: LengthUnit | None = None,
        radial_unit: LengthUnit | None = None,
        fraction_unit: FractionUnit | None = None,
        sensitivity_unit: SensitivityUnit | None = None,
    ) -> pd.DataFrame:
        """Create a summary DataFrame of the elements in the solution.

        Args:
            axial_unit: Unit to use for the axial quantities along the beam, i.e., the coordinate along the beam.
                If ``None``, this defaults to :attr:`Config.Units.axial <corset.config.Config.Units.axial>`.
            radial_unit: Unit to use for the radial across the beam, i.e., the beam radius.
                If ``None``, this defaults to :attr:`Config.Units.radial <corset.config.Config.Units.radial>`.
            fraction_unit: Unit for fractional quantities, i.e., the mode overlap and coupling coefficients.
                If ``None``, this defaults to :attr:`Config.Units.fraction <corset.config.Config.Units.fraction>`.
            sensitivity_unit: Unit for sensitivity quantities, i.e., the overlap lost for a certain squared displacement.
                If ``None``, this defaults to :attr:`Config.Units.sensitivity <corset.config.Config.Units.sensitivity>`.

        Returns:
            A DataFrame containing the summary data for each element, see :meth:`element_summary` for details.
        """

        fraction_unit = Config.get(fraction_unit, Config.Units.fraction)
        sensitivity_unit = Config.get(sensitivity_unit, Config.Units.sensitivity)
        radial_unit = Config.get(radial_unit, Config.Units.radial)
        axial_unit = Config.get(axial_unit, Config.Units.axial)
        axial_per_axial_unit = axial_unit / axial_unit
        radial_per_axial_unit = radial_unit / axial_unit

        formatters = {
            "focal_length": axial_unit.format,
            "position": axial_unit.format,
            "sensitivity": sensitivity_unit.format,
            "clearance_left": axial_unit.format,
            "clearance_right": axial_unit.format,
            "dof": lambda x: str(int(x)),  # handle pandas converting column to float when there are NaNs
            "grad_focus": axial_per_axial_unit.format,
            "grad_waist": radial_per_axial_unit.format,
            "sensitivities": sensitivity_unit.format,
            "couplings": fraction_unit.format,
        }

        return FormattedDataFrame(self.element_summary, formatters=formatters)  # pyright: ignore[reportCallIssue]
