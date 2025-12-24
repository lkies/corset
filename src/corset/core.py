"""Core classes for representing and simulating Gaussian beams and optical setups.

Setups are represented as :class:`OpticalSetup` instances which propagate an initial :class:`Beam`
through a sequence of :class:`Lens` elements. This yields a piecewise defined beam radius made
from Gaussian beam segments between the elements. The individual beams are represented using
a complex beam parameter combined with an axial offset and wavelength. The beams are propagated
using the ray transfer matrix method for Gaussian beams, see `here <https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis#Gaussian_beams>`_.
"""

from dataclasses import InitVar, dataclass
from functools import cached_property
from itertools import pairwise
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .plot import OpticalSetupPlot, fig_to_png, plot_optical_setup


# TODO should beam include wavelength or should it be part of the larger setup?
# TODO refractive index?
@dataclass(frozen=True)
class Beam:
    """Paraxial Gaussian beam representation.

    Implements :meth:`_repr_png_` to show a plot of the beam radius in IPython environments.
    """

    beam_parameter: complex
    """Complex beam parameter :math:`q = z - z_0 + i z_R` defined at position :math:`z`
    with focus at :math:`z_0` and Rayleigh range :math:`z_R`."""
    z_offset: float  #: Axial position at which the ray is defined
    wavelength: float  #: Wavelength of the beam
    gauss_cov: np.ndarray | None = None  #: 2x2 covariance matrix for focus position and waist
    # range: tuple[float, float] # TODO is this necessary

    @cached_property
    def waist(self) -> float:
        """Waist radius"""
        return np.sqrt(self.rayleigh_range * self.wavelength / np.pi)

    @cached_property
    def rayleigh_range(self) -> float:
        """Rayleigh range"""
        return abs(self.beam_parameter.imag)

    @cached_property
    def focus(self) -> float:
        """Axial position of the beam focus i.e. waist position"""
        return self.z_offset - self.beam_parameter.real

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius at axial position(s).

        Args:
            z: Axial position(s) where the beam radius is evaluated.

        Returns:
            Beam radius at the specified axial position(s).
        """
        return self.waist * np.sqrt(1 + ((z - self.focus) / self.rayleigh_range) ** 2)

    def radius_dev(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius (standard) deviation at axial position(s).

        Args:
            z: Axial position(s) where the beam radius deviation is evaluated.

        Returns:
            Beam radius deviation at the specified axial position(s).

        Raises:
            ValueError: If the covariance matrix is not defined for this beam.
        """
        # derivation:
        """
        >>> import sympy as sp
        >>> z, focus = sp.symbols('z focus', real=True)
        >>> waist, wavelength = sp.symbols('self.waist self.wavelength', real=True, positive=True)
        >>> rayleigh_range = sp.pi * waist**2 / wavelength
        >>> radius = waist * sp.sqrt(1 + ((z - focus) / rayleigh_range) ** 2)
        >>> res = radius.diff(sp.Matrix([focus, waist])).simplify()
        >>> print(res[0])
        self.wavelength**2*(focus - z)/(pi*self.waist*sqrt(pi**2*self.waist**4 + self.wavelength**2*(focus - z)**2))
        >>> print(res[1])
        (pi**2*self.waist**4 - self.wavelength**2*(focus - z)**2)/(pi*self.waist**2*sqrt(pi**2*self.waist**4 + self.wavelength**2*(focus - z)**2))
        """
        if self.gauss_cov is None:
            raise ValueError("Covariance matrix is not defined for this beam.")
        wl = self.wavelength
        common_denom = np.pi * self.waist * np.sqrt(np.pi**2 * self.waist**4 + wl**2 * (self.focus - z) ** 2)
        jac = np.array(
            [
                [wl**2 * (self.focus - z) / common_denom],
                [(np.pi**2 * self.waist**4 - wl**2 * (self.focus - z) ** 2) / common_denom / self.waist],
            ]
        )
        if np.isscalar(z):
            return float(np.sqrt(jac.T @ self.gauss_cov @ jac))
        else:
            return np.sqrt(np.einsum("ij,jk,ik->i", jac, self.gauss_cov, jac))

    @classmethod
    def from_gauss(cls, focus: float, waist: float, wavelength: float, cov: np.ndarray | None = None) -> "Beam":
        """Create a Beam instance from focus position and waist.

        Args:
            focus: The axial position of the beam focus.
            waist: The beam waist radius.
            wavelength: The wavelength of the beam.
            cov: Optional 2x2 covariance matrix for the focus and waist.
        Returns:
            Beam instance.
        """

        rayleigh_range = np.pi * (waist**2) / wavelength
        return cls(beam_parameter=1j * rayleigh_range, z_offset=focus, wavelength=wavelength, gauss_cov=cov)

    @classmethod
    def fit(
        cls, positions: np.ndarray, radii: np.ndarray, wavelength: float, p0: tuple[float, float] | None = None
    ) -> "Beam":
        """Fit a Gaussian beam radius to measured data.

        This uses scipy.optimize.curve_fit to estimate the focus position
        and waist given arrays of axial `positions` and measured `radii`.

        Args:
            positions: Axial positions where radii were measured.
            radii: Measured beam radii corresponding to the positions.
            wavelength: Wavelength used to relate waist and Rayleigh range.
            p0: Initial guess for (focus, waist). If omitted a simple heuristic is used.

        Returns:
            Beam instance fitted to the data.
        """
        if p0 is None:  # TODO is this a good idea?
            p0 = (positions[np.argmin(radii)], np.min(radii))
        # yes using the class itself is pretty inefficient but its convenient and not performance critical
        (focus, waist), cov = curve_fit(
            lambda z, f, w: cls.from_gauss(f, w, wavelength).radius(z), positions, radii, p0=p0
        )
        return cls.from_gauss(focus, waist, wavelength, cov=cov)

    def plot(self, **kwargs) -> OpticalSetupPlot:  # pyright: ignore[reportPrivateImportUsage]
        """Plot the beam as part of an optical setup with no other elements.

        Args:
            **kwargs: Keyword arguments forwarded to :func:`OpticalSetup.plot`.

        Returns:
            OpticalSetupPlot instance for further customization.
        """

        return OpticalSetup(self, []).plot(**kwargs)

    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)


# TODO general Element class?
@dataclass(frozen=True)
class ThinLens:
    """Thin lens element including additional information."""

    focal_length: float  #: Focal length of the lens
    left_margin: float = 0.0  #: Physical size to the left of the focal plane
    right_margin: float = 0.0  #: Physical size to the right of the focal plane
    name: str | None = None  #: Name for reference and plotting

    def __post_init__(self):
        if self.focal_length == 0:
            raise ValueError("Focal length cannot be zero.")
        if self.left_margin + self.right_margin < 0:
            raise ValueError("Lens must have non-negative physical size.")  # focal plane outside physical lens is ok

    @cached_property
    def matrix(self) -> np.ndarray:
        """ABCD matrix of the lens element."""
        return np.array([[1, 0], [-1 / self.focal_length, 1]])

    def __str__(self) -> str:
        return self.name if self.name is not None else f"f={round(self.focal_length*1e3)}mm"


@dataclass(frozen=True)
class ThickLens:
    """Thick lens element including additional information.

    A positive radius of curvature is a convex surface while a negative radius is concave.
    Use :attr:`ThickLens.FLAT` to represent a flat surface. This is an alias for ``float('inf')``.
    """

    in_roc: float  #: Input surface radius of curvature, positive is convex
    out_roc: float  #: Output surface radius of curvature, positive is convex
    thickness: float  #: Thickness of the lens
    refractive_index: float  #: Refractive index of the lens material
    left_margin: float = 0.0  #: Physical size to the left of the lens center
    right_margin: float = 0.0  #: Physical size to the right of the lens center
    name: str | None = None  #: Name for reference and plotting

    FLAT: ClassVar[float] = float("inf")  #: Surface radius representing a flat surface

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError("Lens thickness must be positive.")
        if self.refractive_index <= 1:
            raise ValueError("Refractive index must be greater than 1.")
        if self.left_margin + self.right_margin < 0:
            raise ValueError("Lens must have non-negative physical size.")

    @cached_property
    def matrix(self) -> np.ndarray:
        """ABCD matrix of the lens element."""
        n2 = self.refractive_index
        in_surface = np.array([[1, 0], [(1 - n2) / (self.in_roc * n2), 1 / n2]])
        propagation = np.array([[1, self.thickness], [0, 1]])
        out_surface = np.array([[1, 0], [(n2 - 1) / (-self.out_roc), n2]])
        thickness_correction = np.array([[1, -self.thickness / 2], [0, 1]])
        return thickness_correction @ out_surface @ propagation @ in_surface @ thickness_correction

    @cached_property
    def focal_length(self) -> float:
        """Approximate focal length of the thick lens."""
        n2, r1, r2 = self.refractive_index, self.in_roc, self.out_roc
        return 1 / ((n2 - 1) * (1 / r1 + 1 / r2 + ((1 - n2) * self.thickness) / (n2 * r1 * r2)))

    def __str__(self) -> str:
        return self.name if self.name is not None else f"f≈{round(self.focal_length*1e3)}mm"


Lens = ThinLens | ThickLens  #: Lens type union


@dataclass(frozen=True)
class OpticalSetup:
    """Optical setup described by an initial beam and a sequence of elements.

    Implements :meth:`_repr_png_` to show a plot of the optical setup in IPython environments.
    """

    initial_beam: Beam  #: Initial beam before left most element
    elements: list[tuple[float, Lens]]  #: Optical elements as (position, element) tuples sorted by position
    validate: InitVar[bool] = True  #: Validate that elements are sorted by position

    def __post_init__(self, validate: bool) -> None:
        if validate and not all(left < right for (left, _), (right, _) in pairwise(self.elements)):
            raise ValueError("Optical elements must be sorted by position.")

    # TODO eliminate this and just put it into beams?
    @cached_property
    def beam_parameters(self) -> list[complex]:
        """Compute the ray vectors between elements including before the first element and after the last."""
        q = self.initial_beam.beam_parameter
        prev_pos = self.initial_beam.z_offset
        beam_parameters = [q]
        for pos, element in self.elements:
            q += pos - prev_pos  # free space propagation
            vec = element.matrix @ np.array([q, 1])  # lens transformation
            q = vec[0] / vec[1]  # normalize
            beam_parameters.append(q)
            prev_pos = pos
        return beam_parameters

    @cached_property
    def gauss_covariances(self) -> list[np.ndarray | None]:
        """Compute the covariance matrices between elements including before the first element and after the last."""
        if self.initial_beam.gauss_cov is None:
            return [None] * (len(self.elements) + 1)
        wavelength = self.initial_beam.wavelength
        cov = self.initial_beam.gauss_cov
        prev_pos = self.initial_beam.z_offset
        gauss_covariances = [cov]
        # for (pos, element), (q_prev, q_after) in zip(self.elements, pairwise(self.beam_parameters), strict=True):
        for (pos, element), (prev_beam, beam) in zip(self.elements, pairwise(self.beams_fast), strict=True):
            q_before = prev_beam.beam_parameter + (pos - prev_pos)  # beam parameter before element
            jac_fw_to_ri = self._jac_fw_to_ri(prev_beam.waist, wavelength)
            jac_ri_to_ri = self._jac_ri_to_ri(q_before, element.matrix)
            jac_ri_to_fw = self._jac_ri_to_fw(beam.beam_parameter, wavelength)
            jac_total = jac_ri_to_fw @ jac_ri_to_ri @ jac_fw_to_ri
            cov = jac_total @ cov @ jac_total.T
            gauss_covariances.append(cov)
            prev_pos = pos
        return gauss_covariances  # pyright: ignore[reportReturnType]

    @cached_property
    def beams_fast(self) -> list[Beam]:
        """Compute the Beam instances without propagating covariances."""
        return [self.initial_beam] + [
            Beam(beam_parameter=param, z_offset=pos, wavelength=self.initial_beam.wavelength)
            for (pos, _), param in zip(self.elements, self.beam_parameters[1:], strict=True)
        ]

    @cached_property
    def beams(self) -> list[Beam]:
        """Compute the Beam instances between elements including before the first element and after the last."""
        return [self.initial_beam] + [
            Beam(beam_parameter=param, z_offset=pos, wavelength=self.initial_beam.wavelength, gauss_cov=cov)
            for (pos, _), param, cov in zip(
                self.elements, self.beam_parameters[1:], self.gauss_covariances[1:], strict=True
            )
        ]

    # TODO optimize this to pass data to beams in batches?
    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius at axial position(s)."""
        if not np.isscalar(z):
            return np.array([self.radius(zi) for zi in z])  # pyright: ignore[reportGeneralTypeIssues]
        index = np.searchsorted([pos for pos, _ in self.elements], z)
        return self.beams_fast[index].radius(z)  # pyright: ignore[reportArgumentType, reportCallIssue]

    # TODO only specify default k in one place?
    def radius_dev(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius confidence interval at axial position(s)."""
        if not np.isscalar(z):
            return np.array([self.radius_dev(zi) for zi in z])  # pyright: ignore[reportGeneralTypeIssues]
        index = np.searchsorted([pos for pos, _ in self.elements], z)
        return self.beams[index].radius_dev(z)  # pyright: ignore[reportArgumentType, reportCallIssue]

    plot = plot_optical_setup  #: Plot the optical setup, see :func:`corset.plot.plot_optical_setup`

    # TODO cache this (and the other repr png functions)?
    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)

    # TODO include derivation code
    @staticmethod
    def _jac_fw_to_ri(waist: float, wavelength: float) -> np.ndarray:
        """Jacobian from [waist, wavelength] to [Re(q0p), Im(q0p)]
        ie from Gaussian beam parameters to complex beam parameter which
        has already been propagated to the new position.

        Args:
            waist: Beam waist radius.
            wavelength: Wavelength of the beam.

        Returns:
            2x2 Jacobian matrix.
        """
        # derivation:
        """
        >>> import sympy as sp
        >>> w0, f0, wl, z0, z1 = sp.symbols('waist focus wavelength z_{0} z_{1}', real=True)
        >>> q0 = z0 - f0 + sp.I * sp.pi * w0**2 / wl
        >>> q0p = q0 + z1 - z0 # propagate to just before element
        >>> print(sp.Matrix([sp.re(q0p), sp.im(q0p)]).jacobian(sp.Matrix([f0, w0])))
        Matrix([[-1, 0], [0, 2*pi*waist/wavelength]])
        """
        return np.array([[-1, 0], [0, 2 * np.pi * waist / wavelength]])

    @staticmethod
    def _jac_ri_to_ri(q0: complex, mat: np.ndarray) -> np.ndarray:
        """Jacobian from [Re(q0), Im(q0)] to [Re(q1), Im(q1)]
        ie from complex beam parameter before an optical element
        to complex beam parameter after the element.

        Args:
            q0: Complex beam parameter before the element.
            mat: 2x2 ABCD matrix of the optical element.

        Returns:
            2x2 Jacobian matrix.
        """
        # derivation:
        """
        >>> import sympy as sp
        >>> q0r, q0i, A, B, C, D = sp.symbols('q0.real q0.imag A B C D', real=True)
        >>> q0 = q0r + sp.I * q0i
        >>> q1 = ((A * q0 + B) / (C * q0 + D)).simplify()
        >>> jac = sp.simplify(sp.Matrix([sp.re(q1), sp.im(q1)]).jacobian(sp.Matrix([q0r, q0i])))
        >>> print("jac[0,0].factor())
        -(A*D - B*C)*(C*q0.imag - C*q0.real - D)*(C*q0.imag + C*q0.real + D)/(C**2*q0.imag**2 + C**2*q0.real**2 + 2*C*D*q0.real + D**2)**2
        >>> print("jac[0,1].factor())
        2*C*q0.imag*(A*D - B*C)*(C*q0.real + D)/(C**2*q0.imag**2 + C**2*q0.real**2 + 2*C*D*q0.real + D**2)**2
        >>> assert jac[0,0].factor() == jac[1,1].factor()
        >>> assert jac[1,0].factor() == -jac[0,1].factor()
        """
        A, B, C, D = mat.flat  # noqa: N806
        res = np.zeros((2, 2))
        # fmt: off
        res[0,0] = -(A*D - B*C)*(C*q0.imag - C*q0.real - D)*(C*q0.imag + C*q0.real + D)/(C**2*q0.imag**2 + C**2*q0.real**2 + 2*C*D*q0.real + D**2)**2
        res[0,1] = 2*C*q0.imag*(A*D - B*C)*(C*q0.real + D)/(C**2*q0.imag**2 + C**2*q0.real**2 + 2*C*D*q0.real + D**2)**2
        # fmt: on
        res[1, 0] = -res[0, 1]
        res[1, 1] = res[0, 0]
        return res

    @staticmethod
    def _jac_ri_to_fw(q1: complex, wavelength: float) -> np.ndarray:
        """Jacobian from [Re(q1), Im(q1)] to [waist, wavelength]
        ie from complex beam parameter after an optical element
        to Gaussian beam parameters.

        Args:
            q1: Complex beam parameter after the element.
            wavelength: Wavelength of the beam.

        Returns:
            2x2 Jacobian matrix.
        """
        # derivation:
        """
        >>> import sympy as sp
        >>> q1r, q1i, z1, wl = sp.symbols('q1.real q1.imag z_{1} wavelength', real=True, positive=True)
        >>> f1 = z1 - q1r
        >>> w1 = sp.sqrt(q1i * wl / sp.pi)
        >>> print(sp.Matrix([f1, w1]).jacobian(sp.Matrix([q1r, q1i])))
        Matrix([[-1, 0], [0, sqrt(wavelength)/(2*sqrt(pi)*sqrt(q1.imag))]])
        """
        return np.array([[-1, 0], [0, np.sqrt(wavelength) / (2 * np.sqrt(np.pi) * np.sqrt(q1.imag))]])
