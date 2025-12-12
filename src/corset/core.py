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
    gauss_cov: np.ndarray | None = None  #: Covariance matrix for focus position and waist
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

    @classmethod
    def from_gauss(cls, focus: float, waist: float, wavelength: float, cov: np.ndarray | None = None) -> "Beam":
        """Create a Beam instance from focus position and waist.

        Args:
            focus: The axial position of the beam focus.
            waist: The beam waist radius.
            wavelength: The wavelength of the beam.
            cov: Optional covariance matrix for the beam parameters.
        Returns:
            Beam instance.
        """

        rayleigh_range = np.pi * (waist**2) / wavelength
        return cls(beam_parameter=1j * rayleigh_range, z_offset=focus, wavelength=wavelength, gauss_cov=cov)

    @classmethod
    def fit(cls, zs: np.ndarray, rs: np.ndarray, wavelength: float, p0: tuple[float, float] | None = None) -> "Beam":
        """Fit a Gaussian beam radius to measured data.

        This uses scipy.optimize.curve_fit to estimate the focus position
        and waist given arrays of axial positions `zs` and measured
        radii `rs`.

        Args:
            zs: Axial positions where radii were measured.
            rs: Measured beam radii corresponding to `zs`.
            wavelength: Wavelength used to relate waist and Rayleigh range.
            p0: Initial guess for (focus, waist). If omitted a simple heuristic is used.

        Returns:
            Beam instance fitted to the data..
        """
        if p0 is None:  # TODO is this a good idea?
            p0 = (zs[np.argmin(rs)], np.min(rs))
        # yes using the class itself is pretty inefficient but its convenient and not performance critical
        (focus, waist), cov = curve_fit(lambda z, f, w: cls.from_gauss(f, w, wavelength).radius(z), zs, rs, p0=p0)
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
    left_margin: float = 0  #: Physical size to the left of the focal plane
    right_margin: float = 0  #: Physical size to the right of the focal plane
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
    """Thick lens element including additional information."""

    in_roc: float  #: Input surface radius of curvature
    out_roc: float  #: Output surface radius of curvature
    thickness: float  #: Thickness of the lens
    refractive_index: float  #: Refractive index of the lens material
    left_margin: float = 0  #: Physical size to the left of the lens center
    right_margin: float = 0  #: Physical size to the right of the lens center
    name: str | None = None  #: Name for reference and plotting

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
        out_surface = np.array([[1, 0], [(n2 - 1) / (self.out_roc), n2]])
        thickness_correction = np.array([[1, -self.thickness / 2], [0, 1]])
        return thickness_correction @ out_surface @ propagation @ in_surface @ thickness_correction

    @cached_property
    def focal_length(self) -> float:
        """Approximate focal length of the thick lens."""
        n2, r1, r2 = self.refractive_index, self.in_roc, self.out_roc
        return 1 / ((n2 - 1) * (1 / r1 - 1 / r2 + ((n2 - 1) * self.thickness) / (n2 * r1 * r2)))

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
    def beams(self) -> list[Beam]:
        """Compute the Beam instances between elements including before the first element and after the last."""
        return [self.initial_beam] + [
            Beam(beam_parameter=param, z_offset=pos, wavelength=self.initial_beam.wavelength)
            for (pos, _), param in zip(self.elements, self.beam_parameters[1:], strict=True)
        ]

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius at axial position(s)."""
        if not np.isscalar(z):
            return np.array([self.radius(zi) for zi in z])  # pyright: ignore[reportGeneralTypeIssues]
        index = np.searchsorted([pos for pos, _ in self.elements], z)
        return self.beams[index].radius(z)  # pyright: ignore[reportArgumentType, reportCallIssue]

    plot = plot_optical_setup  #: Plot the optical setup, see :func:`corset.plot.plot_optical_setup`

    # TODO cache this (and the other repr png functions)?
    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)
