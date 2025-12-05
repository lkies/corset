"""Core classes for representing and simulating Gaussian beams and optical setups."""

from dataclasses import InitVar, dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .plot import OpticalSetupPlot, fig_to_png, plot_optical_setup


# TODO should beam include wavelength or should it be part of the larger setup?
# TODO refractive index?
@dataclass(frozen=True)
class Beam:
    """Paraxial Gaussian beam representation."""

    ray: np.ndarray  #: 2-element array representing the complex ray vector [q, 1]
    z_offset: float  #: Axial position at which the ray is defined
    wavelength: float  #: Wavelength of the beam
    gauss_cov: np.ndarray | None = None  #: Optional covariance matrix for focus position and waist
    # range: tuple[float, float] # TODO is this necessary

    @property
    def waist(self) -> float:
        """Waist radius"""
        return np.sqrt(self.rayleigh_range * self.wavelength / np.pi)

    @property
    def rayleigh_range(self) -> float:
        """Rayleigh range"""
        return np.imag(self.ray[0])

    @property
    def focus(self) -> float:
        """Axial position of the beam focus i.e. waist position"""
        return self.z_offset - np.real(self.ray[0])

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
        return cls(ray=np.array([0 + 1j * rayleigh_range, 1]), z_offset=focus, wavelength=wavelength, gauss_cov=cov)

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
class Lens:
    """Thin lens element including additional information."""

    focal_length: float  #: Focal length of the lens
    left_margin: float = 0  #: Physical size to the left of the focal plane
    right_margin: float = 0  #: Physical size to the right of the focal plane
    name: str | None = None  #: Name for reference and plotting

    @cached_property
    def matrix(self) -> np.ndarray:
        """ABCD matrix of the lens element."""
        return np.array([[1, 0], [-1 / self.focal_length, 1]])

    def __str__(self) -> str:
        return self.name if self.name is not None else f"f={round(self.focal_length*1e3)}mm"


@dataclass(frozen=True)
class OpticalSetup:
    """Optical setup described by an initial beam and a sequence of elements."""

    initial_beam: Beam  #: Initial beam before left most element
    elements: list[tuple[float, Lens]]  #: Optical elements as (position, element) tuples sorted by position
    validate: InitVar[bool] = True  #: Validate that elements are sorted by position

    def __post_init__(self, validate: bool) -> None:
        if validate and not all(
            left < right for (left, _), (right, _) in zip(self.elements[:-1], self.elements[1:], strict=True)
        ):
            raise ValueError("Optical elements must be sorted by position.")

    # TODO eliminate this and just put it into beams?
    @cached_property
    def rays(self) -> list[np.ndarray]:
        """Compute the ray vectors between elements including before the first element and after the last."""
        ray = self.initial_beam.ray
        prev_pos = self.initial_beam.z_offset
        rays = [ray]
        for pos, element in self.elements:
            distance = pos - prev_pos
            ray = element.matrix @ self._free_space(distance) @ ray
            ray = np.array([ray[0] / ray[1], 1])  # normalize
            rays.append(ray)
            prev_pos = pos
        return rays

    @cached_property
    def beams(self) -> list[Beam]:
        """Compute the Beam instances between elements including before the first element and after the last."""
        return [self.initial_beam] + [
            Beam(ray=ray, z_offset=pos, wavelength=self.initial_beam.wavelength)
            for (pos, _), ray in zip(self.elements, self.rays[1:], strict=True)
        ]

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the beam radius at axial position(s)."""
        if not np.isscalar(z):
            return np.array([self.radius(zi) for zi in z])  # pyright: ignore[reportGeneralTypeIssues]
        index = np.searchsorted([pos for pos, _ in self.elements], z)
        return self.beams[index].radius(z)  # pyright: ignore[reportArgumentType, reportCallIssue]

    plot = plot_optical_setup  #: Plot the optical setup, see :func:`plot_optical_setup`

    # TODO cache this (and the other repr png functions)?
    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)

    @staticmethod
    def _free_space(distance: float) -> np.ndarray:
        return np.array([[1, distance], [0, 1]])
