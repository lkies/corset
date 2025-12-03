from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .plot import OpticalSetupPlot, fig_to_png, plot_optical_setup


# TODO should beam include wavelength or should it be part of the larger setup?
# TODO refractive index?
@dataclass(frozen=True)
class Beam:
    ray: np.ndarray  # [(z-focus)+ j*rayleigh_range, 0]
    z_offset: float
    wavelength: float
    gauss_cov: np.ndarray | None = None
    # range: tuple[float, float] # TODO is this necessary

    @property
    def waist(self) -> float:
        return np.sqrt(self.rayleigh_range * self.wavelength / np.pi)

    @property
    def rayleigh_range(self) -> float:
        return np.imag(self.ray[0])

    @property
    def focus(self) -> float:
        return self.z_offset - np.real(self.ray[0])

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        return self.waist * np.sqrt(1 + ((z - self.focus) / self.rayleigh_range) ** 2)

    @classmethod
    def from_gauss(cls, focus: float, waist: float, wavelength: float, cov: np.ndarray | None = None) -> "Beam":
        rayleigh_range = np.pi * (waist**2) / wavelength
        return cls(ray=np.array([0 + 1j * rayleigh_range, 1]), z_offset=focus, wavelength=wavelength, gauss_cov=cov)

    @classmethod
    def fit(cls, zs: np.ndarray, rs: np.ndarray, wavelength: float, p0: tuple[float, float] | None = None) -> "Beam":
        if p0 is None:  # TODO is this a good idea?
            p0 = (zs[np.argmin(rs)], np.min(rs))
        # yes using the class itself is pretty inefficient but its convenient and not performance critical
        (focus, waist), cov = curve_fit(lambda z, f, w: cls.from_gauss(f, w, wavelength).radius(z), zs, rs, p0=p0)
        return cls.from_gauss(focus, waist, wavelength, cov=cov)

    def plot(self, **kwargs) -> OpticalSetupPlot:  # pyright: ignore[reportPrivateImportUsage]
        return OpticalSetup(self, []).plot(**kwargs)

    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)


def free_space(distance: float) -> np.ndarray:
    return np.array([[1, distance], [0, 1]])


def thin_lens(focal_length: float) -> np.ndarray:
    return np.array([[1, 0], [-1 / focal_length, 1]])


# TODO general Element class?
@dataclass(frozen=True)
class Lens:
    focal_length: float
    left_margin: float = 0  # TODO default values?
    right_margin: float = 0  # TODO default values?
    name: str | None = None

    @cached_property
    def matrix(self) -> np.ndarray:
        return thin_lens(self.focal_length)

    def __str__(self) -> str:
        return self.name if self.name is not None else f"f={round(self.focal_length*1e3)}mm"


@dataclass(frozen=True)
class OpticalSetup:
    initial_beam: Beam
    elements: list[tuple[float, Lens]]

    # TODO eliminate this and just put it into beams?
    @cached_property
    def rays(self) -> list[np.ndarray]:
        ray = self.initial_beam.ray
        prev_pos = self.initial_beam.z_offset
        rays = [ray]
        for pos, element in self.elements:
            distance = pos - prev_pos
            ray = element.matrix @ free_space(distance) @ ray
            ray = np.array([ray[0] / ray[1], 1])  # normalize
            rays.append(ray)
            prev_pos = pos
        return rays

    @cached_property
    def beams(self) -> list[Beam]:
        return [self.initial_beam] + [
            Beam(ray=ray, z_offset=pos, wavelength=self.initial_beam.wavelength)
            for (pos, _), ray in zip(self.elements, self.rays[1:], strict=True)
        ]

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        if not np.isscalar(z):
            return np.array([self.radius(zi) for zi in z])  # pyright: ignore[reportGeneralTypeIssues]
        index = np.searchsorted([pos for pos, _ in self.elements], z)
        return self.beams[index].radius(z)  # pyright: ignore[reportArgumentType, reportCallIssue]

    plot = plot_optical_setup

    # TODO cache this (and the other repr png functions)?
    def _repr_png_(self) -> bytes:
        fig, ax = plt.subplots()
        self.plot(ax=ax)
        return fig_to_png(fig)
