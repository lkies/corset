from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# TODO should beam include wavelength or should it be part of the larger setup?
# TODO refractive index?
@dataclass(frozen=True)
class Beam:
    ray: np.ndarray  # [x, theta]
    z_offset: float
    wavelength: float
    # range: tuple[float, float] # TODO is this necessary

    @property
    def waist(self) -> float:
        return self.wavelength / (np.pi * abs(self.ray[1]))  # [x, theta]

    @property
    def rayleigh_range(self) -> float:
        if self.wavelength is None:
            raise ValueError("Wavelength must be set to compute rayleigh range")
        return np.pi * (self.waist**2) / self.wavelength

    @property
    def focus(self) -> float:
        return self.z_offset - (self.ray[0] / self.ray[1])

    def radius(self, z: float | np.ndarray) -> float | np.ndarray:
        return self.waist * np.sqrt(1 + ((z - self.focus) / self.rayleigh_range) ** 2)

    @staticmethod
    def from_gauss(waist: float, z_offset: float, wavelength: float) -> "Beam":
        return Beam(ray=np.array([0, wavelength / (np.pi * waist)]), z_offset=z_offset, wavelength=wavelength)


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

    @cached_property
    def matrix(self) -> np.ndarray:
        return thin_lens(self.focal_length)


@dataclass(frozen=True)
class OpticalSetup:
    initial_beam: Beam
    elements: list[tuple[float, Lens]]

    @cached_property
    def rays(self) -> list[np.ndarray]:
        prev_ray = self.initial_beam.ray
        prev_pos = self.initial_beam.z_offset
        rays = [prev_ray]
        for pos, element in self.elements:
            distance = pos - prev_pos
            prev_ray = element.matrix @ free_space(distance) @ prev_ray
            rays.append(prev_ray)
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
        return self.beams[index].radius(z)  # pyright: ignore[reportArgumentType]

    def plot(
        self,
        ax: plt.Axes,  # pyright: ignore[reportPrivateImportUsage]
        *,
        points: None | int | np.ndarray = None,
        limits: tuple[float, float] | None = None,
        beam_kwargs: dict = {"alpha": 0.5},  # noqa: B006
        lens_kwargs: dict = {"zorder": 100},  # noqa: B006
    ) -> None:
        lens_positions = [pos for pos, _ in self.elements]

        if isinstance(points, np.ndarray):
            zs = points
        else:
            if not limits:
                all_positions = [
                    self.beams[0].focus - self.beams[0].rayleigh_range,
                    *lens_positions,
                    self.beams[-1].focus + self.beams[0].rayleigh_range,
                ]
                limits = (min(all_positions), max(all_positions))

            num_points = points if isinstance(points, int) else 500
            zs = np.linspace(limits[0], limits[1], num_points)

        rs = self.radius(zs)
        ax.fill_between(zs, -rs, rs, **beam_kwargs)
        r_max = np.max(rs)
        ax.vlines(lens_positions, -r_max * 1.1, r_max * 1.1, **lens_kwargs)
        ax.set_xlabel("z in mm")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e3:.0f}"))

        ax.set_ylabel(r"w(z) in $\mathrm{\mu m}$")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e6:.0f}"))
