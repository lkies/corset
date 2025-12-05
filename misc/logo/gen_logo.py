# type: ignore  # noqa: PGH003
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from corset.core import Beam

colors = [
    "#cd001a",
    "#ef6a00",
    "#f2cd00",
    "#79c300",
    "#1961ae",
    "#61007d",
]


def make_logo(filename, dark, crop=None):
    # setup plot
    figsize = (8, 4.5) if not crop else (crop[0][1] - crop[0][0], crop[1][1] - crop[1][0])
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-4.5, 4.5)
    ax.axis("off")
    fig.set_facecolor("none")

    beam = Beam.from_gauss(focus=0, waist=1, wavelength=1)
    radius_scale = 1.6

    # make the beam
    zs = np.linspace(-8, 8, 100)
    rs = beam.radius(zs)
    fractions = np.linspace(-1, 1, len(colors) + 1) * radius_scale
    half = len(zs) // 2
    eps = -5e-2
    for c1, c2, lower, upper in zip(colors, colors[::-1], fractions[:-1], fractions[1:], strict=True):
        ax.fill_between(zs[:half], rs[:half] * lower - eps, rs[:half] * upper + eps, color=c1, ec="none")
        ax.fill_between(zs[half:], rs[half:] * lower - eps, rs[half:] * upper + eps, color=c2, ec="none")

    # make the corset
    corset_color = (0.9,) * 3 if dark else (0.1,) * 3
    corset_left, corset_right = -2.5, 2.5
    zs_corset = np.linspace(corset_left, corset_right, 50)
    rs_corset = beam.radius(zs_corset) * radius_scale
    ax.fill_between(zs_corset, -rs_corset, rs_corset, color=corset_color, lw=15)

    # make the laces
    lace_color = (0.2,) * 3 if dark else (0.8,) * 3
    n_laces = 3
    bounds = np.linspace(corset_left, corset_right, n_laces + 1)
    width = np.diff(bounds)[0] * 0.6
    positions = (bounds[:-1] + bounds[1:]) / 2
    for pos in positions:
        kwargs = {"color": lace_color, "lw": 6, "solid_capstyle": "round"}
        ax.plot([pos - width / 2, pos + width / 2], [-width / 2, width / 2], **kwargs)
        ax.plot([pos - width / 2, pos + width / 2], [width / 2, -width / 2], **kwargs)

    if crop is not None:
        ax.set_xlim(*crop[0])
        ax.set_ylim(*crop[1])

    fig.savefig(filename, transparent=True)


if __name__ == "__main__":
    # run with "pixi run -e dev python ./misc/logo/gen_logo.py"
    directory = Path(__file__).parent.resolve()
    make_logo(directory / "logo_light.svg", dark=False)
    make_logo(directory / "logo_dark.svg", dark=True)
    # only show the right expanding part for favicon
    make_logo(directory / "favicon.svg", dark=False, crop=([0, 8], [-4, 4]))
