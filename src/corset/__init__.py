from importlib.metadata import PackageNotFoundError, version

from .config import Config  # noqa: F401
from .core import Beam, OpticalSetup, ThickLens, ThinLens  # noqa: F401
from .database import LensList  # noqa: F401
from .serialize import load_yaml, save_yaml  # noqa: F401
from .solver import (  # noqa: F401
    Aperture,
    ModeMatchingSolution,
    Passage,
    ShiftingRange,
    SolutionList,
    mode_match,
)

try:
    __version__ = version("corset")
except PackageNotFoundError:
    __version__ = "dev"
