"""Module defining a LensList class for managing collections of lenses."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import cast, overload

import numpy as np
import pandas as pd

from corset.serialize import YamlSerializableMixin

from .core import Lens, ThickLens, ThinLens


@dataclass(frozen=True)
class LensList(YamlSerializableMixin):
    """A list of lenses with convenient access methods.

    Supports (array) indexing and other list-like operations, indices can also be
    strings or list of strings to access lenses by name.
    Implements :meth:`_repr_html_` to show a :class:`pandas.DataFrame` representation in IPython environments.
    """

    lenses: list[Lens]  #: Underlying list of lenses

    def __post_init__(self):
        self._by_name  # trigger validation  # noqa: B018

    def __iter__(self) -> Iterator[Lens]:
        return iter(self.lenses)

    def __len__(self) -> int:
        return len(self.lenses)

    @overload
    def __getitem__(self, index: int | str) -> Lens:
        pass

    @overload
    def __getitem__(self, index: slice | list[int] | list[str]) -> "LensList":
        pass

    def __getitem__(self, index: int | str | slice | list[int] | list[str]) -> Lens | "LensList":
        if isinstance(index, int):
            return self.lenses[index]
        elif isinstance(index, str):
            return self._by_name[index]
        elif isinstance(index, list) and len(index) > 0 and all(isinstance(i, str) for i in index):
            return LensList(lenses=[self._by_name[i] for i in index])  # pyright: ignore[reportArgumentType]
        elif isinstance(index, slice) or (isinstance(index, list) and all(isinstance(i, int) for i in index)):
            return LensList(lenses=np.array(self.lenses)[index].tolist())
        else:
            raise TypeError(f"Invalid index {index}")

    def concatenate(*lists: Iterable[Lens]) -> "LensList":
        """Concatenate multiple lens lists or lists of lenses.

        Args:
            lists: LensList or list of Lens instances to concatenate.

        Returns:
            LensList: The concatenated lens list.
        """
        return LensList(lenses=list(chain(*lists)))

    def __add__(self, other: Iterable[Lens]) -> "LensList":
        return LensList.concatenate(self, other)

    def __radd__(self, other: Iterable[Lens]) -> "LensList":
        return LensList.concatenate(other, self)

    @cached_property
    def _by_name(self) -> dict[str, Lens]:
        names = [lens.name for lens in self.lenses if lens.name]
        if len(names) != len(set(names)):
            raise ValueError("Lens names must be unique.")
        return {lens.name: lens for lens in self.lenses if lens.name}

    @cached_property
    def df(self) -> pd.DataFrame:
        """A :class:`pandas.DataFrame` representation of the lens list."""
        df = pd.DataFrame(
            [
                {
                    "name": lens.name,
                    "type": "thin" if not isinstance(lens, ThickLens) else "thick",
                    "focal_length": lens.focal_length,
                    "left_margin": lens.left_margin,
                    "right_margin": lens.right_margin,
                    "in_roc": vars(lens).get("in_roc"),
                    "out_roc": vars(lens).get("out_roc"),
                    "thickness": vars(lens).get("thickness"),
                    "refractive_index": vars(lens).get("refractive_index"),
                    "lens": lens,
                }
                for lens in self.lenses
            ]
        )
        if all(df["type"] == "thin"):
            df = df.drop(columns=["in_roc", "out_roc", "thickness", "refractive_index"])
        return df

    def _repr_html_(self) -> str:
        return self.df.to_html(notebook=True)

    def save_csv(self, path: str | Path) -> None:
        """Save the lens list to a CSV file.

        Args:
            path: Path to the CSV file. This is passed directly to :func:`pandas.DataFrame.to_csv` so it also
                accepts other types supported by that function like file-like objects.
        """
        self.df.drop(columns=["lens"]).to_csv(path, index=False)

    @classmethod
    def load_csv(cls, path: str | Path) -> "LensList":
        """Load a lens list from a CSV file.

        Args:
            path: Path to the CSV file. This is passed directly to :func:`pandas.read_csv` so it also
                accepts other types supported by that function like URL strings or file-like objects.

        Returns:
            LensList: The loaded lens list.

        Raises:
            ValueError: If the CSV file contains invalid data.
            KeyError: If required fields are missing in the CSV file.
        """
        df = pd.read_csv(path)
        lenses: list[Lens] = []
        for i, row in df.iterrows():
            line = cast(int, i) + 2
            lens_cls = {"thin": ThinLens, "thick": ThickLens}.get(row["type"])
            if lens_cls is None:
                raise ValueError(f"Unknown lens type: {row['type']}")
            try:
                kwargs = {f.name: row[f.name] for f in fields(lens_cls)}
                kwargs["name"] = kwargs["name"] if pd.notna(kwargs["name"]) else None
                lens: Lens = lens_cls(**kwargs)  # type: ignore[arg-type]
                lenses.append(lens)
            except ValueError as e:
                raise ValueError(f"Error creating lens from line {line}: {e}") from e
            except KeyError as e:
                raise ValueError(f"Missing required field '{e.args[0]}' for lens from line {line}") from e
            if lens_cls is ThickLens and not np.isclose(lens.focal_length, row.get("focal_length", lens.focal_length)):
                raise ValueError(f"If specified, thick lens focal length must match computed value (line {line})")
        return cls(lenses=lenses)

    @classmethod
    def available(cls) -> list[str]:
        """List available built-in lens lists.

        Returns:
            list[str]: Names of available lens lists.
        """
        raise NotImplementedError("Built-in lens library not implemented yet.")

    @classmethod
    def load(cls, name: str) -> "LensList":
        """Load a lens list from the built-in library.

        Args:
            name: Name of the lens list to load.

        Returns:
            LensList: The loaded lens list.

        Raises:
            ValueError: If the lens list name is not found in the library.
        """
        raise NotImplementedError("Built-in lens library not implemented yet.")
