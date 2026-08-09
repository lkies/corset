"""Utilities for formatting values with units in plots and tables."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import override

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


class FormattedDataFrame(pd.DataFrame):
    """A subclass of pandas DataFrame that allows for custom formatting of columns when displayed in HTML for Jupyter notebooks."""

    _metadata = ["_formatters"]  # noqa: RUF012
    _formatters: dict[str, Callable]

    def __init__(
        self,
        *args,
        formatters: dict[str, Callable],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._formatters = formatters

    @override
    def to_html(self, buf: None = None, *, notebook: bool = False, **kwargs) -> str:
        """Render the DataFrame as an HTML table with the specified formatters applied.

        See :meth:`pandas.DataFrame.to_html` for details on the parameters.

        Args:
            buf: Buffer to write the HTML to. If None, returns the HTML as a string.
            notebook: Whether to format the HTML for display in a Jupyter notebook.
            **kwargs: Additional keyword arguments passed to :meth:`pandas.DataFrame.to_html`.

        Returns:
            The HTML representation of the DataFrame as a string.
        """
        return super().to_html(buf=buf, formatters=self._formatters, **({"na_rep": ""} | kwargs), notebook=notebook)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def _repr_html_(self):
        return self.to_html(notebook=True)


@dataclass(frozen=True)
class Unit:
    """Unit for displaying values in plots and tables.

    Can also be used as unit literals for multiplication with values,
    e.g. ``2 * Units.Length.MILLIMETER == 0.002``.
    """

    unicode: str  #: ASCII representation of the unit
    tex: str  #: LaTeX representation of the unit
    factor: float  #: Conversion factor to the SI base unit
    decimals: int  #: Default number of decimals to display when formatting values with this unit

    @cached_property
    def dollar_tex(self) -> str:
        """LaTeX representation of the unit wrapped in dollar signs for use in Matplotlib labels."""
        return f"${self.tex}$" if self.tex else ""

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(
            unicode=f"{self.unicode}/{other.unicode}",
            tex=f"{self.tex}/{other.tex}",
            factor=self.factor / other.factor,
            decimals=max(self.decimals - other.decimals + 3, 0),
        )

    def format(
        self,
        value: float | list[float] | np.ndarray | None,
        *,
        dev: float | None = None,
        decimals: int | None = None,
        tex: bool = False,
        array_threshold: int = 5,
    ) -> str:
        """Format a value with this unit.

        Args:
            value: The value to format.
            dev: The deviation of the value.
            decimals: The number of decimal places to display.
            tex: Whether to use the LaTeX representation of the unit.
            array_threshold: The threshold for displaying arrays. If the number of elements in the array exceeds this threshold, it will be truncated.

        Returns:
            A string representation of the value with the unit.
        """
        decimals = decimals if decimals is not None else self.decimals

        if dev is not None and not isinstance(value, (int, float)):
            raise ValueError("Deviation can only be specified for scalar values.")

        raw_str = f"\\,{self.tex}" if tex else self.unicode
        unit_str = f" {raw_str}" if self.unicode else ""  # handle case where unit is empty string

        if value is None or (np.isscalar(value) and np.isnan(value)):
            return ""
        elif isinstance(value, (int, float)):
            value_str = f"{value / self.factor:.{decimals}f}"
            pm = "\\pm" if tex else "±"
            uncertainty_str = f" {pm} {dev / self.factor:.{decimals}f}" if dev is not None else ""
            number_str = f"{value_str}{uncertainty_str}"
            number_str = f"({number_str})" if dev is not None and unit_str else number_str
            return f"{number_str}{unit_str}"
        elif isinstance(value, np.ndarray):
            return f"{np.array2string(value / self.factor, precision=decimals, separator=', ', threshold=array_threshold)}{unit_str}"
        elif isinstance(value, list):
            return f"{np.array2string(np.array(value) / self.factor, precision=decimals, separator=', ', threshold=array_threshold)}{unit_str}"
        else:
            raise TypeError(f"Unsupported type {type(value)} for formatting with unit {self.unicode}")

    def __mul__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other * self.factor

    def __rmul__(self, other: float | np.ndarray) -> float | np.ndarray:
        return other * self.factor

    def axis_formatter(self, decimals: int | None = None, include_unit: bool = False) -> FuncFormatter:
        """Create a matplotlib axis formatter for this unit.

        Args:
            decimals: The number of decimal places to display. If None, use the default number of decimals for this unit.
            include_unit: Whether to include the unit in the formatted string.

        Returns:
            A matplotlib FuncFormatter that formats axis ticks with this unit.
        """
        decimals = decimals if decimals is not None else self.decimals
        unit_string = self.dollar_tex if include_unit and self.unicode else ""
        return FuncFormatter(lambda x, _: f"{x / self.factor:.{decimals}f}{unit_string}")


class FractionUnit(Unit):
    """Unit for fractional quantities, e.g. overlap or axis coupling."""

    pass


class SensitivityUnit(Unit):
    """Unit for sensitivity analysis, e.g. overlap sensitivity per unit length squared."""

    pass


class LengthUnit(Unit):
    """Unit for length quantities, e.g. coordinates along the beam or beam radii."""

    pass


class Units:
    """Namespace for units used in the package."""

    class Fraction:
        """Units for overlap."""

        PERCENT = FractionUnit(unicode="%", tex=r"\%", factor=1e-2, decimals=1)  #: Overlap and coupling in percent
        UNITY = FractionUnit(unicode="", tex="", factor=1, decimals=3)
        """#: Overlap and coupling as a fraction, i.e. unitless"""

    class Sensitivity:
        """Units for sensitivity analysis."""

        PER_METER2 = SensitivityUnit(unicode="%/m²", tex=r"\%/\mathrm{m}^2", factor=1, decimals=0)  #:
        PERCENT_PER_MILLIMETER2 = SensitivityUnit(
            unicode="%/mm²", tex=r"\%/\mathrm{mm}^2", factor=1e-2 / (1e-3**2), decimals=0
        )  #:
        PERCENT_PER_CENTIMETER2 = SensitivityUnit(
            unicode="%/cm²", tex=r"\%/\mathrm{cm}^2", factor=1e-2 / (1e-2**2), decimals=2
        )  #:

    class Length:
        """Units for length."""

        NANOMETER = LengthUnit(unicode="nm", tex=r"\mathrm{nm}", factor=1e-9, decimals=0)  #:
        MICROMETER = LengthUnit(unicode="μm", tex=r"\mathrm{\mu m}", factor=1e-6, decimals=0)  #:
        MILLIMETER = LengthUnit(unicode="mm", tex=r"\mathrm{mm}", factor=1e-3, decimals=0)  #:
        CENTIMETER = LengthUnit(unicode="cm", tex=r"\mathrm{cm}", factor=1e-2, decimals=1)  #:
        METER = LengthUnit(unicode="m", tex=r"\mathrm{m}", factor=1, decimals=3)  #:

    pct = Fraction.PERCENT  #: Alias for :attr:`Units.Fraction.PERCENT`
    ul = Fraction.UNITY  #: Alias for :attr:`Units.Fraction.UNITY`

    per_m2 = Sensitivity.PER_METER2  #: Alias for :attr:`Units.Sensitivity.PER_METER2`
    pct_per_mm2 = Sensitivity.PERCENT_PER_MILLIMETER2  #: Alias for :attr:`Units.Sensitivity.PERCENT_PER_MILLIMETER2`
    pct_per_cm2 = Sensitivity.PERCENT_PER_CENTIMETER2  #: Alias for :attr:`Units.Sensitivity.PERCENT_PER_CENTIMETER2`

    nm = Length.NANOMETER  #: Alias for :attr:`Units.Length.NANOMETER`
    um = Length.MICROMETER  #: Alias for :attr:`Units.Length.MICROMETER`
    mm = Length.MILLIMETER  #: Alias for :attr:`Units.Length.MILLIMETER`
    cm = Length.CENTIMETER  #: Alias for :attr:`Units.Length.CENTIMETER`
    m = Length.METER  #: Alias for :attr:`Units.Length.METER`
