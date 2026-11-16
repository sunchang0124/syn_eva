"""Skipping numeric columns that are constant in the real data.

Metrics that standardise by the real standard deviation have nothing to divide
by when the real column has no spread, so the standardised difference is
undefined there and the column is left out with a note (#39).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantRealColumns:
    """Collects the columns a metric skips because they are constant in the real data."""

    def __init__(self, notes: list[str]) -> None:
        self._notes = notes
        self.names: list[str] = []

    def skip(self, name: str, real_values: np.ndarray | pd.Series) -> bool:
        """Record ``name`` and add a note if ``real_values`` are all equal.

        Compares min and max exactly: the std of a constant float column can
        come out as ~1e-17 rather than 0.
        """
        if real_values.min() != real_values.max():
            return False
        self._notes.append(f"column '{name}' skipped (constant in the real data)")
        self.names.append(name)
        return True

    def skip_reason(self, fallback: str) -> str:
        """Why no column was left: names the constant columns if there were any."""
        if not self.names:
            return fallback
        names = ", ".join(f"'{n}'" for n in self.names)
        return f"No numeric column can be compared; skipped as constant in the real data: {names}."
