from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata


@dataclass
class SubgroupSpec:
    """Declares a subgroup of rows for the preservation metrics: a display name
    and AND-ed per-column conditions — a list of allowed values for a
    categorical/boolean column, or an inclusive ``(lo, hi)`` range tuple for a
    numeric column (``None`` = open end)."""

    name: str
    conditions: dict[str, list | tuple]

    def validate(self, meta: Metadata) -> str | None:
        """Return a problem description, or None when the spec is runnable."""
        for col, cond in self.conditions.items():
            if col not in meta.columns:
                return f"unknown column '{col}'"
            dtype = meta.columns[col].dtype
            if isinstance(cond, tuple):
                if dtype is not ColumnType.NUMERIC:
                    return f"range condition on non-numeric column '{col}'"
                if len(cond) != 2:
                    return f"range for '{col}' must be a (lo, hi) tuple"
            elif isinstance(cond, list):
                if dtype is ColumnType.NUMERIC:
                    return f"values list on numeric column '{col}'"
            else:
                return f"condition for '{col}' must be a list or a (lo, hi) tuple"
        return None

    def matches(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask (aligned to df.index) of rows in the subgroup."""
        mask = pd.Series(True, index=df.index)
        for col, cond in self.conditions.items():
            if isinstance(cond, tuple):
                v = pd.to_numeric(df[col], errors="coerce")
                lo, hi = cond
                if lo is not None:
                    mask &= v >= lo
                if hi is not None:
                    mask &= v <= hi
            else:
                mask &= df[col].isin(cond)
        return mask
