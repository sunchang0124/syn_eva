from __future__ import annotations

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata

_NAN_SENTINEL = "§NA§"


def _row_tuples(df: pd.DataFrame) -> list[tuple]:
    """Convert DataFrame rows to tuples with NaN replaced by a fixed sentinel.

    Using a sentinel ensures that NaN-containing rows compare equal when they
    are verbatim copies, which is required for correct duplicate detection.
    """
    return [tuple(_NAN_SENTINEL if pd.isna(v) else v for v in row) for row in df.values.tolist()]


def comparable_row_keys(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    meta: Metadata,
) -> tuple[list[tuple], list[tuple]]:
    """Return hashable row keys for exact-match comparison of real vs synthetic.

    Columns are taken in ``real``'s order (so a reordered synthetic table still
    lines up), ``ColumnType.ID`` columns are dropped (unique identifiers would
    make every copied row look novel), and NaN is normalized so NaN-containing
    copies compare equal. If every column is an ID column, all columns are kept.
    """
    cols = [
        c
        for c in real.columns
        if c not in meta.columns or meta.columns[c].dtype is not ColumnType.ID
    ]
    if not cols:
        cols = list(real.columns)
    return _row_tuples(real[cols]), _row_tuples(synthetic[cols])
