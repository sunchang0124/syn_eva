from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def load_table(file: Any) -> pd.DataFrame:
    """Read an uploaded table by extension.

    `file` may be a path string/Path or a file-like object that carries a
    ``name`` attribute (e.g. a Streamlit UploadedFile). Both CSV and Parquet
    are supported; anything else raises ValueError.
    """
    name = getattr(file, "name", None) or str(file)
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    if suffix == ".parquet":
        return pd.read_parquet(file)
    raise ValueError(f"unsupported file type '{suffix}'; use .csv or .parquet")


def metadata_rows(meta: Metadata) -> list[dict]:
    """Flatten Metadata into editor rows (one dict per column)."""
    return [
        {"column": name, "dtype": cm.dtype.value, "sensitive": bool(cm.sensitive)}
        for name, cm in meta.columns.items()
    ]


def metadata_from_editor(rows: list[dict]) -> Metadata:
    """Rebuild Metadata from edited rows produced by `metadata_rows`."""
    cols = {
        r["column"]: ColumnMetadata(
            name=r["column"],
            dtype=ColumnType(r["dtype"]),
            sensitive=bool(r["sensitive"]),
        )
        for r in rows
    }
    return Metadata(columns=cols)
