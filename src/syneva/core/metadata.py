from __future__ import annotations

import re
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from syneva.core.errors import MetadataError


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"


_ID_PATTERN = re.compile(r"^(id|.*_id|.*_key)$", re.IGNORECASE)


@dataclass
class ColumnMetadata:
    name: str
    dtype: ColumnType
    # Privacy roles (Sweeney, 2002; Machanavajjhala et al., 2007). A quasi-identifier is
    # an attribute an attacker can link on; a sensitive attribute is one whose disclosure
    # is the harm. They are independent: a column may be both.
    sensitive: bool = False
    quasi_identifier: bool = False
    categories: list[str] | None = None
    value_range: tuple[float, float] | None = None


@dataclass
class Metadata:
    columns: dict[str, ColumnMetadata]
    primary_key: str | None = None

    @classmethod
    def infer(cls, df: pd.DataFrame) -> Metadata:
        cols: dict[str, ColumnMetadata] = {}
        for name in df.columns:
            cols[name] = ColumnMetadata(name=name, dtype=_infer_column_type(df, name))
        return cls(columns=cols)

    def override(self, **patches: dict[str, Any]) -> Metadata:
        new = {k: ColumnMetadata(**vars(v)) for k, v in self.columns.items()}
        for col, patch in patches.items():
            if col not in new:
                raise MetadataError(f"unknown column for override: {col}")
            for k, v in patch.items():
                if k == "dtype" and isinstance(v, str):
                    v = ColumnType(v)
                setattr(new[col], k, v)
        return Metadata(columns=new, primary_key=self.primary_key)

    def validate_against(self, df: pd.DataFrame) -> None:
        for name, meta in self.columns.items():
            if name not in df.columns:
                raise MetadataError(f"metadata declares '{name}' but DataFrame lacks it")
            actual = _infer_column_type(df, name)
            if meta.dtype is ColumnType.ID:
                continue
            if meta.dtype is ColumnType.NUMERIC and actual is not ColumnType.NUMERIC:
                raise MetadataError(f"column '{name}' declared {meta.dtype} but is {actual}")
            if meta.dtype is ColumnType.CATEGORICAL and actual is ColumnType.NUMERIC:
                # numeric → categorical is allowed (coercion), the reverse is not
                continue
        if self.primary_key and self.primary_key not in df.columns:
            raise MetadataError(f"primary_key '{self.primary_key}' not in DataFrame")


def _infer_column_type(df: pd.DataFrame, name: Hashable) -> ColumnType:
    # Labels needn't be strings: pd.DataFrame(ndarray) labels its columns 0, 1, ...
    if _ID_PATTERN.match(str(name)):
        return ColumnType.ID
    s = df[name]
    if is_bool_dtype(s):
        return ColumnType.BOOLEAN
    if is_datetime64_any_dtype(s):
        return ColumnType.DATETIME
    if is_numeric_dtype(s):
        return ColumnType.NUMERIC
    return ColumnType.CATEGORICAL
