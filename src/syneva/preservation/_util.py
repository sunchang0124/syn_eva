from __future__ import annotations

from syneva.core.metadata import ColumnType, Metadata
from syneva.preservation.spec import SubgroupSpec

_EPS = 1e-12


def categorical_columns(meta: Metadata) -> list[str]:
    return [
        n
        for n, cm in meta.columns.items()
        if cm.dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]


def numeric_columns(meta: Metadata) -> list[str]:
    return [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]


def runnable_specs(
    specs: list[SubgroupSpec], meta: Metadata
) -> tuple[list[SubgroupSpec], list[str]]:
    """Split specs into runnable ones and a note per rejected spec."""
    ok: list[SubgroupSpec] = []
    notes: list[str] = []
    for sp in specs:
        problem = sp.validate(meta)
        if problem is None:
            ok.append(sp)
        else:
            notes.append(f"subgroup '{sp.name}' skipped ({problem})")
    return ok, notes
