from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from syneva.core.metadata import ColumnType, Metadata


@dataclass
class UtilityTask:
    target: str
    task_type: Literal["classification", "regression"] | None = None
    features: list[str] | None = None


def suggest_tasks(meta: Metadata) -> list[UtilityTask]:
    out: list[UtilityTask] = []
    for name, cm in meta.columns.items():
        if cm.dtype is ColumnType.ID:
            continue
        if cm.dtype is ColumnType.NUMERIC:
            out.append(UtilityTask(target=name, task_type="regression"))
        elif cm.dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            out.append(UtilityTask(target=name, task_type="classification"))
    return out
