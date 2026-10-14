from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata, _infer_column_type


@dataclass
class UtilityTask:
    target: str
    task_type: Literal["classification", "regression"] | None = None
    features: list[str] | None = None


def resolve_task_type(
    task: UtilityTask, meta: Metadata, data: pd.DataFrame
) -> Literal["classification", "regression"]:
    """Return the task's declared type, or infer it from the target column.

    The target's metadata dtype decides; a target the metadata does not cover
    falls back to the dtype inferred from ``data``.
    """
    if task.task_type is not None:
        return task.task_type
    cm = meta.columns.get(task.target)
    dtype = cm.dtype if cm is not None else _infer_column_type(data, task.target)
    if dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
        return "classification"
    return "regression"


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
