from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from syneva.core.metadata import ColumnType, Metadata


@dataclass
class UtilityTask:
    target: str
    task_type: Literal["classification", "regression"] | None = None
    features: list[str] | None = None


def resolve_task_type(task: UtilityTask, meta: Metadata) -> Literal["classification", "regression"]:
    """Return the task's declared type, or infer it from the target column's metadata."""
    if task.task_type is not None:
        return task.task_type
    if meta.columns[task.target].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
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
