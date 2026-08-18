from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, runtime_checkable

import pandas as pd

from syneva.core.errors import MetricError
from syneva.core.metadata import Metadata
from syneva.core.plot import PlotPayload  # forward; created in step 4

C = Literal[
    "congruence",
    "coverage",
    "compliance",
    "utility",
    "fairness",
    "preservation",
    "constraint",
    "completeness",
    "comprehension",
    "consistency",
]
Tier = Literal["core", "extended", "custom"]
DataType = Literal["static", "longitudinal", "relational"]
Scope = Literal["per-column", "pairwise", "table-level"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    c: C
    tier: Tier
    data_types: frozenset[DataType]
    requires_real: bool
    scope: Scope


@dataclass
class MetricResult:
    spec: MetricSpec
    scalars: dict[str, float] | None = None
    per_column: dict[str, dict[str, float]] | None = None
    plot_payload: PlotPayload | None = None
    notes: list[str] = field(default_factory=list)
    error: MetricError | None = None


@runtime_checkable
class Metric(Protocol):
    spec: ClassVar[MetricSpec]

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult: ...
