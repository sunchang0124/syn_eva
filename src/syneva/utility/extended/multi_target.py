from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility.task import suggest_tasks
from syneva.utility.tstr import TSTRSuite


@registry.register
@dataclass
class MultiTargetUtility:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="multi_target_utility",
        c="utility",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        tasks = suggest_tasks(meta)
        suite = TSTRSuite(tasks=tasks, random_state=self.random_state)
        inner = suite.compute(real, synthetic, meta)
        return MetricResult(
            spec=self.spec,
            scalars={"score": inner.scalars.get("score", 1.0)},
            per_column=inner.per_column,
            notes=inner.notes,
        )
