from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_pipeline, select_features
from syneva.utility.task import UtilityTask


@registry.register
@dataclass
class FeatureImportanceCorrelation:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="feature_importance_spearman",
        c="utility",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tasks: list[UtilityTask] = field(default_factory=list)
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_task: dict[str, dict[str, float]] = {}

        for t in self.tasks:
            if t.target not in real.columns:
                notes.append(f"task '{t.target}' missing from real; skipped")
                continue
            tt = t.task_type or (
                "classification"
                if meta.columns[t.target].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
                else "regression"
            )
            features = select_features(meta, t.target, t.features)
            pipe_real = build_pipeline(tt, meta, features, self.random_state)
            pipe_syn = build_pipeline(tt, meta, features, self.random_state)
            pipe_real.fit(real[features], real[t.target])
            pipe_syn.fit(synthetic[features], synthetic[t.target])
            fi_real = pipe_real.named_steps["model"].feature_importances_
            fi_syn = pipe_syn.named_steps["model"].feature_importances_
            n = min(len(fi_real), len(fi_syn))
            if n < 2:
                notes.append(f"task '{t.target}' has fewer than 2 features; skipped")
                continue
            rho = float(spearmanr(fi_real[:n], fi_syn[:n]).correlation)  # pyright: ignore[reportAttributeAccessIssue]  # scipy result types are untyped
            if np.isnan(rho):
                # a constant importance vector (e.g. a constant target) has no ranking
                notes.append(f"task '{t.target}': importance ranking undefined; rho set to 0")
                rho = 0.0
            per_task[t.target] = {"spearman_rho": rho, "n_features": float(len(features))}

        if not per_task:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No utility task could be evaluated.",
            )
        rho = float(np.mean([v["spearman_rho"] for v in per_task.values()]))
        score = float((rho + 1) / 2)  # map [-1, 1] -> [0, 1]
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "spearman_rho": rho},
            per_column=per_task,
            notes=notes,
        )
