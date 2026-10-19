from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.stats import spearmanr

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import (
    build_pipeline,
    encoded_features,
    select_features,
    source_columns,
)
from syneva.utility.task import UtilityTask, resolve_task_type


def _per_feature_importance(pipe, features: list[str]) -> np.ndarray:
    """Mean decrease in impurity per original feature, in ``features`` order.

    Each side's encoder learns its own category set, so one-hot columns do not line
    up between real and synthetic. A split on a one-hot column is a split on its
    source feature, so summing those columns' importances gives the feature's total
    impurity decrease, which is the MDI definition applied to the original variable.
    Features the model never saw (dropped as all-missing) score 0.

    References
    ----------
    Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). Understanding variable
    importances in forests of randomized trees. Advances in Neural Information
    Processing Systems 26, 431-439.
    """
    totals = dict.fromkeys(features, 0.0)
    src = source_columns(pipe.named_steps["pre"])
    for col, imp in zip(src, pipe.named_steps["model"].feature_importances_, strict=True):
        totals[col] += float(imp)
    return np.array([totals[f] for f in features])


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
            tt = resolve_task_type(t, meta, real)
            # rank only features the model can use; dropped dtypes would add tied zeros
            features = encoded_features(meta, select_features(meta, t.target, t.features))
            if len(features) < 2:
                notes.append(f"task '{t.target}' has fewer than 2 usable features; skipped")
                continue
            pipe_real = build_pipeline(tt, meta, features, self.random_state)
            pipe_syn = build_pipeline(tt, meta, features, self.random_state)
            pipe_real.fit(real[features], real[t.target])
            pipe_syn.fit(synthetic[features], synthetic[t.target])
            fi_real = _per_feature_importance(pipe_real, features)
            fi_syn = _per_feature_importance(pipe_syn, features)
            rho = float(spearmanr(fi_real, fi_syn).correlation)  # pyright: ignore[reportAttributeAccessIssue]  # scipy result types are untyped
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
