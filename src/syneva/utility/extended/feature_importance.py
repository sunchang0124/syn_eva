from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from scipy.stats import spearmanr

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_pipeline, select_features


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
    target: str = ""
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        if not self.target or self.target not in real.columns:
            return MetricResult(
                spec=self.spec,
                skip_reason="No target column is set, or it is missing from the real data.",
            )
        tt = (
            "classification"
            if meta.columns[self.target].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
            else "regression"
        )
        features = select_features(meta, self.target, None)
        pipe_real = build_pipeline(tt, meta, features, self.random_state)
        pipe_syn = build_pipeline(tt, meta, features, self.random_state)
        pipe_real.fit(real[features], real[self.target])
        pipe_syn.fit(synthetic[features], synthetic[self.target])
        fi_real = pipe_real.named_steps["model"].feature_importances_
        fi_syn = pipe_syn.named_steps["model"].feature_importances_
        n = min(len(fi_real), len(fi_syn))
        if n < 2:
            return MetricResult(
                spec=self.spec,
                skip_reason="Feature-importance correlation needs at least 2 features.",
            )
        rho = float(spearmanr(fi_real[:n], fi_syn[:n]).correlation or 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # scipy result types are untyped
        score = float((rho + 1) / 2)  # map [-1, 1] -> [0, 1]
        return MetricResult(spec=self.spec, scalars={"score": score, "spearman_rho": rho})
