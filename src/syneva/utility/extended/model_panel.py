from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_panel_pipelines, select_features
from syneva.utility.task import UtilityTask
from syneva.utility.tstr import _score

_MODELS = ("linear", "random_forest", "hist_gbdt")


@registry.register
@dataclass
class ModelPanelUtility:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="model_panel_utility",
        c="utility",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tasks: list[UtilityTask] = field(default_factory=list)
    random_state: int = 42
    holdout: pd.DataFrame | None = None

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        ratios_by_model: dict[str, list[float]] = {m: [] for m in _MODELS}

        for t in self.tasks:
            if t.target not in real.columns:
                notes.append(f"task '{t.target}' missing from real; skipped")
                continue
            features = select_features(meta, t.target, t.features)
            tt = t.task_type or "regression"
            X_syn = synthetic[features]
            y_syn = synthetic[t.target]
            if self.holdout is not None:
                X_test = self.holdout[features]
                y_test = self.holdout[t.target]
                X_real_tr, y_real_tr = real[features], real[t.target]
            else:
                X_real_tr, X_test, y_real_tr, y_test = train_test_split(
                    real[features],
                    real[t.target],
                    test_size=0.3,
                    random_state=self.random_state,
                    stratify=real[t.target] if t.task_type == "classification" else None,
                )
            trtr_pipes = build_panel_pipelines(tt, meta, features, self.random_state)
            tstr_pipes = build_panel_pipelines(tt, meta, features, self.random_state)
            for name in _MODELS:
                trtr = _score(tt, trtr_pipes[name], X_real_tr, y_real_tr, X_test, y_test)
                tstr = _score(tt, tstr_pipes[name], X_syn, y_syn, X_test, y_test)
                ratio = float(tstr / trtr) if trtr not in (0, 0.0) else 0.0
                ratio = float(min(1.0, max(0.0, ratio)))
                ratios_by_model[name].append(ratio)
                per_column[f"{t.target}/{name}"] = {
                    "trtr": float(trtr),
                    "tstr": float(tstr),
                    "ratio": ratio,
                }

        all_ratios = [r for rs in ratios_by_model.values() for r in rs]
        if not all_ratios:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No utility task could be evaluated.",
            )
        per_model_mean = {
            m: (float(np.mean(rs)) if rs else 0.0) for m, rs in ratios_by_model.items()
        }
        score = float(min(1.0, max(0.0, float(np.mean(all_ratios)))))
        scalars = {
            "score": score,
            "utility_ratio_mean": score,
            "ratio_linear": per_model_mean["linear"],
            "ratio_random_forest": per_model_mean["random_forest"],
            "ratio_hist_gbdt": per_model_mean["hist_gbdt"],
            "ratio_spread": float(np.std(list(per_model_mean.values()))),
        }
        return MetricResult(spec=self.spec, scalars=scalars, per_column=per_column, notes=notes)
