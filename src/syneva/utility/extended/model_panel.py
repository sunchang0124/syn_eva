from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_panel_pipelines, select_features
from syneva.utility.task import UtilityTask, resolve_task_type
from syneva.utility.tstr import (
    _TEST_SIZE,
    _no_baseline_note,
    _score,
    _stratify_labels,
    choose_scorer,
    utility_ratio,
)

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
            features = select_features(meta, t.target, t.features, notes)
            if not features:
                notes.append(f"task '{t.target}' has no usable features; skipped")
                continue
            tt = resolve_task_type(t, meta, real)
            if t.task_type is None:
                notes.append(f"task '{t.target}': task_type inferred as {tt}")
            X_syn = synthetic[features]
            y_syn = synthetic[t.target]
            if self.holdout is not None:
                X_test = self.holdout[features]
                y_test = self.holdout[t.target]
                X_real_tr, y_real_tr = real[features], real[t.target]
            else:
                stratify = _stratify_labels(tt, real[t.target], _TEST_SIZE, t.target, notes)
                X_real_tr, X_test, y_real_tr, y_test = train_test_split(
                    real[features],
                    real[t.target],
                    test_size=_TEST_SIZE,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            scorer = choose_scorer(tt, y_real_tr, y_test)
            trtr_pipes = build_panel_pipelines(tt, meta, features, self.random_state)
            tstr_pipes = build_panel_pipelines(tt, meta, features, self.random_state)
            for name in _MODELS:
                trtr = _score(scorer, trtr_pipes[name], X_real_tr, y_real_tr, X_test, y_test)
                tstr = _score(scorer, tstr_pipes[name], X_syn, y_syn, X_test, y_test)
                key = f"{t.target}/{name}"
                per_column[key] = {"trtr": float(trtr), "tstr": float(tstr)}
                ratio = utility_ratio(trtr, tstr)
                if ratio is None:
                    notes.append(_no_baseline_note(key, trtr))
                    continue
                ratios_by_model[name].append(ratio)
                per_column[key]["ratio"] = ratio

        all_ratios = [r for rs in ratios_by_model.values() for r in rs]
        if not all_ratios:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No utility task could be evaluated.",
            )
        per_model_mean = {m: float(np.mean(rs)) for m, rs in ratios_by_model.items() if rs}
        score = float(np.mean(all_ratios))
        scalars = {
            "score": score,
            "utility_ratio_mean": score,
            **{f"ratio_{m}": v for m, v in per_model_mean.items()},
            "ratio_spread": float(np.std(list(per_model_mean.values()))),
        }
        return MetricResult(spec=self.spec, scalars=scalars, per_column=per_column, notes=notes)
