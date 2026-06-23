from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_pipeline, select_features
from syneva.utility.task import UtilityTask


@registry.register
@dataclass
class TSTRSuite:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="tstr_suite",
        c="utility",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tasks: list[UtilityTask] = field(default_factory=list)
    random_state: int = 42
    holdout: object | None = None  # pandas DataFrame test set, or None

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        scalars: dict[str, float] = {}
        notes: list[str] = []
        per_task: dict[str, dict[str, float]] = {}

        for t in self.tasks:
            if t.target not in real.columns:
                notes.append(f"task '{t.target}' missing from real; skipped")
                continue
            features = select_features(meta, t.target, t.features)
            X_syn = synthetic[features]
            y_syn = synthetic[t.target]
            tt = t.task_type or "regression"
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
            trtr_score = _score(
                tt,
                build_pipeline(tt, meta, features, self.random_state),
                X_real_tr,
                y_real_tr,
                X_test,
                y_test,
            )
            tstr_score = _score(
                tt,
                build_pipeline(tt, meta, features, self.random_state),
                X_syn,
                y_syn,
                X_test,
                y_test,
            )
            ratio = float(tstr_score / trtr_score) if trtr_score not in (0, 0.0) else 0.0
            scalars[f"trtr_{t.target}"] = float(trtr_score)
            scalars[f"tstr_{t.target}"] = float(tstr_score)
            scalars[f"utility_ratio_{t.target}"] = float(min(1.0, max(0.0, ratio)))
            per_task[t.target] = {
                "trtr": float(trtr_score),
                "tstr": float(tstr_score),
                "ratio": float(ratio),
            }

        if not per_task:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable utility tasks"],
            )
        score = float(np.mean([v["ratio"] for v in per_task.values()]))
        scalars["score"] = min(1.0, max(0.0, score))
        return MetricResult(spec=self.spec, scalars=scalars, per_column=per_task, notes=notes)


def _score(task_type: str, pipe, X_train, y_train, X_test, y_test) -> float:
    pipe.fit(X_train, y_train)
    if task_type == "classification":
        proba_classes = getattr(pipe, "classes_", None)
        try:
            y_proba = pipe.predict_proba(X_test)
            if proba_classes is not None and len(proba_classes) == 2:
                return float(roc_auc_score(y_test, y_proba[:, 1]))
            return float(accuracy_score(y_test, pipe.predict(X_test)))
        except Exception:
            return float(accuracy_score(y_test, pipe.predict(X_test)))
    return float(r2_score(y_test, pipe.predict(X_test)))
