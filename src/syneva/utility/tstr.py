from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.utility._models import build_pipeline, select_features
from syneva.utility.task import UtilityTask, resolve_task_type

_TEST_SIZE = 0.3


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
    holdout: pd.DataFrame | None = None  # test set, or None

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
            tt = resolve_task_type(t, meta, real)
            if t.task_type is None:
                notes.append(f"task '{t.target}': task_type inferred as {tt}")
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
            trtr_score = _score(
                scorer,
                build_pipeline(tt, meta, features, self.random_state),
                X_real_tr,
                y_real_tr,
                X_test,
                y_test,
            )
            tstr_score = _score(
                scorer,
                build_pipeline(tt, meta, features, self.random_state),
                X_syn,
                y_syn,
                X_test,
                y_test,
            )
            scalars[f"trtr_{t.target}"] = float(trtr_score)
            scalars[f"tstr_{t.target}"] = float(tstr_score)
            per_task[t.target] = {"trtr": float(trtr_score), "tstr": float(tstr_score)}
            ratio = utility_ratio(trtr_score, tstr_score)
            if ratio is None:
                notes.append(_no_baseline_note(t.target, trtr_score))
                continue
            scalars[f"utility_ratio_{t.target}"] = ratio
            per_task[t.target]["ratio"] = ratio

        ratios = [v["ratio"] for v in per_task.values() if "ratio" in v]
        if not ratios:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No utility task could be evaluated.",
            )
        scalars["score"] = float(np.mean(ratios))
        return MetricResult(spec=self.spec, scalars=scalars, per_column=per_task, notes=notes)


def _stratify_labels(
    task_type: str, y: pd.Series, test_size: float, target: str, notes: list[str]
) -> pd.Series | None:
    """Labels to stratify the train/test split on, or None when stratifying is impossible.

    A stratified split needs at least 2 rows per class and at least one row per
    class on each side. Rare categories break that, so fall back to a plain
    random split and say so in the notes.
    """
    if task_type != "classification":
        return None
    counts = y.value_counts(dropna=False)
    n_test = math.ceil(test_size * len(y))
    n_train = len(y) - n_test
    if counts.min() < 2 or len(counts) > min(n_train, n_test):
        notes.append(
            f"task '{target}': {len(counts)} classes, smallest has {counts.min()} row(s); "
            "split without stratification"
        )
        return None
    return y


def utility_ratio(trtr: float, tstr: float) -> float | None:
    """TSTR / TRTR clipped to [0, 1], or None when TRTR <= 0.

    A ratio only means "fraction of real-data utility retained" when the
    real-trained model has positive skill. For R^2 <= 0 the quotient flips
    sign or explodes (-2.0 / -0.5 = 4.0), so the task is excluded instead.
    """
    if trtr <= 0:
        return None
    return float(min(1.0, max(0.0, tstr / trtr)))


def _no_baseline_note(label: str, trtr: float) -> str:
    return (
        f"task '{label}': TRTR score {trtr:.3g} <= 0, so the real-trained model has "
        "no skill to compare against; utility ratio undefined, excluded from score"
    )


@dataclass(frozen=True)
class Scorer:
    metric: Literal["roc_auc", "accuracy", "r2"]
    positive: object = None  # positive class label for roc_auc


def choose_scorer(task_type: str, y_real_train, y_test) -> Scorer:
    """Pick one metric from the real labels, so TRTR and TSTR share units.

    Binary targets use ROC-AUC on the sorted second label (sklearn's
    convention); multiclass targets, or a test split missing a class, use
    accuracy; regression uses R^2.
    """
    if task_type != "classification":
        return Scorer("r2")
    y_test = pd.Series(y_test)
    classes = sorted(pd.concat([pd.Series(y_real_train), y_test]).dropna().unique())
    if len(classes) == 2 and y_test.nunique() == 2:
        return Scorer("roc_auc", positive=classes[1])
    return Scorer("accuracy")


def _score(scorer: Scorer, pipe, X_train, y_train, X_test, y_test) -> float:
    train_classes = y_train.dropna().unique() if scorer.metric != "r2" else ()
    if len(train_classes) == 1:
        # A single-class training set yields a constant predictor; some
        # estimators (e.g. LogisticRegression) refuse to fit it at all.
        only = train_classes[0]
        if scorer.metric == "roc_auc":
            return float(roc_auc_score(y_test == scorer.positive, np.zeros(len(y_test))))
        return float(accuracy_score(y_test, np.full(len(y_test), only)))
    pipe.fit(X_train, y_train)
    if scorer.metric == "roc_auc":
        classes = list(pipe.classes_)
        if scorer.positive in classes:
            p = pipe.predict_proba(X_test)[:, classes.index(scorer.positive)]
        else:
            p = np.zeros(len(y_test))
        return float(roc_auc_score(y_test == scorer.positive, p))
    if scorer.metric == "accuracy":
        return float(accuracy_score(y_test, pipe.predict(X_test)))
    return float(r2_score(y_test, pipe.predict(X_test)))
