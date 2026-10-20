from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import runnable_specs
from syneva.preservation.spec import SubgroupSpec
from syneva.utility._models import _build_preprocessor
from syneva.utility.task import UtilityTask, resolve_task_type


def _make_model(meta, features: list[str]) -> Pipeline:
    pre = _build_preprocessor(meta, features)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def _labels(y: pd.Series, dtype) -> pd.Series:
    # cast to the real target's dtype first, so int 1 and float 1.0 both become "1"
    with contextlib.suppress(TypeError, ValueError):
        y = y.astype(dtype)
    return y.astype(str)


@registry.register
@dataclass
class MinorityUtilityGap:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_utility_gap",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    tasks: list[UtilityTask] | None = None
    holdout: pd.DataFrame | None = None
    min_rows: int = 10
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        if not self.tasks:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "No utility tasks are configured, so there is no utility gap to compute."
                ),
            )
        class_tasks = []
        for t in self.tasks:
            if resolve_task_type(t, meta, real) == "classification":
                class_tasks.append(t)
            else:
                notes.append(f"task '{t.target}' skipped (regression not supported)")
        if not class_tasks or not specs:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No classification task and subgroup spec pair is configured.",
            )

        if self.holdout is not None:
            train_real, eval_real = real, self.holdout
        else:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(len(real))
            half = len(real) // 2
            train_real = real.iloc[perm[:half]].reset_index(drop=True)
            eval_real = real.iloc[perm[half:]].reset_index(drop=True)

        per_column: dict[str, dict[str, float]] = {}
        pair_scores: list[float] = []
        gaps_syn: list[float] = []
        gaps_real: list[float] = []
        excesses: list[float] = []
        for t in class_tasks:
            features = t.features or [
                n
                for n, cm in meta.columns.items()
                if n != t.target and cm.dtype is not ColumnType.ID
            ]
            target_dtype = train_real[t.target].dtype
            y_syn = _labels(synthetic[t.target], target_dtype)
            y_train_real = _labels(train_real[t.target], target_dtype)
            y_eval = _labels(eval_real[t.target], target_dtype)
            if y_syn.nunique() < 2 or y_train_real.nunique() < 2:
                notes.append(f"task '{t.target}' skipped (constant target in training data)")
                continue
            model_syn = _make_model(meta, features).fit(synthetic[features], y_syn)
            model_real = _make_model(meta, features).fit(train_real[features], y_train_real)
            pred_syn = model_syn.predict(eval_real[features])
            pred_real = model_real.predict(eval_real[features])
            overall_syn = balanced_accuracy_score(y_eval, pred_syn)
            overall_real = balanced_accuracy_score(y_eval, pred_real)
            for sp in specs:
                mask = sp.matches(eval_real).to_numpy()
                if int(mask.sum()) < self.min_rows:
                    notes.append(
                        f"pair '{sp.name}|{t.target}' skipped (<{self.min_rows} eval rows)"
                    )
                    continue
                if y_eval[mask].nunique() < 2:
                    notes.append(
                        f"pair '{sp.name}|{t.target}' skipped (constant target in subgroup)"
                    )
                    continue
                sub_syn = balanced_accuracy_score(y_eval[mask], pred_syn[mask])
                sub_real = balanced_accuracy_score(y_eval[mask], pred_real[mask])
                gap_syn = float(overall_syn - sub_syn)
                gap_real = float(overall_real - sub_real)
                excess = max(0.0, gap_syn - gap_real)
                per_column[f"{sp.name}|{t.target}"] = {
                    "gap_synthetic": gap_syn,
                    "gap_real": gap_real,
                    "excess_gap": excess,
                }
                gaps_syn.append(gap_syn)
                gaps_real.append(gap_real)
                excesses.append(excess)
                pair_scores.append(float(min(1.0, max(0.0, 1.0 - excess))))
        if not pair_scores:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "No (task, subgroup) pair had enough rows and a varying target to evaluate."
                ),
            )
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float(sum(pair_scores) / len(pair_scores)),
                "worst_excess_gap": max(excesses),
                "mean_gap_synthetic": sum(gaps_syn) / len(gaps_syn),
                "mean_gap_real": sum(gaps_real) / len(gaps_real),
            },
            per_column=per_column,
            notes=notes,
        )
