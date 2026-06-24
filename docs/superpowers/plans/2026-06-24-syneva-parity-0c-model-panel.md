# Model-Panel (Multi-Classifier) Utility (Phase 0c-d) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an extended-tier `model_panel_utility` metric that runs TSTR/TRTR across a panel of model families (linear, random forest, HistGBDT) and reports per-family ratios + their spread, leaving the core single-RF `tstr_suite` unchanged.

**Architecture:** A `build_panel_pipelines` factory in `_models.py` returns three named pipelines sharing one preprocessor; a new `ModelPanelUtility` dataclass metric loops tasks × models, reusing `tstr.py`'s `_score`; metric-info + registry wiring expose it. Behavior-preserving (golden green).

**Tech Stack:** Python 3.10+, scikit-learn, numpy, pandas, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-24-syneva-parity-0c-model-panel.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Behavior preservation:** `tstr_suite` and `build_pipeline`'s single-RF output must NOT change. Existing utility tests pass UNCHANGED; golden snapshot stays green (NEVER `--update-golden`). If an existing test or the golden changes, STOP and report.
- The metric-info guard tests (`tests/unit/core/test_metric_info.py`) require every registered metric to have a `METRIC_NAMES` (full name ≠ code), a `METRIC_INFO` description (NOT containing the phrases "higher is better"/"lower is better"), and a `RAW_HINT`.

## File structure
- Modify: `src/syneva/utility/_models.py` (extract `_build_preprocessor`, add `build_panel_pipelines`)
- Create: `src/syneva/utility/extended/model_panel.py` (the `ModelPanelUtility` metric)
- Modify: `src/syneva/utility/extended/__init__.py` (register the new module)
- Modify: `src/syneva/core/metric_info.py` (METRIC_INFO + RAW_HINT + METRIC_NAMES + _SCALAR_LABELS)
- Tests under `tests/unit/utility/`, `tests/integration/`.

---

## Task 1: `build_panel_pipelines` factory

**Files:** Modify `src/syneva/utility/_models.py`; Test `tests/unit/utility/test_model_panel_pipelines.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/utility/test_model_panel_pipelines.py
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from syneva.core.metadata import Metadata
from syneva.utility._models import build_panel_pipelines, select_features

_DF = pd.DataFrame(
    {"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"], "y": [0, 1, 0, 1]}
)


def _setup():
    meta = Metadata.infer(_DF)
    features = select_features(meta, "y", None)
    return meta, features


def test_panel_has_three_named_models_classification():
    meta, features = _setup()
    pipes = build_panel_pipelines("classification", meta, features, 42)
    assert set(pipes) == {"linear", "random_forest", "hist_gbdt"}
    for p in pipes.values():
        assert isinstance(p, Pipeline)
        assert "pre" in p.named_steps and "model" in p.named_steps


def test_panel_classification_estimator_types():
    meta, features = _setup()
    pipes = build_panel_pipelines("classification", meta, features, 42)
    assert isinstance(pipes["linear"].named_steps["model"], LogisticRegression)
    assert isinstance(pipes["random_forest"].named_steps["model"], RandomForestClassifier)
    assert isinstance(pipes["hist_gbdt"].named_steps["model"], HistGradientBoostingClassifier)


def test_panel_regression_estimator_types():
    meta, features = _setup()
    pipes = build_panel_pipelines("regression", meta, features, 42)
    assert isinstance(pipes["linear"].named_steps["model"], Ridge)
    assert isinstance(pipes["hist_gbdt"].named_steps["model"], HistGradientBoostingRegressor)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/utility/test_model_panel_pipelines.py --no-cov -v` (ImportError: build_panel_pipelines).

- [ ] **Step 3: Implement** — edit `src/syneva/utility/_models.py`. Add imports at the top alongside the existing sklearn imports:
```python
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
```
(The existing file already imports `RandomForestClassifier, RandomForestRegressor` from `sklearn.ensemble` — MERGE the new names into that existing import rather than duplicating; add the `sklearn.linear_model` line new.)

Extract the preprocessor into a helper and have `build_pipeline` call it (byte-identical ColumnTransformer — `tstr_suite` behavior unchanged), then add the panel factory:
```python
def _build_preprocessor(meta: Metadata, features: list[str]) -> ColumnTransformer:
    num = [f for f in features if meta.columns[f].dtype is ColumnType.NUMERIC]
    cat = [
        f for f in features if meta.columns[f].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(task_type: str, meta: Metadata, features: list[str], random_state: int):
    pre = _build_preprocessor(meta, features)
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    return Pipeline([("pre", pre), ("model", model)])


def build_panel_pipelines(
    task_type: str, meta: Metadata, features: list[str], random_state: int
) -> dict[str, Pipeline]:
    """Return {model_name: Pipeline} for the utility model panel. Each pipeline
    shares the same preprocessor as build_pipeline; only the final estimator differs."""
    if task_type == "classification":
        estimators = {
            "linear": LogisticRegression(max_iter=1000, random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            "hist_gbdt": HistGradientBoostingClassifier(random_state=random_state),
        }
    else:
        estimators = {
            "linear": Ridge(random_state=random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            "hist_gbdt": HistGradientBoostingRegressor(random_state=random_state),
        }
    return {
        name: Pipeline([("pre", _build_preprocessor(meta, features)), ("model", est)])
        for name, est in estimators.items()
    }
```
(`build_pipeline`'s resulting Pipeline is structurally identical to before — same step names, same estimator, same params — so `tstr_suite` is unchanged. Keep `select_features` as-is.)

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/utility/test_model_panel_pipelines.py --no-cov -v` (4 passed).

- [ ] **Step 5: Guard existing utility behavior** — `uv run pytest tests/unit/utility/ --no-cov -v` (existing TSTR/multi_target tests pass UNCHANGED, proving `build_pipeline` extraction is behavior-identical).

- [ ] **Step 6: Commit** — `git commit -m "feat(utility): build_panel_pipelines (linear/RF/HistGBDT factory)"`

---

## Task 2: `ModelPanelUtility` metric + wiring

**Files:** Create `src/syneva/utility/extended/model_panel.py`; Modify `src/syneva/utility/extended/__init__.py`, `src/syneva/core/metric_info.py`; Test `tests/unit/utility/test_model_panel.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/utility/test_model_panel.py
import numpy as np

from syneva.utility.extended.model_panel import ModelPanelUtility
from syneva.utility.task import UtilityTask


def test_panel_reports_per_model_and_spread(real_df, syn_good_df, metadata):
    m = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")]
    )
    r = m.compute(real_df, syn_good_df, metadata)
    for key in (
        "score",
        "utility_ratio_mean",
        "ratio_linear",
        "ratio_random_forest",
        "ratio_hist_gbdt",
        "ratio_spread",
    ):
        assert key in r.scalars
        assert np.isfinite(r.scalars[key])
    assert 0.0 <= r.scalars["score"] <= 1.0
    # per_column has a <task>/<model> entry per model
    assert "high_income/linear" in r.per_column
    assert "high_income/hist_gbdt" in r.per_column


def test_panel_identical_data_ratios_near_one(real_df, metadata):
    m = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")]
    )
    r = m.compute(real_df, real_df, metadata)  # synthetic == real
    assert r.scalars["score"] > 0.8
    assert r.scalars["ratio_spread"] < 0.3


def test_panel_no_tasks_returns_unit_score(real_df, syn_good_df, metadata):
    m = ModelPanelUtility(tasks=[])
    r = m.compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] == 1.0


def test_panel_runs_with_holdout(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=4)
    m = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")],
        holdout=holdout,
    )
    r = m.compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["score"] <= 1.0
```

(Confirm fixtures `real_df`/`syn_good_df`/`metadata` and target `"high_income"` from `tests/conftest.py`.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/utility/test_model_panel.py --no-cov -v` (ModuleNotFoundError).

- [ ] **Step 3: Implement the metric** — create `src/syneva/utility/extended/model_panel.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
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
    holdout: object | None = None

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
            # Fresh (unfitted) pipelines per fit: build the dict twice.
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
                scalars={"score": 1.0},
                notes=[*notes, "no runnable utility tasks"],
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
        return MetricResult(
            spec=self.spec, scalars=scalars, per_column=per_column, notes=notes
        )
```

- [ ] **Step 4: Register the module** — in `src/syneva/utility/extended/__init__.py`, append (matching the existing side-effect import style):
```python
from syneva.utility.extended import model_panel as model_panel  # side-effect: registers it
```

- [ ] **Step 5: metric_info wiring** — edit `src/syneva/core/metric_info.py`:

(a) In the `METRIC_INFO` dict (utility section, near the `"multi_target_utility"` entry ~line 65), add:
```python
    "model_panel_utility": "Trains linear, random-forest, and gradient-boosting models on the synthetic data and on the real data, then compares how well each performs on a real test set; reports the average ratio and the per-family breakdown.",
```
(MUST NOT contain the phrases "higher is better"/"lower is better".)

(b) In `RAW_HINT` (near `"multi_target_utility"` ~line 106), add:
```python
    "model_panel_utility": "Average utility ratio across model families, around 1 when synthetic is just as useful as real. Higher is better.",
```

(c) In `METRIC_NAMES` (near `"multi_target_utility"` ~line 147), add:
```python
    "model_panel_utility": "Model-panel utility",
```

(d) In `_SCALAR_LABELS` (~line 155-187), add:
```python
    "utility_ratio_mean": "Mean utility ratio",
    "ratio_linear": "Utility ratio — linear model",
    "ratio_random_forest": "Utility ratio — random forest",
    "ratio_hist_gbdt": "Utility ratio — gradient boosting",
    "ratio_spread": "Spread across model families (std)",
```

- [ ] **Step 6: Run, expect pass** — `uv run pytest tests/unit/utility/test_model_panel.py tests/unit/core/test_metric_info.py --no-cov -v` (4 metric tests + metric_info guards pass).

- [ ] **Step 7: Commit** — `git commit -m "feat(utility): model_panel_utility metric + metric-info wiring"`

---

## Task 3: integration, parity manifest, full-suite gate

**Files:** Test `tests/integration/test_model_panel.py`; possibly Modify `tests/parity/test_syntheval_parity.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_model_panel.py
import syneva
from syneva.utility.task import UtilityTask


def test_model_panel_runs_via_evaluate(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df, syn_good_df, metadata,
        tiers=("core", "extended"),
        run_utility=True,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "model_panel_utility" in by
    assert by["model_panel_utility"].error is None
    assert "ratio_spread" in by["model_panel_utility"].scalars


def test_model_panel_runs_via_full_preset(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df, syn_good_df, metadata,
        preset="full",
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    names = {r.spec.name for r in rep.results}
    assert "model_panel_utility" in names
```

- [ ] **Step 2: Run, expect failure first** — `uv run pytest tests/integration/test_model_panel.py --no-cov -v`. (It may already pass once Task 2 is merged, since the metric registers and the runner injects `tasks`. If it passes immediately, that's acceptable — this task is primarily the integration + parity + full-suite gate; still confirm both tests pass.)

- [ ] **Step 3: Parity manifest** — inspect `tests/parity/test_syntheval_parity.py`. If the `SYNTHEVAL_PARITY` manifest has an entry for multi-model / multiple-classifier utility that is currently unimplemented, mark it implemented and point it at `model_panel_utility` (match the manifest's exact entry shape). If there is NO such entry, make NO change and note it in the commit message. Do not invent manifest entries.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/integration/test_model_panel.py tests/parity/test_syntheval_parity.py --no-cov -v` (all pass).

- [ ] **Step 5: FULL SUITE (behavior-preservation gate)** — `uv run pytest -q`. ALL pass (2 pre-existing skips allowed); golden snapshot UNCHANGED (NEVER `--update-golden`); coverage >= 85%. If the golden fails, STOP and report BLOCKED — a no-change path regressed.

- [ ] **Step 6: Commit** — `git commit -m "test(utility): model_panel integration + parity manifest"`

---

## Self-review notes
- **Spec coverage:** build_panel_pipelines factory (T1) · ModelPanelUtility metric with per-model ratios + spread + holdout + no-task guard (T2) · registration + metric_info (METRIC_NAMES/METRIC_INFO/RAW_HINT/_SCALAR_LABELS) (T2) · integration + preset + parity manifest + golden gate (T3). Behavior preservation: `build_pipeline` extraction is structurally identical (T1 Step 5 proves it) and `tstr_suite` is untouched; golden checked full-suite in T3.
- **Type/signature consistency:** `build_panel_pipelines(task_type, meta, features, random_state)` matches `build_pipeline`'s signature; metric uses the same `_score(tt, pipe, X_tr, y_tr, X_te, y_te)` contract as `TSTRSuite`; `_MODELS` tuple keys match the `ratio_<name>` scalar keys and `per_model_mean` dict keys; `ModelPanelUtility` fields (`tasks`/`random_state`/`holdout`) match what the runner's `_instantiate` injects.
- **No placeholders:** every step has complete code; the parity-manifest step explicitly says make no change if no entry exists.
- **Direction-claim guard:** the METRIC_INFO description avoids the banned phrases; the direction claim lives only in RAW_HINT, as the guard test requires.
