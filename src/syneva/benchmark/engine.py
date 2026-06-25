from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.core.presets import _UNSET
from syneva.core.report import Report, _normalize_scalars
from syneva.core.runner import evaluate


def benchmark(
    real: pd.DataFrame | None,
    candidates: dict[str, pd.DataFrame],
    metadata: Metadata | None = None,
    *,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    tiers=_UNSET,
    cs=_UNSET,
    run_utility=_UNSET,
    utility_tasks=None,
    run_fairness=_UNSET,
    fairness_specs=None,
    distance=_UNSET,
    random_state: int = 42,
) -> BenchmarkResult:
    if not candidates:
        raise SynevaError("benchmark needs at least one candidate")
    ref_cols = (
        set(real.columns) if real is not None else set(next(iter(candidates.values())).columns)
    )
    for name, df in candidates.items():
        if set(df.columns) != ref_cols:
            raise SynevaError(f"candidate '{name}' columns must match real")
    reports: dict[str, Report] = {}
    for name, df in candidates.items():
        reports[name] = evaluate(
            real,
            df,
            metadata,
            holdout=holdout,
            preset=preset,
            tiers=tiers,
            cs=cs,
            run_utility=run_utility,
            utility_tasks=utility_tasks,
            run_fairness=run_fairness,
            fairness_specs=fairness_specs,
            distance=distance,
            random_state=random_state,
        )
    return BenchmarkResult(reports=reports, random_state=random_state)


@dataclass
class BenchmarkResult:
    reports: dict[str, Report]
    random_state: int = 42
    notes: list[str] = field(default_factory=list)

    @property
    def score_matrix(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, rep in self.reports.items():
            row: dict[str, float] = {}
            for r in rep.results:
                if r.error is None and r.scalars is not None:
                    row[r.spec.name] = _normalize_scalars(r)
            out[name] = row
        return out

    @property
    def c_scores(self) -> dict[str, dict[str, float]]:
        return {name: rep.aggregated for name, rep in self.reports.items()}

    @property
    def overall(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, cs in self.c_scores.items():
            out[name] = (sum(cs.values()) / len(cs)) if cs else 0.0
        return out

    def metrics(self) -> list[str]:
        names: set[str] = set()
        for row in self.score_matrix.values():
            names.update(row)
        return sorted(names)

    def c_dims(self) -> list[str]:
        dims: set[str] = set()
        for cs in self.c_scores.values():
            dims.update(cs)
        return sorted(dims)

    def _metric_c_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rep in self.reports.values():
            for r in rep.results:
                out[r.spec.name] = r.spec.c
        return out
