from __future__ import annotations

from collections import defaultdict
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
    subgroup_specs=None,
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
            subgroup_specs=subgroup_specs,
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

    def _normalized_per_metric(self, normalization: str) -> dict[str, dict[str, float]]:
        from syneva.benchmark.normalize import normalize_across

        sm = self.score_matrix
        out: dict[str, dict[str, float]] = {}
        for m in self.metrics():
            present = [c for c in self.reports if m in sm[c]]
            vals = [sm[c][m] for c in present]
            normed = normalize_across(vals, normalization)
            out[m] = dict(zip(present, normed, strict=True))
        return out

    def _candidate_key(
        self, candidate: str, metric_names: list[str], nm: dict[str, dict[str, float]]
    ) -> float:
        cmap = self._metric_c_map()
        by_c: dict[str, list[float]] = defaultdict(list)
        for m in metric_names:
            v = nm.get(m, {}).get(candidate)
            if v is not None:
                by_c[cmap[m]].append(v)
        if not by_c:
            return float("-inf")
        c_means = [sum(vs) / len(vs) for vs in by_c.values()]
        return sum(c_means) / len(c_means)

    def ranking(
        self, normalization: str = "absolute", by: str = "overall"
    ) -> list[tuple[int, str, float]]:
        cands = list(self.reports)
        if (
            len(cands) < 2
            and normalization != "absolute"
            and "single-candidate cohort: relative normalization is degenerate" not in self.notes
        ):
            self.notes.append("single-candidate cohort: relative normalization is degenerate")
        nm = self._normalized_per_metric(normalization)
        cmap = self._metric_c_map()
        if by == "overall":
            metric_names = self.metrics()
        elif by in self.c_dims():
            metric_names = [m for m in self.metrics() if cmap.get(m) == by]
        elif by in self.metrics():
            metric_names = [by]
        else:
            raise SynevaError(
                f"unknown ranking target by='{by}'; use 'overall', a C dimension, or a metric name"
            )
        keys = {c: self._candidate_key(c, metric_names, nm) for c in cands}
        order = sorted(cands, key=lambda c: (-keys[c], c))
        return [(i + 1, c, keys[c]) for i, c in enumerate(order)]

    def to_dict(self) -> dict:
        return {
            "reports": {name: rep.to_dict() for name, rep in self.reports.items()},
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(
            reports={name: Report.from_dict(r) for name, r in d["reports"].items()},
            random_state=d.get("random_state", 42),
        )

    def to_json(self, path) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def to_html(self, path, *, normalization: str = "absolute", interactive: bool = False) -> None:
        from pathlib import Path

        from syneva.render.html.benchmark_renderer import render_leaderboard

        Path(path).write_text(
            render_leaderboard(self, normalization=normalization, interactive=interactive)
        )

    @classmethod
    def from_json(cls, path) -> BenchmarkResult:
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))
